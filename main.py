import sys
import torch
import torch.nn as nn
import random
import math
from torch.nn import functional as F
from datasets import load_dataset
from dotenv import load_dotenv

# Invocation guard
if __name__ != "__main__":
    sys.exit(-1)

# Hyperparameters
d_model = 512
d_hid=4*d_model
max_len = 600
batch_size = 30
n_head = 4
d_head = d_model // n_head
n_stack = 2
p_dropout = 0.1

assert d_model % n_head == 0
# ---------------

def detect_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

load_dotenv()

device = detect_device()
print(f'device: {device}')

#############################
# Load and initialise dataset
ds = load_dataset("Helsinki-NLP/opus_books", "en-it")['train']

it_full, en_full = [], []
for x in ds['translation']:
    # Cap max length to eliminate outliers
    if len(x['it']) > max_len or len(x['en']) > max_len:
        continue
    it_full.append(x['it'])
    en_full.append(x['en'])

############
# Vocabulary
pad_token = '@'
bos_token = '^'
eos_token = '%'

vocab = set("".join(it_full + en_full))

# Adding extra tokens
vocab.add(pad_token)
vocab.add(bos_token)
vocab.add(eos_token)

vocab = sorted(list(vocab))
vocab_size = len(vocab)

print(f'vocab_size: {vocab_size}')

#################
# Encode / Decode
stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

pad_token_idx = stoi[pad_token]
bos_token_idx = stoi[bos_token]
eos_token_idx = stoi[eos_token]

######################
# Split train and eval
split_index = int(0.9 * len(it_full))

it_train = [[stoi[c] for c in x] for x in it_full[:split_index]]
en_train = [[stoi[c] for c in x] for x in en_full[:split_index]]
it_eval = [[stoi[c] for c in x] for x in it_full[split_index:]]
en_eval = [[stoi[c] for c in x] for x in en_full[split_index:]]

print(f'it train length: {len(it_train)}')
print(f'en train length: {len(en_train)}')
print(f'it eval length: {len(it_eval)}')
print(f'en eval length: {len(en_eval)}')

##################
# Batch generation

# Sorting on en as that's the translation target.
#
# This ends up leaving a wide gap between min / max length in the it batch entries,
# but after looking at the length distribution of both it and en datasets it seems
# inevitable, and it's either a question of having it happening on the it or en side.
#
# Sorting on en to begin with, if it becomes a problem I'll revisit.
en_train_sorted_idx = sorted(range(len(en_train)), key=lambda i: len(en_train[i]))
en_eval_sorted_idx = sorted(range(len(en_train)), key=lambda i: len(en_train[i]))

def pad(ls):
    return nn.utils.rnn.pad_sequence(ls, batch_first=True, padding_value=pad_token_idx)

def generate_batch(dataset):
    it_data = it_train if dataset == 'train' else it_eval
    en_data = en_train if dataset == 'train' else en_eval
    sorted_idx = en_train_sorted_idx if dataset == 'train' else en_eval_sorted_idx

    idx_start = random.randint(0, len(it_data) - batch_size)
    idxs = sorted_idx[idx_start:idx_start+batch_size]

    it_x = pad([torch.tensor(it_data[x]) for x in idxs])

    bos = torch.tensor([bos_token_idx])
    eos = torch.tensor([eos_token_idx])
    en_seqs = [torch.tensor(en_data[x]) for x in idxs]
    en_x = pad([torch.cat([bos, s]) for s in en_seqs])
    en_y = pad([torch.cat([s, eos]) for s in en_seqs])

    return it_x, en_x, en_y

##################
# Model definition

class PositionalEncoding(nn.Module):

    def __init__(self):
        super().__init__()
        # Pre-generate lookup table
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class AttentionHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.q_wei = nn.Linear(d_model, d_head, bias=False)
        self.k_wei = nn.Linear(d_model, d_head, bias=False)
        self.v_wei = nn.Linear(d_model, d_head, bias=False)
        self.register_buffer('causal_mask', torch.tril(torch.ones(max_len, max_len)) == 0)

    def forward(self, q_x, kv_x, pad_mask, apply_causal_mask=False):
        q = self.q_wei(q_x)
        k = self.k_wei(kv_x)
        v = self.v_wei(kv_x)

        out = (q @ k.transpose(-1, -2))/d_head**0.5
        out = out.masked_fill(pad_mask, float('-inf'))
        if apply_causal_mask:
            q_len = q_x.shape[1]
            k_len = kv_x.shape[1]
            out = out.masked_fill(self.causal_mask[:q_len,:k_len], float('-inf'))
        out = F.softmax(out, dim=-1)
        out = out @ v

        return out


class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList(AttentionHead() for i in range(n_head)) # Can likely optimise by running in parallel as a single big matrix
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, q_x, kv_x, pad_mask, apply_causal_mask=False):
        out = [head(q_x, kv_x, pad_mask, apply_causal_mask) for head in self.heads]
        out = torch.cat(out, dim=-1)
        out = self.linear(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_hid)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(d_hid, d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.dropout(out)
        return out


class EncoderStack(nn.Module):

    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ln1 = nn.LayerNorm(d_model)
        self.ffw = FeedForward()
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask):
        out = self.ln1(x + self.attn(x, x, pad_mask))
        out = self.ln2(out + self.ffw(out))
        return out


class DecoderStack(nn.Module):

    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ln1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention()
        self.ln2 = nn.LayerNorm(d_model)
        self.ffw = FeedForward()
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, trs_x, pad_mask_trs_x, src_out, pad_mask_src_x):
        out = self.ln1(trs_x + self.attn(trs_x, trs_x, pad_mask_trs_x, apply_causal_mask=True))
        out = self.ln2(out + self.cross_attn(trs_x, src_out, pad_mask_src_x))
        out = self.ln3(out + self.ffw(out))
        return out


class AttentionReplica(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb_table = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding()

        self.encoder = nn.ModuleList([EncoderStack() for _ in range(n_stack)])
        self.enc_dropout = nn.Dropout(p_dropout)

        self.decoder = nn.ModuleList([DecoderStack() for _ in range(n_stack)])
        self.dec_dropout = nn.Dropout(p_dropout)

        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src_x, trs_x):

        pad_mask_src_x = (src_x == pad_token_idx).unsqueeze(-2)
        pad_mask_trs_x = (trs_x == pad_token_idx).unsqueeze(-2)

        # Encoder
        src_out = self.emb_table(src_x)
        src_out = self.pos_enc(src_out)
        src_out = self.enc_dropout(src_out)
        for encoderStack in self.encoder:
            src_out = encoderStack(src_out, pad_mask_src_x)

        trs_out = self.emb_table(trs_x)
        trs_out = self.pos_enc(trs_out)
        trs_out = self.dec_dropout(trs_out)
        for decoderStack in self.decoder:
            trs_out = decoderStack(trs_out, pad_mask_trs_x, src_out, pad_mask_src_x)

        trs_out = self.linear(trs_out)
        return trs_out

#################
# Tran / Generate

m = AttentionReplica()

it_x, en_x, en_y = generate_batch('train')

logits = m(it_x, en_x)

print(logits.shape)

# Must remember to ignore padding in loss!!!

sys.exit(0)