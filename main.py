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
max_len = 600
batch_size = 30
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
padding_token = '@'
bos_token = '^'
eos_token = '%'

vocab = set("".join(it_full + en_full))

# Adding extra tokens
vocab.add(padding_token)
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

padding_token_idx = stoi[padding_token]
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
    return nn.utils.rnn.pad_sequence(ls, batch_first=True, padding_value=padding_token_idx)

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
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_hid: int):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_hid)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(d_hid, d_model)

    def forward(self, input):
        out = self.lin1(input)
        out = self.relu(out)
        out = self.lin2(out)
        return out

class AttentionReplica(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb_table = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding()

    def forward(self, src_x, trs_x):
        src_out = self.emb_table(src_x)
        src_out = self.pos_enc(src_out)
        return None

m = AttentionReplica()

it_x, en_x, en_y = generate_batch('train')

m(it_x, en_x)


sys.exit(0)