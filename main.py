import os
# Must be set before torch initialises the CUDA allocator. expandable_segments
# lets the caching allocator grow/shrink segments instead of reserving a fixed
# block per tensor size, which avoids fragmentation blowups from our
# variable-length batches and autoregressive generation (reserved memory then
# tracks live usage instead of pinning most of the card). setdefault so an
# explicit PYTORCH_CUDA_ALLOC_CONF from the environment still wins.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
import torch
import torch.nn as nn
import random
import math
import time
import sacrebleu
from datetime import datetime
from torch.nn import functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import WordPiece as WordPieceDecoder

# Invocation guard
if __name__ != "__main__":
    sys.exit(-1)

# Hyperparameters
d_model = 512
d_hid = 4 * d_model
max_len = 600
batch_size = 128
n_head = 8
d_head = d_model // n_head
n_stack = 6
p_dropout = 0.1

# Dataset + shared WordPiece tokenizer (trained once on the train split, cached
# on disk; the filename embeds the dataset/vocab so a new corpus forces a retrain)
dataset_name = "Helsinki-NLP/opus-100"
target_vocab_size = 16000
tokenizer_path = f'tokenizer-opus100-{target_vocab_size}.json'

training_steps = 100_000
warmup_steps = 4_000

eval_interval = 5_000
eval_iters = 50

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
# Load and initialise dataset (opus-100 ships official train/validation/test)
raw = load_dataset(dataset_name, "en-it")

# Drop empty/whitespace pairs and cap length to eliminate outliers. For training
# we also drop badly misaligned pairs via a source/target char length-ratio
# filter (opus-100 carries some OPUS noise); the eval split keeps only the length
# cap so BLEU stays comparable to a standard benchmark.
def clean_split(split, apply_ratio):
    it_out, en_out = [], []
    for x in split['translation']:
        it, en = x['it'].strip(), x['en'].strip()
        if not it or not en:
            continue
        if len(it) > max_len or len(en) > max_len:
            continue
        if apply_ratio and not (0.5 <= len(it) / len(en) <= 2.0):
            continue
        it_out.append(it)
        en_out.append(en)
    return it_out, en_out

it_train_txt, en_train_txt = clean_split(raw['train'], apply_ratio=True)
it_eval_txt, en_eval_txt = clean_split(raw['validation'], apply_ratio=False)

############
# Vocabulary
pad_token = '[PAD]'
bos_token = '[BOS]'
eos_token = '[EOS]'
unk_token = '[UNK]'

# Train a shared WordPiece tokenizer over the combined it+en TRAIN text (as in
# the attention paper), caching it on disk so we only pay the training cost once.
# Train split only, so val/test never leak into the vocabulary.
if os.path.exists(tokenizer_path):
    print(f'loading tokenizer from {tokenizer_path}')
    tokenizer = Tokenizer.from_file(tokenizer_path)
else:
    print('training tokenizer')
    tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = WordPieceDecoder()  # merge ## continuations back into words on decode
    trainer = WordPieceTrainer(
        vocab_size=target_vocab_size,
        special_tokens=[pad_token, bos_token, eos_token, unk_token],
    )
    tokenizer.train_from_iterator(it_train_txt + en_train_txt, trainer)
    tokenizer.save(tokenizer_path)

vocab_size = tokenizer.get_vocab_size()
print(f'vocab_size: {vocab_size}')

#################
# Encode / Decode
encode = lambda s: tokenizer.encode(s).ids
decode = lambda ids: tokenizer.decode(ids)
# encode_batch runs multithreaded in Rust — far faster than a Python loop over
# the ~1M-sentence train split.
encode_all = lambda texts: [e.ids for e in tokenizer.encode_batch(texts)]

pad_token_idx = tokenizer.token_to_id(pad_token)
bos_token_idx = tokenizer.token_to_id(bos_token)
eos_token_idx = tokenizer.token_to_id(eos_token)

######################
# Encode train and eval (official opus-100 splits)
it_train = encode_all(it_train_txt)
en_train = encode_all(en_train_txt)
it_eval = encode_all(it_eval_txt)
en_eval = encode_all(en_eval_txt)

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
en_eval_sorted_idx = sorted(range(len(en_eval)), key=lambda i: len(en_eval[i]))

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

    return it_x.to(device), en_x.to(device), en_y.to(device)

##################
# Model definition

class PositionalEncoding(nn.Module):

    def __init__(self):
        super().__init__()
        # Pre-generate lookup table
        pe_len = max_len + 1 # Accounts for adding bos / eos
        pe = torch.zeros(pe_len, d_model)
        pos = torch.arange(pe_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)   # (1, max_len + 1, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()
        # All heads fused into single projections (d_model -> n_head * d_head == d_model).
        # Same parameter count as one bias-free Linear per head, but a single matmul.
        self.q_wei = nn.Linear(d_model, d_model, bias=False)
        self.k_wei = nn.Linear(d_model, d_model, bias=False)
        self.v_wei = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p_dropout)
        mask_len = max_len + 1 # Accounts for adding bos / eos
        self.register_buffer('causal_mask', torch.tril(torch.ones(mask_len, mask_len)) == 0)

    def forward(self, q_x, kv_x, pad_mask, apply_causal_mask=False):
        B, T_q, _ = q_x.shape
        T_kv = kv_x.shape[1]

        # Project then split into heads: (B, n_head, T, d_head)
        q = self.q_wei(q_x).view(B, T_q, n_head, d_head).transpose(1, 2)
        k = self.k_wei(kv_x).view(B, T_kv, n_head, d_head).transpose(1, 2)
        v = self.v_wei(kv_x).view(B, T_kv, n_head, d_head).transpose(1, 2)

        # Build an additive mask (-inf where masked) broadcastable to (B, n_head, T_q, T_kv).
        # pad_mask is (B, 1, T_kv); add a head dim so it broadcasts over heads and queries.
        mask = pad_mask.unsqueeze(1)                          # (B, 1, 1, T_kv)
        if apply_causal_mask:
            mask = mask | self.causal_mask[:T_q, :T_kv]       # broadcast (T_q, T_kv)
        attn_mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(mask, float('-inf'))

        # scaled_dot_product_attention applies the 1/sqrt(d_head) scaling internally and
        # uses a memory-efficient kernel that never materialises the full T_q x T_kv matrix.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T_q, d_model)    # re-merge heads

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
        # Pre-LN: normalise the input to each sublayer, keep the residual stream clean.
        normed = self.ln1(x)
        out = x + self.attn(normed, normed, pad_mask)
        out = out + self.ffw(self.ln2(out))
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
        # Pre-LN: normalise each sublayer's input. src_out is the encoder memory,
        # already normalised by the encoder's final LN, so it's used as-is for cross-attn k/v.
        normed = self.ln1(trs_x)
        out = trs_x + self.attn(normed, normed, pad_mask_trs_x, apply_causal_mask=True)
        out = out + self.cross_attn(self.ln2(out), src_out, pad_mask_src_x)
        out = out + self.ffw(self.ln3(out))
        return out


class AttentionReplica(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb_table = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_idx)
        self.pos_enc = PositionalEncoding()

        self.encoder = nn.ModuleList([EncoderStack() for _ in range(n_stack)])
        self.enc_dropout = nn.Dropout(p_dropout)
        self.enc_norm = nn.LayerNorm(d_model)  # Pre-LN: normalise encoder memory before it feeds cross-attn.

        self.decoder = nn.ModuleList([DecoderStack() for _ in range(n_stack)])
        self.dec_dropout = nn.Dropout(p_dropout)
        self.dec_norm = nn.LayerNorm(d_model)  # Pre-LN: normalise decoder output before the projection.

        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src_x, trs_x):

        pad_mask_src_x = (src_x == pad_token_idx).unsqueeze(-2)
        pad_mask_trs_x = (trs_x == pad_token_idx).unsqueeze(-2)

        # Encoder
        src_out = self.emb_table(src_x) * math.sqrt(d_model)
        src_out = self.pos_enc(src_out)
        src_out = self.enc_dropout(src_out)
        for encoderStack in self.encoder:
            src_out = encoderStack(src_out, pad_mask_src_x)
        src_out = self.enc_norm(src_out)

        trs_out = self.emb_table(trs_x) * math.sqrt(d_model)
        trs_out = self.pos_enc(trs_out)
        trs_out = self.dec_dropout(trs_out)
        for decoderStack in self.decoder:
            trs_out = decoderStack(trs_out, pad_mask_trs_x, src_out, pad_mask_src_x)
        trs_out = self.dec_norm(trs_out)

        trs_out = self.linear(trs_out)
        return trs_out

##################
# Train / Generate

m = AttentionReplica().to(device)

ps = (p for p in m.parameters() if p.requires_grad)
n_params = sum(p.numel() for p in ps)
print(f'number of parameters: {n_params/1e6:.2f}M')

def lr_lambda(step):
    step = max(step, 1)
    return d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)

optimizer = torch.optim.AdamW(m.parameters(), lr=1.0, betas=(0.9, 0.98))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def calculate_loss(logits, expected):
    B, T, E = logits.shape
    loss = F.cross_entropy(logits.view(B*T, E), expected.view(B*T), ignore_index=pad_token_idx, label_smoothing=0.1)
    return loss

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            it_x, en_x, en_y = generate_batch(split)
            logits = m(it_x, en_x)
            loss = calculate_loss(logits, en_y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

@torch.no_grad()
def generate(src_x, max_new_tokens=max_len + 1):
    was_training = m.training
    m.eval()

    B = src_x.shape[0]
    src_x = src_x.to(device)
    trs = torch.full((B, 1), bos_token_idx, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        logits = m(src_x, trs)                 # (B, T, vocab)
        next_tok = logits[:, -1].argmax(-1)    # (B,), Add switch between argmax and multinomial so that we can use this function for BLEU and normal generation!!!
        # once a sequence has emitted <eos>, keep it padded
        next_tok = torch.where(finished, torch.full_like(next_tok, pad_token_idx), next_tok)
        trs = torch.cat([trs, next_tok.unsqueeze(1)], dim=1)
        finished |= (next_tok == eos_token_idx)
        if finished.all():
            break

    if was_training:
        m.train()
    return trs   # (B, T), includes leading <bos>

def ids_to_text(ids):
    ids = [int(i) for i in ids]
    # Cut at the first <eos> so we don't decode trailing padding, then let the
    # tokenizer drop the remaining special tokens and merge the wordpieces.
    if eos_token_idx in ids:
        ids = ids[:ids.index(eos_token_idx)]
    return tokenizer.decode(ids)

@torch.no_grad()
def estimate_bleu(n_sentences=512, max_print=None):
    hyps, refs, srcs = [], [], []
    # Walk the length-bucketed eval set to minimise padding, like generate_batch
    # Not reusing generate_batch to ensure that sentences don't overlap between batches,
    # and to ensure we're using the same sample every time (reproducibility).
    n_sentences = min(n_sentences, len(en_eval_sorted_idx))
    stride = len(en_eval_sorted_idx) / n_sentences
    sample_idx = [en_eval_sorted_idx[int(i * stride)] for i in range(n_sentences)]
    for start in range(0, min(n_sentences, len(it_eval)), batch_size):
        idxs = sample_idx[start:start + batch_size]
        src_x = pad([torch.tensor(it_eval[i]) for i in idxs]).to(device)

        out = generate(src_x)
        hyps.extend(ids_to_text(row) for row in out)
        refs.extend(decode(en_eval[i]) for i in idxs)
        srcs.extend(decode(it_eval[i]) for i in idxs)

    samples = list(zip(hyps, refs, srcs))
    if max_print is not None:
        samples = samples[:max_print]
    for h, r, s in samples:
        print(f'  HYP: {h!r}\n  REF: {r!r}\n  SRC: {s!r}\n')
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu.score

def now():
    if device == 'cuda':
        torch.cuda.synchronize()
    return time.perf_counter()

print('training')
m.train()
total_start = now()
train_start = now()
for iter in range(training_steps):
    if iter % eval_interval == 0:
        train_elapsed = now() - train_start

        t = now(); losses = estimate_loss();             loss_elapsed = now() - t
        t = now(); bleu = estimate_bleu(n_sentences=64); bleu_elapsed = now() - t

        print(f"step {iter}: train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}, BLEU {bleu:.2f}")
        if iter == 0:
            print(f"  timing: loss {loss_elapsed:.1f}s | bleu {bleu_elapsed:.1f}s | total {now()-total_start:.0f}s")
        else:
            print(f"  timing: {eval_interval} train steps {train_elapsed:.1f}s "
                  f"({train_elapsed/eval_interval*1000:.0f}ms/step) | "
                  f"loss {loss_elapsed:.1f}s | bleu {bleu_elapsed:.1f}s | "
                  f"total {now()-total_start:.0f}s")

        train_start = now()

    it_x, en_x, en_y = generate_batch('train')

    logits = m(it_x, en_x)
    loss = calculate_loss(logits, en_y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

total_elapsed = now() - total_start
print(f"training complete: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min) over {training_steps} steps")

print('final bleu on 512 sentences')
t = now()
final_bleu = estimate_bleu(n_sentences=512, max_print=10)
print(f"final BLEU (512 sentences): {final_bleu:.2f} in {now()-t:.0f}s")

# Save weights
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
ckpt_path = f'attention-replica-{timestamp}.pt'
torch.save(m.state_dict(), ckpt_path)
print(f'saved weights to {ckpt_path}')

sys.exit(0)
