import os
# Must be set before torch initialises the CUDA allocator.
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
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

# Invocation guard
if __name__ != "__main__":
    sys.exit(-1)

#################
# Hyperparameters
d_model = 512
d_hid = 4 * d_model
max_len = 600
max_tokens = 25000
gen_batch_size = 128 # Max generated batch size
n_head = 8
d_head = d_model // n_head; assert d_model % n_head == 0
n_stack = 6
p_dropout = 0.1

dataset_name = "Helsinki-NLP/opus-100"
target_vocab_size = 16000
tokenizer_path = f"tokenizer-{dataset_name.split('/')[-1]}-{target_vocab_size}.json"

training_steps = 100_000
warmup_steps = 4_000

eval_interval = 5_000
eval_iters = 50
eval_bleu_sentences = 256
# ---------------

###################
# Setup environment
load_dotenv()

def detect_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

device = detect_device()
print(f'device: {device}')

#############################
# Load and initialise dataset
raw = load_dataset(dataset_name, "en-it")

def clean_split(split, apply_ratio):
    n_discarded_malformed, n_discarded_len, n_discarded_ratio = 0, 0, 0
    it_out, en_out = [], []
    for x in split['translation']:
        it, en = x['it'].strip(), x['en'].strip()
        if not it or not en:
            n_discarded_malformed += 1
            continue
        if len(it) > 600 or len(en) > 600:
            n_discarded_len += 1
            continue
        if apply_ratio and not (0.5 <= len(it) / len(en) <= 2.0):
            n_discarded_ratio += 1
            continue
        
        it_out.append(it)
        en_out.append(en)

    print(f'total: {len(it_out)}, malformend: {n_discarded_malformed}, > {600}: {n_discarded_len}, bad ratio: {n_discarded_ratio if apply_ratio == True else False}')
    return it_out, en_out

it_train_txt, en_train_txt = clean_split(raw['train'], apply_ratio=True)
it_eval_txt, en_eval_txt = clean_split(raw['validation'], apply_ratio=False)

############
# Vocabulary
pad_token = '[PAD]'
bos_token = '[BOS]'
eos_token = '[EOS]'
unk_token = '[UNK]'

if os.path.exists(tokenizer_path):
    print(f'loading tokenizer from {tokenizer_path}')
    tokenizer = Tokenizer.from_file(tokenizer_path)
else:
    print('training tokenizer')
    tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = WordPieceDecoder()
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
# using tokenizer.encode_batch as it's faster than iterating on each in Python
encode_all = lambda texts: [e.ids for e in tokenizer.encode_batch(texts)]

pad_token_idx = tokenizer.token_to_id(pad_token)
bos_token_idx = tokenizer.token_to_id(bos_token)
eos_token_idx = tokenizer.token_to_id(eos_token)

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
def seq_len(it_data, en_data, i):
    # +1 on the target accounts for the bos/eos added at collate time
    return max(len(it_data[i]), len(en_data[i]) + 1)

def build_token_batches(it_data, en_data):
    order = sorted(range(len(it_data)), key=lambda i: seq_len(it_data, en_data, i))
    batches, cur, cur_max = [], [], 0
    for i in order:
        new_max = max(cur_max, seq_len(it_data, en_data, i))
        if cur and new_max * (len(cur) + 1) > max_tokens: # Calculates the nr. of tokens we'd need to add to the batch after padding
            batches.append(cur)
            cur, new_max = [], seq_len(it_data, en_data, i)
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches

train_batches = build_token_batches(it_train, en_train)
eval_batches = build_token_batches(it_eval, en_eval)
# Length-sorted eval order for the deterministic BLEU sample (across all lengths)
eval_sorted_idx = sorted(range(len(en_eval)), key=lambda i: seq_len(it_eval, en_eval, i))
print(f'train batches: {len(train_batches)}, eval batches: {len(eval_batches)} (~{max_tokens} tokens each)')

def pad(ls):
    return nn.utils.rnn.pad_sequence(ls, batch_first=True, padding_value=pad_token_idx)

def collate(idxs, it_data, en_data):
    it_x = pad([torch.tensor(it_data[i]) for i in idxs])

    bos = torch.tensor([bos_token_idx])
    eos = torch.tensor([eos_token_idx])
    en_seqs = [torch.tensor(en_data[i]) for i in idxs]
    en_x = pad([torch.cat([bos, s]) for s in en_seqs])
    en_y = pad([torch.cat([s, eos]) for s in en_seqs])

    return it_x.to(device), en_x.to(device), en_y.to(device)

def generate_batch(dataset):
    if dataset == 'train':
        return collate(random.choice(train_batches), it_train, en_train)
    return collate(random.choice(eval_batches), it_eval, en_eval)

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

    # --- Incremental cached attention, used only by generate() ---
    def _project_heads(self, lin, tok_x):
        B, T, _ = tok_x.shape
        return lin(tok_x).view(B, T, n_head, d_head).transpose(1, 2)

    def step_self_attn(self, tok_x, cache):
        q = self._project_heads(self.q_wei, tok_x)
        k = self._project_heads(self.k_wei, tok_x)
        v = self._project_heads(self.v_wei, tok_x)
        if cache['k'] is not None:
            k = torch.cat([cache['k'], k], dim=2)
            v = torch.cat([cache['v'], v], dim=2)
        cache['k'], cache['v'] = k, v
        # Causal mask is implicit and not needed when processing one token at a time
        # since we have no "future" tokens
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(tok_x.shape[0], 1, d_model)
        return self.dropout(self.linear(out))

    def step_cross_attn(self, tok_x, cache, memory, pad_mask_src):
        # Source K/V depend only on the fixed encoder memory, so compute them once.
        q = self._project_heads(self.q_wei, tok_x)
        if cache['k'] is None:
            cache['k'] = self._project_heads(self.k_wei, memory)
            cache['v'] = self._project_heads(self.v_wei, memory)
        attn_mask = torch.zeros_like(pad_mask_src, dtype=q.dtype) \
            .masked_fill(pad_mask_src, float('-inf')).unsqueeze(1)
        out = F.scaled_dot_product_attention(q, cache['k'], cache['v'], attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(tok_x.shape[0], 1, d_model)
        return self.dropout(self.linear(out))


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

    def step(self, x, memory, pad_mask_src, cache): # Single token
        x = x + self.attn.step_self_attn(self.ln1(x), cache['self'])
        x = x + self.cross_attn.step_cross_attn(self.ln2(x), cache['cross'], memory, pad_mask_src)
        x = x + self.ffw(self.ln3(x))
        return x


class AttentionReplica(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb_table = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_idx)
        self.pos_enc = PositionalEncoding()

        self.encoder = nn.ModuleList([EncoderStack() for _ in range(n_stack)])
        self.enc_dropout = nn.Dropout(p_dropout)
        self.enc_norm = nn.LayerNorm(d_model)

        self.decoder = nn.ModuleList([DecoderStack() for _ in range(n_stack)])
        self.dec_dropout = nn.Dropout(p_dropout)
        self.dec_norm = nn.LayerNorm(d_model)

        self.linear = nn.Linear(d_model, vocab_size)

    def encode(self, src_x, pad_mask_src_x):
        src_out = self.emb_table(src_x) * math.sqrt(d_model)
        src_out = self.pos_enc(src_out)
        src_out = self.enc_dropout(src_out)
        for encoderStack in self.encoder:
            src_out = encoderStack(src_out, pad_mask_src_x)
        return self.enc_norm(src_out)

    def forward(self, src_x, trs_x):

        pad_mask_src_x = (src_x == pad_token_idx).unsqueeze(-2)
        pad_mask_trs_x = (trs_x == pad_token_idx).unsqueeze(-2)

        src_out = self.encode(src_x, pad_mask_src_x)

        trs_out = self.emb_table(trs_x) * math.sqrt(d_model)
        trs_out = self.pos_enc(trs_out)
        trs_out = self.dec_dropout(trs_out)
        for decoderStack in self.decoder:
            trs_out = decoderStack(trs_out, pad_mask_trs_x, src_out, pad_mask_src_x)
        trs_out = self.dec_norm(trs_out)

        trs_out = self.linear(trs_out)
        return trs_out

    def decode_step(self, tok, pos, memory, pad_mask_src, caches):
        # Encode and add new token
        x = self.emb_table(tok) * math.sqrt(d_model)
        x = x + self.pos_enc.pe[:, pos:pos + 1]

        x = self.dec_dropout(x)
        for layer, cache in zip(self.decoder, caches):
            x = layer.step(x, memory, pad_mask_src, cache)
        x = self.dec_norm(x)
        return self.linear(x)[:, -1]

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
def _avg_loss(batches, it_data, en_data):
    # Token-weighted mean cross-entropy over the given batches (sum of per-token
    # loss / number of non-pad target tokens), so batches of different sizes are
    # weighted by their real token count rather than counted equally.
    total_loss, total_tokens = 0.0, 0
    for idxs in batches:
        it_x, en_x, en_y = collate(idxs, it_data, en_data)
        logits = m(it_x, en_x)
        B, T, E = logits.shape
        total_loss += F.cross_entropy(logits.view(B*T, E), en_y.view(B*T),
                                      ignore_index=pad_token_idx, label_smoothing=0.1,
                                      reduction='sum').item()
        total_tokens += (en_y != pad_token_idx).sum().item()
    return total_loss / total_tokens

@torch.no_grad()
def estimate_loss():
    m.eval()
    # Eval: deterministic full pass over the whole eval set (exact, and cheap
    # since it's only a few batches). Train: a random sample, since the full
    # train set is far too large to sweep every eval.
    train_sample = [random.choice(train_batches) for _ in range(eval_iters)]
    out = {
        'train': _avg_loss(train_sample, it_train, en_train),
        'eval': _avg_loss(eval_batches, it_eval, en_eval),
    }
    m.train()
    return out

@torch.no_grad()
def generate(src_x, max_new_tokens=max_len + 1):
    was_training = m.training
    m.eval()

    B = src_x.shape[0]
    src_x = src_x.to(device)

    pad_mask_src = (src_x == pad_token_idx).unsqueeze(-2)
    memory = m.encode(src_x, pad_mask_src) # Cache and reuse encoder result

    # Per-layer K/V caches: self-attn grows each step, cross-attn computed once.
    caches = [{'self': {'k': None, 'v': None}, 'cross': {'k': None, 'v': None}}
              for _ in range(n_stack)]

    trs = torch.full((B, 1), bos_token_idx, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        pos = trs.shape[1] - 1
        # Only the new token is fed, kv is cached, encoder data is precomputed
        logits = m.decode_step(trs[:, -1:], pos, memory, pad_mask_src, caches)
        next_tok = logits.argmax(-1)
        # once a sequence has emitted <eos>, keep it padded
        next_tok = torch.where(finished, torch.full_like(next_tok, pad_token_idx), next_tok)
        trs = torch.cat([trs, next_tok.unsqueeze(1)], dim=1)
        finished |= (next_tok == eos_token_idx)
        if finished.all():
            break

    if was_training:
        m.train()
    return trs

def ids_to_text(ids):
    ids = [int(i) for i in ids]
    # Cut at the first <eos> so we don't decode trailing padding, then let the
    # tokenizer drop the remaining special tokens and merge the wordpieces.
    if eos_token_idx in ids:
        ids = ids[:ids.index(eos_token_idx)]
    return tokenizer.decode(ids)

@torch.no_grad()
def translate_eval(n_sentences):
    # Translate a deterministic, length-spread sample of the eval set and return
    # (hyps, refs, srcs). Walks the length-sorted order so batches stay tightly
    # padded; the deterministic stride keeps the sample fixed across evals
    # (reproducibility). n_sentences >= len(eval) translates the whole set.
    hyps, refs, srcs = [], [], []
    n_sentences = min(n_sentences, len(eval_sorted_idx))
    stride = len(eval_sorted_idx) / n_sentences
    sample_idx = [eval_sorted_idx[int(i * stride)] for i in range(n_sentences)]
    for start in range(0, n_sentences, gen_batch_size):
        idxs = sample_idx[start:start + gen_batch_size]
        src_x = pad([torch.tensor(it_eval[i]) for i in idxs]).to(device)

        out = generate(src_x)
        hyps.extend(ids_to_text(row) for row in out)
        refs.extend(decode(en_eval[i]) for i in idxs)
        srcs.extend(decode(it_eval[i]) for i in idxs)
    return hyps, refs, srcs

def write_samples(f, hyps, refs, srcs, max_print):
    # Translations are written to a log file only, never printed to the screen.
    for h, r, s in list(zip(hyps, refs, srcs))[:max_print]:
        f.write(f'  HYP: {h!r}\n  REF: {r!r}\n  SRC: {s!r}\n\n')
    f.flush()

def bootstrap_bleu_ci(hyps, refs, n_boot=1000, level=0.95, seed=12345):
    # 95% confidence interval via bootstrap resampling of sentence pairs:
    # corpus_bleu is cheap (just n-gram counting), so we resample with
    # replacement many times and take percentiles of the resulting scores.
    score = sacrebleu.corpus_bleu(hyps, [refs]).score
    rng = random.Random(seed)
    n = len(hyps)
    scores = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        scores.append(sacrebleu.corpus_bleu([hyps[i] for i in idx],
                                            [[refs[i] for i in idx]]).score)
    scores.sort()
    lo = scores[int((1 - level) / 2 * n_boot)]
    hi = scores[int((1 + level) / 2 * n_boot)]
    return score, lo, hi

def now():
    if device == 'cuda':
        torch.cuda.synchronize()
    return time.perf_counter()

# ---- Per-run output folder: stats log, full log (+ samples), loss plot, checkpoints ----
run_dir = os.path.join('training', datetime.now().strftime('%Y%m%d-%H%M%S'))
os.makedirs(run_dir, exist_ok=True)
stats_file = open(os.path.join(run_dir, 'stats.log'), 'w')   # stats only (mirrors the screen)
full_file = open(os.path.join(run_dir, 'full.log'), 'w')     # stats + hyp/ref/src samples

def log(msg):
    # Stats lines go to the screen AND both log files; samples go to full_file only.
    print(msg)
    for f in (stats_file, full_file):
        f.write(msg + '\n')
        f.flush()

# Loss history for the plot: train loss every loss_record_interval steps, eval loss per eval.
loss_record_interval = 100
train_loss_steps, train_loss_hist = [], []
eval_loss_steps, eval_loss_hist = [], []
best_bleu, best_loss = float('-inf'), float('inf')

log(f'logging this run to {run_dir}/')
log('training')
m.train()
total_start = now()
train_start = now()
for iter in range(training_steps):
    if iter % eval_interval == 0:
        train_elapsed = now() - train_start

        t = now(); losses = estimate_loss(); loss_elapsed = now() - t
        t = now()
        hyps, refs, srcs = translate_eval(eval_bleu_sentences)
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        bleu_elapsed = now() - t

        # Translations to the full log only (kept off the screen).
        full_file.write(f"=== step {iter} samples ===\n")
        write_samples(full_file, hyps, refs, srcs, max_print=5)

        eval_loss_steps.append(iter)
        eval_loss_hist.append(losses['eval'])

        log(f"step {iter}: train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}, BLEU {bleu:.2f}")
        if iter == 0:
            log(f"  timing: loss {loss_elapsed:.1f}s | bleu {bleu_elapsed:.1f}s | total {now()-total_start:.0f}s")
        else:
            log(f"  timing: {eval_interval} train steps {train_elapsed:.1f}s "
                f"({train_elapsed/eval_interval*1000:.0f}ms/step) | "
                f"loss {loss_elapsed:.1f}s | bleu {bleu_elapsed:.1f}s | "
                f"total {now()-total_start:.0f}s")

        # Best-eval checkpoints: BLEU and loss tracked independently (BLEU is the
        # task metric but noisy; eval loss is the more stable signal).
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(m.state_dict(), os.path.join(run_dir, 'best-bleu.pt'))
            log(f"  new best BLEU {bleu:.2f} -> saved best-bleu.pt")
        if losses['eval'] < best_loss:
            best_loss = losses['eval']
            torch.save(m.state_dict(), os.path.join(run_dir, 'best-loss.pt'))
            log(f"  new best eval loss {losses['eval']:.4f} -> saved best-loss.pt")

        train_start = now()

    it_x, en_x, en_y = generate_batch('train')

    logits = m(it_x, en_x)
    loss = calculate_loss(logits, en_y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    if iter % loss_record_interval == 0:
        train_loss_steps.append(iter)
        train_loss_hist.append(loss.item())

total_elapsed = now() - total_start
log(f"training complete: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min) over {training_steps} steps")

# Decision-quality BLEU: translate the WHOLE eval set once, then report the
# score with a bootstrap 95% confidence interval (a trustworthy number, unlike
# the small periodic samples used for the training-time trend).
log('final bleu on the full eval set')
t = now()
hyps, refs, srcs = translate_eval(len(eval_sorted_idx))
full_file.write("=== final full-eval samples ===\n")
write_samples(full_file, hyps, refs, srcs, max_print=10)
score, lo, hi = bootstrap_bleu_ci(hyps, refs)
log(f"final BLEU (full eval, {len(hyps)} sentences): {score:.2f} "
    f"95% CI [{lo:.2f}, {hi:.2f}] (±{(hi - lo) / 2:.2f}) in {now()-t:.0f}s")

# ---- Training-loss plot ----
plt.figure(figsize=(9, 5))
plt.plot(train_loss_steps, train_loss_hist, linewidth=0.7, label=f'train loss (every {loss_record_interval} steps)')
plt.plot(eval_loss_steps, eval_loss_hist, marker='o', label='eval loss')
plt.xlabel('step'); plt.ylabel('loss'); plt.title('Training loss')
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(run_dir, 'loss.png'), dpi=120, bbox_inches='tight')
log(f'saved loss plot to {os.path.join(run_dir, "loss.png")}')

# ---- Final weights ----
torch.save(m.state_dict(), os.path.join(run_dir, 'final.pt'))
log(f'saved final weights to {os.path.join(run_dir, "final.pt")}')

stats_file.close()
full_file.close()
sys.exit(0)

sys.exit(0)
