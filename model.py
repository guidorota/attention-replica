import math
import torch
import torch.nn as nn
from torch.nn import functional as F

import config
from text import vocab_size, pad_token_idx, bos_token_idx, eos_token_idx

d_model, d_hid, n_head, d_head, n_stack, p_dropout, max_len = (
    config.d_model, config.d_hid, config.n_head, config.d_head,
    config.n_stack, config.p_dropout, config.max_len)

##################
# Model definition
class PositionalEncoding(nn.Module):

    def __init__(self):
        super().__init__()
        # Pre-generate lookup table
        pe_len = max_len + 1  # +1 accounts for adding bos / eos
        pe = torch.zeros(pe_len, d_model)
        pos = torch.arange(pe_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

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
        mask_len = max_len + 1  # Accounts for adding bos / eos
        self.register_buffer('causal_mask', torch.tril(torch.ones(mask_len, mask_len)) == 0)

    def forward(self, q_x, kv_x, pad_mask, apply_causal_mask=False):
        B, T_q, _ = q_x.shape
        T_kv = kv_x.shape[1]

        # Project then split into heads: (B, n_head, T, d_head)
        q = self.q_wei(q_x).view(B, T_q, n_head, d_head).transpose(1, 2)
        k = self.k_wei(kv_x).view(B, T_kv, n_head, d_head).transpose(1, 2)
        v = self.v_wei(kv_x).view(B, T_kv, n_head, d_head).transpose(1, 2)

        mask = pad_mask.unsqueeze(1)                          # (B, 1, 1, T_kv)
        if apply_causal_mask:
            mask = mask | self.causal_mask[:T_q, :T_kv]       # broadcast (T_q, T_kv)
        attn_mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(mask, float('-inf'))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T_q, d_model)    # re-merge heads

        out = self.linear(out)
        out = self.dropout(out)
        return out

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
        # Pre-LN
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
        # Pre-LN: src_out is the encoder memory,
        # already normalised by the encoder's final LN, so it's used as-is for cross-attn k/v.
        normed = self.ln1(trs_x)
        out = trs_x + self.attn(normed, normed, pad_mask_trs_x, apply_causal_mask=True)
        out = out + self.cross_attn(self.ln2(out), src_out, pad_mask_src_x)
        out = out + self.ffw(self.ln3(out))
        return out

    def step(self, x, memory, pad_mask_src, cache):  # Single token
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


##########
# Decoding
@torch.no_grad()
def generate(model, src_x, max_new_tokens=max_len + 1):
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    B = src_x.shape[0]
    src_x = src_x.to(device)

    pad_mask_src = (src_x == pad_token_idx).unsqueeze(-2)
    memory = model.encode(src_x, pad_mask_src)  # Cache and reuse encoder result

    # Per-layer K/V caches: self-attn grows each step, cross-attn computed once.
    caches = [{'self': {'k': None, 'v': None}, 'cross': {'k': None, 'v': None}}
              for _ in range(n_stack)]

    trs = torch.full((B, 1), bos_token_idx, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        pos = trs.shape[1] - 1
        # Only the new token is fed, kv is cached, encoder data is precomputed
        logits = model.decode_step(trs[:, -1:], pos, memory, pad_mask_src, caches)
        next_tok = logits.argmax(-1)
        # Once a sequence has emitted <eos>, keep it padded
        next_tok = torch.where(finished, torch.full_like(next_tok, pad_token_idx), next_tok)
        trs = torch.cat([trs, next_tok.unsqueeze(1)], dim=1)
        finished |= (next_tok == eos_token_idx)
        if finished.all():
            break

    if was_training:
        model.train()
    return trs


def reorder_caches(caches, index):
    # Beam search reorders the surviving hypotheses every step, so the per-layer
    # K/V caches (batch is dim 0) must follow their parent beams to stay aligned
    # with the sequences. Cross-attn K/V is constant per source but lives in the
    # same cache, so it gets reordered identically (harmless).
    for layer in caches:
        for attn in ('self', 'cross'):
            for kv in ('k', 'v'):
                t = layer[attn][kv]
                if t is not None:
                    layer[attn][kv] = t.index_select(0, index)


@torch.no_grad()
def generate_beam(model, src_x, beam=config.beam_size, alpha=config.beam_length_penalty,
                  max_new_tokens=max_len + 1):
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    B = src_x.shape[0]
    src_x = src_x.to(device)

    pad_mask_src = (src_x == pad_token_idx).unsqueeze(-2)
    memory = model.encode(src_x, pad_mask_src)

    N = B * beam
    # Give every beam its own copy of the encoder outputs
    memory = memory.repeat_interleave(beam, dim=0)
    pad_mask_src = pad_mask_src.repeat_interleave(beam, dim=0)

    caches = [{'self': {'k': None, 'v': None}, 'cross': {'k': None, 'v': None}}
              for _ in range(n_stack)]

    seqs = torch.full((N, 1), bos_token_idx, dtype=torch.long, device=device)
    # Per-beam cumulative log-prob. Seed only beam 0 of each source as "live" so
    # the first step doesn't expand `beam` identical <bos> rows into duplicates.
    beam_scores = torch.full((B, beam), float('-inf'), device=device)
    beam_scores[:, 0] = 0.0

    finished = [[] for _ in range(B)]   # (length_penalised_score, ids) per source
    base = (torch.arange(B, device=device) * beam).unsqueeze(1)  # (B,1) row offsets

    for _ in range(max_new_tokens):
        pos = seqs.shape[1] - 1
        logits = model.decode_step(seqs[:, -1:], pos, memory, pad_mask_src, caches)  # (N, V)
        logp = F.log_softmax(logits, dim=-1)
        V = logp.shape[-1]

        # Score every (beam, token) continuation, then keep the best `beam` per source.
        cand = (beam_scores.view(N, 1) + logp).view(B, beam * V)   # (B, beam*V)
        # topk returns scores and indexes in decreasing order of value. That, combined with
        # the fact that the same parent can appear multiple times in the topk is what makes
        # reordering of the caches necessary.
        top_scores, top_idx = cand.topk(beam, dim=-1)              # (B, beam)
        parent = top_idx // V                                      # (B, beam) in [0, beam)
        next_tok = top_idx % V                                     # (B, beam)

        abs_parent = (base + parent).view(-1)                      # (N,) into [0, N)
        reorder_caches(caches, abs_parent)
        seqs = torch.cat([seqs[abs_parent], next_tok.view(-1, 1)], dim=1)
        beam_scores = top_scores

        gen_len = seqs.shape[1] - 1   # nr. of tokens generated after <bos>
        # Retire any beam that just emitted <eos> and park its slot at -inf so it
        # is never extended or re-selected.
        eos_mask = next_tok == eos_token_idx
        if eos_mask.any():
            for b, k in eos_mask.nonzero(as_tuple=False).tolist():
                if beam_scores[b, k].item() == float('-inf'):
                    continue
                lp = beam_scores[b, k].item() / (gen_len ** alpha)
                finished[b].append((lp, seqs[b * beam + k].clone()))
                beam_scores[b, k] = float('-inf')

        if all(len(f) >= beam for f in finished):
            break

    # Select the best beam (synthesize a score if no beam has finished for a particular input)
    out = []
    final_len = seqs.shape[1] - 1
    for b in range(B):
        pool = finished[b] or [(beam_scores[b, k].item() / (final_len ** alpha),
                                seqs[b * beam + k]) for k in range(beam)]
        out.append(max(pool, key=lambda x: x[0])[1])

    if was_training:
        model.train()
    return out


######################
# Build / save / load
def architecture_config():
    return {
        'd_model': config.d_model, 'd_hid': config.d_hid, 'n_head': config.n_head,
        'd_head': config.d_head, 'n_stack': config.n_stack, 'p_dropout': config.p_dropout,
        'max_len': config.max_len, 'vocab_size': vocab_size, 'pad_token_idx': pad_token_idx,
        'tokenizer_path': config.tokenizer_path,
    }

def build_model(device=None):
    device = device or config.device
    return AttentionReplica().to(device)

def save_checkpoint(model, path):
    torch.save({'model': model.state_dict(), 'config': architecture_config()}, path)

def load_checkpoint(path, device=None, strict=True):
    device = device or config.device
    ckpt = torch.load(path, map_location=device)
    if not (isinstance(ckpt, dict) and 'model' in ckpt and 'config' in ckpt):
        raise ValueError(
            f"{path} is not a self-describing checkpoint; run `python migrate_checkpoints.py`")

    saved, cur = ckpt['config'], architecture_config()
    # tokenizer_path is informational; dims must match for the weights to load at all.
    mismatched = {k: (saved.get(k), cur[k]) for k in cur
                  if k != 'tokenizer_path' and saved.get(k) != cur[k]}
    if mismatched:
        raise ValueError(f"checkpoint architecture differs from current config: {mismatched}")

    model = build_model(device)
    model.load_state_dict(ckpt['model'], strict=strict)
    model.eval()
    return model, ckpt['config']
