import torch
import torch.nn as nn
from datasets import load_dataset

import config
from text import encode_all, pad_token_idx, bos_token_idx, eos_token_idx

#############
# 1. Corpus
def clean_split(translations, apply_ratio):
    n_malformed, n_len, n_ratio = 0, 0, 0
    it_out, en_out = [], []
    for x in translations:
        it, en = x['it'].strip(), x['en'].strip()
        if not it or not en:
            n_malformed += 1
            continue
        if len(it) > config.max_chars or len(en) > config.max_chars:
            n_len += 1
            continue
        if apply_ratio and not (config.ratio_lo <= len(it) / len(en) <= config.ratio_hi):
            n_ratio += 1
            continue
        it_out.append(it)
        en_out.append(en)
    print(f'total: {len(it_out)}, malformed: {n_malformed}, > {config.max_chars}: {n_len}, '
          f'bad ratio: {n_ratio if apply_ratio else False}')
    return it_out, en_out

def load_clean_corpus(splits):
    """splits: iterable of HF split names ('train', 'validation', 'test').
    Returns {split: (it_txt, en_txt)}. The length-ratio filter is applied to 'train' only."""
    raw = load_dataset(config.dataset_name, config.dataset_pair)
    out = {}
    for s in splits:
        out[s] = clean_split(raw[s]['translation'], apply_ratio=(s == 'train'))
    return out

#############
# 3. Encode
def encode_split(it_txt, en_txt):
    return encode_all(it_txt), encode_all(en_txt)

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
        if cur and new_max * (len(cur) + 1) > config.max_tokens:  # nr. of tokens after padding
            batches.append(cur)
            cur, new_max = [], seq_len(it_data, en_data, i)
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches

def sorted_idx(it_data, en_data):
    # Length-sorted order for the deterministic BLEU sample (spread across all lengths)
    return sorted(range(len(en_data)), key=lambda i: seq_len(it_data, en_data, i))

def pad(ls):
    return nn.utils.rnn.pad_sequence(ls, batch_first=True, padding_value=pad_token_idx)

def collate(idxs, it_data, en_data, device):
    it_x = pad([torch.tensor(it_data[i]) for i in idxs])

    bos = torch.tensor([bos_token_idx])
    eos = torch.tensor([eos_token_idx])
    en_seqs = [torch.tensor(en_data[i]) for i in idxs]
    en_x = pad([torch.cat([bos, s]) for s in en_seqs])
    en_y = pad([torch.cat([s, eos]) for s in en_seqs])

    return it_x.to(device), en_x.to(device), en_y.to(device)
