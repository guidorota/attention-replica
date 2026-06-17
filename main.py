import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from dotenv import load_dotenv

# Hyperparameters
d_model=512
# ---------------

def detect_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def create_dataset(max_string_length: int):
    ds = load_dataset("Helsinki-NLP/opus_books", "en-it")['train']

    it_full, en_full = [], []
    for x in ds['translation']:
        if len(x['it']) > max_string_length:
            continue
        it_full.append(x['it'])
        en_full.append(x['en'])

    split_index = int(0.9 * len(it_full))

    it_train = it_full[:split_index]
    en_train = en_full[:split_index]
    it_eval = it_full[split_index:]
    en_eval = en_full[split_index:]

    return ds, it_full, en_full, it_train, en_train, it_eval, en_eval

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

    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb_table = nn.Embedding(vocab_size, d_model)

    def forward(self, source, tr_seq):
        return None


# Guard and main body of the script
if __name__ != "__main__":
    sys.exit(-1)

load_dotenv()

device = detect_device()
print(f'device: {device}')

# Cap max string length to 600 characters to simplify bucketing
# Longer ones are just outliers and would not be enough to fill a batch
#
# An alternative approach to try is to truncate instead
ds, it_full, en_full, it_train, en_train, it_eval, en_eval = create_dataset(600)
print(f'total dataset length: {ds.num_rows}')
print(f'it_train length: {len(it_train)}')
print(f'en_train length: {len(en_train)}')
print(f'it_eval length: {len(it_eval)}')
print(f'en_eval length: {len(en_eval)}')

vocab = sorted(list(set("".join(it_full + en_full))))
vocab_size = len(vocab)
print(f'vocab size: {vocab_size}')

m = AttentionReplica(vocab_size)

sys.exit(0)