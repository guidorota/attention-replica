import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from dataclasses import dataclass

def detect_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def create_dataset():
    ds = load_dataset("Helsinki-NLP/opus_books", "en-it")['train']

    split_index = int(0.9 * ds.num_rows)

    it_full = [x['it'] for x in ds['translation']]
    en_full = [x['en'] for x in ds['translation']]

    it_train = it_full[:split_index]
    en_train = en_full[:split_index]
    it_eval = it_full[split_index:]
    en_eval = en_full[split_index:]

    return ds, it_full, en_full, it_train, en_train, it_eval, en_eval

class AttentionReplica(nn.Module):

    def __init__(self, hy: Hyperparameters, vocab_size: int):
        super().__init__()
        self.emb_table = nn.Embedding(vocab_size, hy.d_model)

    def forward(self, source, tr_seq):
        return None
    
@dataclass
class Hyperparameters:
    d_model: int

def main() -> int:
    load_dotenv()

    device = detect_device()
    print(f'device: {device}')

    ds, it_full, en_full, it_train, en_train, it_eval, en_eval = create_dataset()
    print(f'total dataset length: {ds.num_rows}')
    print(f'train length: {len(it_train)}')
    print(f'eval length: {len(en_eval)}')

    vocab = sorted(list(set("".join(it_full + en_full))))
    vocab_size = len(vocab)
    print(f'vocab size: {vocab_size}')

    # Hyperparameters
    hy = Hyperparameters(
        d_model=512
    )

    m = AttentionReplica(hy, vocab_size)

    return 0

if __name__ == "__main__":
    sys.exit(main())