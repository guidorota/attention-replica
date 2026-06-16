import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

def detect_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
    
device = detect_device()
print(f'device: {device}')

ds = load_dataset("Helsinki-NLP/opus_books", "en-it")['train']

split_index = int(0.9 * ds.num_rows)

raw_it = [x['it'] for x in ds['translation']]
raw_en = [x['en'] for x in ds['translation']]

it_train = raw_it[:split_index]
en_train = raw_en[:split_index]
it_eval = raw_it[split_index:]
en_eval = raw_it[split_index:]

print(f'total dataset length: {ds.num_rows}')
print(f'train length: {len(it_train)}')
print(f'eval length: {len(en_eval)}')