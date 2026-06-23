import os
import config
from tokenizers import Tokenizer

if not os.path.exists(config.tokenizer_path):
    raise FileNotFoundError(
        f"tokenizer not found at {config.tokenizer_path}; run `python build_tokenizer.py` first")

tokenizer = Tokenizer.from_file(config.tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

pad_token_idx = tokenizer.token_to_id(config.pad_token)
bos_token_idx = tokenizer.token_to_id(config.bos_token)
eos_token_idx = tokenizer.token_to_id(config.eos_token)

# using tokenizer.encode_batch as it's faster than iterating on each in Python
def encode(s):
    return tokenizer.encode(s).ids

def decode(ids):
    return tokenizer.decode(ids)

def encode_all(texts):
    return [e.ids for e in tokenizer.encode_batch(texts)]

def ids_to_text(ids):
    ids = [int(i) for i in ids]
    # Cut at the first <eos> so we don't decode trailing padding
    if eos_token_idx in ids:
        ids = ids[:ids.index(eos_token_idx)]
    return tokenizer.decode(ids)
