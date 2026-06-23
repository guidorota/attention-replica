import os
# Must be set before torch initialises the CUDA allocator.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch

#################
# Architecture
d_model = 512
d_hid = 4 * d_model
n_head = 8
d_head = d_model // n_head; assert d_model % n_head == 0
n_stack = 6
p_dropout = 0.3
max_len = 600  # positional-encoding / mask length cap (+1 added for bos/eos)

#####################
# Tokenizer / dataset
dataset_name = "Helsinki-NLP/opus-100"
dataset_pair = "en-it"
target_vocab_size = 16000
tokenizer_path = f"tokenizer-{dataset_name.split('/')[-1]}-{target_vocab_size}.json"

max_tokens = 25000          # token budget per (padded) batch
max_chars = 600             # cleaning: drop pairs whose either side exceeds this
ratio_lo, ratio_hi = 0.5, 2.0  # cleaning: keep train pairs within this length ratio

pad_token = '[PAD]'
bos_token = '[BOS]'
eos_token = '[EOS]'
unk_token = '[UNK]'

############
# Training
training_steps = 100_000
warmup_steps = 1_000
peak_lr = 7e-4
weight_decay = 0.1
label_smoothing = 0.1
grad_clip = 1.0
betas = (0.9, 0.98)

###############
# Eval / logging
eval_interval = 5_000
eval_iters = 50
n_eval_bleu = 256
loss_record_interval = 100

##########
# Decoding
beam_size = 4
beam_length_penalty = 0.6
gen_batch_size = 128  # source sentences per decode batch (beam multiplies the rows)

#############
# Environment
def detect_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

device = detect_device()
