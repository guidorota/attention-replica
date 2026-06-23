import os
import sys
import math
import time
import random
from datetime import datetime

import config
import data
import text
import model as model_mod
from eval import translate_eval, write_samples, bootstrap_bleu_ci, corpus_bleu

import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

device = config.device
print(f'device: {device}')

#############################
# Load and initialise dataset
corpus = data.load_clean_corpus(['train', 'validation'])
it_train, en_train = data.encode_split(*corpus['train'])
it_eval, en_eval = data.encode_split(*corpus['validation'])

print(f'it train length: {len(it_train)}')
print(f'en train length: {len(en_train)}')
print(f'it eval length: {len(it_eval)}')
print(f'en eval length: {len(en_eval)}')

train_batches = data.build_token_batches(it_train, en_train)
eval_batches = data.build_token_batches(it_eval, en_eval)
eval_sorted_idx = data.sorted_idx(it_eval, en_eval)
eval_data = (it_eval, en_eval, eval_sorted_idx)
print(f'train batches: {len(train_batches)}, eval batches: {len(eval_batches)} '
      f'(~{config.max_tokens} tokens each)')

def generate_batch(which):
    if which == 'train':
        return data.collate(random.choice(train_batches), it_train, en_train, device)
    return data.collate(random.choice(eval_batches), it_eval, en_eval, device)

##################
# Model / optimizer
m = model_mod.build_model(device)
n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f'number of parameters: {n_params/1e6:.2f}M')

def lr_lambda(step):
    if step < config.warmup_steps:
        return step / config.warmup_steps
    progress = (step - config.warmup_steps) / (config.training_steps - config.warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

# Exclude biases and 1-D params (LayerNorm) from weight decay
decay, no_decay = [], []
for _n, p in m.named_parameters():
    if not p.requires_grad:
        continue
    (no_decay if p.ndim < 2 else decay).append(p)

optimizer = torch.optim.AdamW(
    [{'params': decay, 'weight_decay': config.weight_decay},
     {'params': no_decay, 'weight_decay': 0.0}],
    lr=config.peak_lr, betas=config.betas,
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def calculate_loss(logits, expected):
    B, T, E = logits.shape
    return F.cross_entropy(logits.view(B*T, E), expected.view(B*T),
                           ignore_index=text.pad_token_idx, label_smoothing=config.label_smoothing)

@torch.no_grad()
def _avg_loss(batches, it_data, en_data):
    total_loss, total_tokens = 0.0, 0
    for idxs in batches:
        it_x, en_x, en_y = data.collate(idxs, it_data, en_data, device)
        logits = m(it_x, en_x)
        B, T, E = logits.shape
        # Use sum reduction instead of mean so that we can aggregate across all batches
        total_loss += F.cross_entropy(logits.view(B*T, E), en_y.view(B*T),
                                      ignore_index=text.pad_token_idx,
                                      label_smoothing=config.label_smoothing,
                                      reduction='sum').item()
        total_tokens += (en_y != text.pad_token_idx).sum().item()
    return total_loss / total_tokens

@torch.no_grad()
def estimate_loss():
    m.eval()
    train_sample = [random.choice(train_batches) for _ in range(config.eval_iters)]
    out = {
        'train': _avg_loss(train_sample, it_train, en_train),
        'eval': _avg_loss(eval_batches, it_eval, en_eval),
    }
    m.train()
    return out

def now():
    if device == 'cuda':
        torch.cuda.synchronize()
    return time.perf_counter()

###########
# Logging
run_dir = os.path.join('training', datetime.now().strftime('%Y%m%d-%H%M%S'))
os.makedirs(run_dir, exist_ok=True)
stats_file = open(os.path.join(run_dir, 'stats.log'), 'w')
full_file = open(os.path.join(run_dir, 'full.log'), 'w')

def log(msg):
    print(msg)
    for f in (stats_file, full_file):
        f.write(msg + '\n')
        f.flush()

train_loss_steps, train_loss_hist = [], []
eval_loss_steps, eval_loss_hist = [], []
best_bleu, best_loss = float('-inf'), float('inf')

##########
# Train
log(f'logging this run to {run_dir}/')
log('training')
m.train()
total_start = now()
train_start = now()
for iteration in range(config.training_steps):
    if iteration % config.eval_interval == 0:
        train_elapsed = now() - train_start

        t = now(); losses = estimate_loss(); loss_elapsed = now() - t
        t = now()
        hyps, refs, srcs = translate_eval(m, eval_data, config.n_eval_bleu)
        bleu = corpus_bleu(hyps, refs)
        bleu_elapsed = now() - t

        full_file.write(f"=== step {iteration} samples ===\n")
        write_samples(full_file, hyps, refs, srcs)

        eval_loss_steps.append(iteration)
        eval_loss_hist.append(losses['eval'])

        log(f"step {iteration}: train loss {losses['train']:.4f}, "
            f"eval loss {losses['eval']:.4f}, BLEU {bleu:.2f}")
        if iteration == 0:
            log(f"  timing: loss {loss_elapsed:.1f}s | bleu {bleu_elapsed:.1f}s | "
                f"total {now()-total_start:.0f}s")
        else:
            log(f"  timing: {config.eval_interval} train steps {train_elapsed:.1f}s "
                f"({train_elapsed/config.eval_interval*1000:.0f}ms/step) | "
                f"loss {loss_elapsed:.1f}s | bleu {bleu_elapsed:.1f}s | "
                f"total {now()-total_start:.0f}s")

        if bleu > best_bleu:
            best_bleu = bleu
            model_mod.save_checkpoint(m, os.path.join(run_dir, 'best-bleu.pt'))
            log(f"  new best BLEU {bleu:.2f} -> saved best-bleu.pt")
        if losses['eval'] < best_loss:
            best_loss = losses['eval']
            model_mod.save_checkpoint(m, os.path.join(run_dir, 'best-loss.pt'))
            log(f"  new best eval loss {losses['eval']:.4f} -> saved best-loss.pt")

        train_start = now()

    it_x, en_x, en_y = generate_batch('train')

    logits = m(it_x, en_x)
    loss = calculate_loss(logits, en_y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), config.grad_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    if iteration % config.loss_record_interval == 0:
        train_loss_steps.append(iteration)
        train_loss_hist.append(loss.item())

total_elapsed = now() - total_start
log(f"training complete: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min) "
    f"over {config.training_steps} steps")

log(f'final bleu on the full eval set (beam size {config.beam_size})')
t = now()
hyps, refs, srcs = translate_eval(m, eval_data, len(eval_sorted_idx), beam=config.beam_size)
full_file.write("=== final full-eval samples ===\n")
write_samples(full_file, hyps, refs, srcs)
score, lo, hi = bootstrap_bleu_ci(hyps, refs)
log(f"final BLEU (full eval, {len(hyps)} sentences): {score:.2f} "
    f"95% CI [{lo:.2f}, {hi:.2f}] (±{(hi - lo) / 2:.2f}) in {now()-t:.0f}s")

####################
# Training loss plot
plt.figure(figsize=(9, 5))
plt.plot(train_loss_steps, train_loss_hist, linewidth=0.7,
         label=f'train loss (every {config.loss_record_interval} steps)')
plt.plot(eval_loss_steps, eval_loss_hist, marker='o', label='eval loss')
plt.xlabel('step'); plt.ylabel('loss'); plt.title('Training loss')
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(run_dir, 'loss.png'), dpi=120, bbox_inches='tight')
log(f'saved loss plot to {os.path.join(run_dir, "loss.png")}')

####################
# Save final weights
model_mod.save_checkpoint(m, os.path.join(run_dir, 'final.pt'))
log(f'saved final weights to {os.path.join(run_dir, "final.pt")}')

stats_file.close()
full_file.close()
sys.exit(0)
