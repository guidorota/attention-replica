import argparse
import random
import torch
import sacrebleu

import config
import data
import text
import model as model_mod

@torch.no_grad()
def translate_eval(model, eval_data, n_sentences, beam=1):
    """eval_data: (it_eval, en_eval, eval_sorted_idx). Returns (hyps, refs, srcs)."""
    it_eval, en_eval, eval_sorted_idx = eval_data
    hyps, refs, srcs = [], [], []
    n_sentences = min(n_sentences, len(eval_sorted_idx))
    stride = len(eval_sorted_idx) / n_sentences
    sample_idx = [eval_sorted_idx[int(i * stride)] for i in range(n_sentences)]
    for start in range(0, n_sentences, config.gen_batch_size):
        idxs = sample_idx[start:start + config.gen_batch_size]
        src_x = data.pad([torch.tensor(it_eval[i]) for i in idxs]).to(config.device)

        if beam == 1:
            out = model_mod.generate(model, src_x)
        else:
            out = model_mod.generate_beam(model, src_x, beam=beam)
        hyps.extend(text.ids_to_text(row) for row in out)
        refs.extend(text.decode(en_eval[i]) for i in idxs)
        srcs.extend(text.decode(it_eval[i]) for i in idxs)
    return hyps, refs, srcs


def write_samples(f, hyps, refs, srcs):
    for h, r, s in zip(hyps, refs, srcs):
        f.write(f'  HYP: {h!r}\n  REF: {r!r}\n  SRC: {s!r}\n\n')
    f.flush()


def corpus_bleu(hyps, refs):
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def bootstrap_bleu_ci(hyps, refs, n_boot=1000, level=0.95, seed=12345):
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


def load_eval_data(split):
    it_txt, en_txt = data.load_clean_corpus([split])[split]
    it_enc, en_enc = data.encode_split(it_txt, en_txt)
    return it_enc, en_enc, data.sorted_idx(it_enc, en_enc)


def main():
    ap = argparse.ArgumentParser(description='Score a checkpoint with BLEU + bootstrap CI.')
    ap.add_argument('checkpoint', help='path to a self-describing .pt checkpoint')
    ap.add_argument('--split', default='validation', choices=['train', 'validation', 'test'])
    ap.add_argument('--beam', type=int, default=config.beam_size,
                    help='beam width (1 == greedy)')
    ap.add_argument('-n', '--n-sentences', type=int, default=None,
                    help='number of sentences to score (default: the whole split)')
    args = ap.parse_args()

    model, ckpt_cfg = model_mod.load_checkpoint(args.checkpoint)
    print(f'loaded {args.checkpoint} on {config.device}')

    eval_data = load_eval_data(args.split)
    n = args.n_sentences or len(eval_data[2])

    hyps, refs, _ = translate_eval(model, eval_data, n, beam=args.beam)
    score, lo, hi = bootstrap_bleu_ci(hyps, refs)
    print(f'{args.split} BLEU (beam {args.beam}, {len(hyps)} sentences): '
          f'{score:.2f} 95% CI [{lo:.2f}, {hi:.2f}] (±{(hi - lo) / 2:.2f})')


if __name__ == '__main__':
    main()
