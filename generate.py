import argparse
import sys
import torch

import config
import text
import model as model_mod

def translate(model, src, beam=config.beam_size):
    src_x = torch.tensor([text.encode(src)], device=config.device)
    if beam == 1:
        out = model_mod.generate(model, src_x)
    else:
        out = model_mod.generate_beam(model, src_x, beam=beam)
    return text.ids_to_text(out[0])


def main():
    ap = argparse.ArgumentParser(description='Translate text with a trained checkpoint.')
    ap.add_argument('checkpoint', help='path to a self-describing .pt checkpoint')
    ap.add_argument('text', nargs='*', help='source text (reads stdin if omitted)')
    ap.add_argument('--beam', type=int, default=config.beam_size, help='beam width (1 == greedy)')
    args = ap.parse_args()

    src = ' '.join(args.text) if args.text else sys.stdin.read().strip()
    if not src:
        ap.error('no source text provided (pass it as arguments or via stdin)')

    model, _ = model_mod.load_checkpoint(args.checkpoint)
    print(translate(model, src, beam=args.beam))


if __name__ == '__main__':
    main()
