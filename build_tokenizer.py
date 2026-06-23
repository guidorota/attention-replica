import os

import config
import data

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import WordPiece as WordPieceDecoder

def main():
    if os.path.exists(config.tokenizer_path):
        print(f'{config.tokenizer_path} already exists; delete it to rebuild')
        return

    it_txt, en_txt = data.load_clean_corpus(['train'])['train']

    print('training tokenizer')
    tokenizer = Tokenizer(WordPiece(unk_token=config.unk_token))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = WordPieceDecoder()
    trainer = WordPieceTrainer(
        vocab_size=config.target_vocab_size,
        special_tokens=[config.pad_token, config.bos_token, config.eos_token, config.unk_token],
    )
    tokenizer.train_from_iterator(it_txt + en_txt, trainer)
    tokenizer.save(config.tokenizer_path)
    print(f'saved tokenizer to {config.tokenizer_path}')


if __name__ == '__main__':
    main()
