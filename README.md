# Attention Replica

## TODO

Improve tokenization:

* Switch to token budgeting for batches
* Switch to word-piece tokenization (would be good to try reverting to post-LN and removing gradient clipping to see if char-level tokenization was really what caused the model to ignore the src input completely)

Performance (nice to have):

* bf16 autocast (see notes further down)
* torch.compile - `m = torch.compile(m, dynamic=True)`
* Parallelise multi-head attention
* Split encoding and decoding so that we don't run the encoding stack for every character of the output translation

Additional changes

* Use multinomial to add some variation to the translations
* Train on a larger dataset

## Diary

### 2026-06-21

* First training run on an NVIDIA GPU. A few learnings on the current code structure:
  * `torch.compile` is way slower than expected, removing it temporarily to run a few training runs and find potential issues
  * BLEU stats take a long time to run, increasing eval_interval (1_000 -> 5_000), and reducing n_sentence (512 -> 64)
  * Running out of memory with batches of 64 sentences, reducing to 32 (arguably I should have run a training run with the longest sentences only to exclude OOMs mid-training)
* Other observations from the very first training run:
  * Loss is decreasing but doesn't seem to be converging very well (4.9994 -> 2.0221 -> 2.406 -> 2.1195 -> 2.3028 ...), need to investigate on this
  * BLEU is still at zero even after a 25K training iterations, need to emit some stats to manually sample the quality of the translations
* BLEU was calculated only using the shortest 64 strings, changing selection algorithm to still be deterministic, but sample across all lengths in the eval set
* Looked at a few more translations from ~20K training cycles: the results are always the same regardless of the src input string, which shows there's either a problem with my model or with my training setup. Results from sparring with claude:
  * Main culprits appears to be char level tokenization instead of wordpiece and significantly smaller dataset. Trying claude's suggestions:
    * move from post-ln to pre-ln (see "On Layer Normalization in the Transformer Architecture", also pre-ln is what karpathy uses in his lessons)
    * gradient clipping to ensure that a gradient spike can't influence the network too much
* Performance is acceptable at the moment, deprioritising bf16 autoscale and `torch.compile`
* First train run is resulting in overfitting (train loss decreases, 1.2557 @ 85_000, but eval loss and BLEU increase)

## Notes

* Training inputs
  * Full source string given to model at training time
  * Full translated string also given to the model at training time, but shifted one token right (teacher forcing)
  * Special token used to shift the translated string by one to the right (`<BOS>` or `<SOS>`, beginning / start of sentence)
  * End of sentence token also used (`<EOS>`)
  * Padding is masked away so that attention will ignore those positions
  * Loss is calculated character by character
  * Casual mask is also used to ensure that attention doesn't see future tokens
* Managing training inputs of different lengths
  * Padding can be used but might be wasteful from a computational perspective
  * Padding can be done on a per-batch basis to avoid having to pad everything on the longest sentence (distribution will help determine how much complexity we need to add)
  * Batches should use sentences of similar length to minimise padding (length bucketing)
  * Truncation might also help if needed
* Batch size
  * There isn't a hard and fast rule
  * HW puts a limitation (i.e., how many tokens can be processed in memory at training time)
  * Learning Rate LR is coupled to batch size
    * Bigger batches need "warmup" (used in the paper)
    * LR needs to be changed proportionally to batch size (Adam / AdamW is LR scales `batch_size**0.5`)
* Split model and related hyperparams in a separate file
* Implement wordpiece tokens (1609.08144)
* Need to try `torch.compile`, however note this will only have benefits when moving to GPU training since support for mps is still in progress: [https://github.com/pytorch/pytorch/issues/150121](https://github.com/pytorch/pytorch/issues/150121)

## Example of input for the network

Encoder input: I love cats

```(markdown)
  Decoder input:   <bos>    J'aime   les      chats
  Prediction at →  J'aime?  les?     chats?   <eos>?     ← logits over vocab at each position
  Label:           J'aime   les      chats    <eos>
  Loss:            CE       CE       CE       CE         ← cross-entropy at every position
                    └──────────── averaged ───────────┘
```

Reminder: padding needs to be excluded from cross entropy (`F.cross_entropy()` `ignore_index` parameter can be used for this purpose)

## Length bucketing

One option is to just order all sentences, batch them, and then shuffle the batches to restore some randomness:

```(python)
  # sort by length, then chunk into batches, then shuffle the batch order
  indices = sorted(range(len(data)), key=lambda i: len(data[i]))
  batches = [indices[i:i+bs] for i in range(0, len(indices), bs)]
  random.shuffle(batches)   # keep randomness across batches, similar lengths within
```

Another option called pool bucketing consists of creating big batches (50x times bigger than a normal batch), and then sort by length within each individual batch, instead of doing that globally.

## Padding

* Need to pad both source and translation
* Encoder attention needs to ignore all padding
* Decoder self attention needs to pad both future tokens, and padding
* Cross attention in the decoder needs to ignore input padding (as that's where K,V come from)
* Padding needs to be also considered when calculating the loss (`ignore_index` in `F.cross_entropy()`)

## bf16 autocast

Similar optimisation needs to be done in the estimate for the eval / train loss. Might not work for mps.

```(python)
  for iter in range(training_steps):
      it_x, en_x, en_y = generate_batch('train')

      with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
          logits = m(it_x, en_x)
          loss = calculate_loss(logits, en_y)

      loss.backward()          # OUTSIDE autocast
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad(set_to_none=True)

```

## word-piece

Train once across the full list of sentences (it/en)

```(python)
  from tokenizers import Tokenizer
  from tokenizers.models import WordPiece
  from tokenizers.trainers import WordPieceTrainer
  from tokenizers.pre_tokenizers import Whitespace

  tokenizer = Tokenizer(WordPiece(unk_token='[UNK]'))
  tokenizer.pre_tokenizer = Whitespace()

  trainer = WordPieceTrainer(
      vocab_size=16000,                                   # 8k–32k typical, review paper
      special_tokens=['[UNK]', '[PAD]', '[BOS]', '[EOS]'],
  )

  # train on the combined it+en text (shared vocab as the attention paper)
  tokenizer.train_from_iterator(it_full + en_full, trainer)
  tokenizer.save('tokenizer.json')                        # reload later with Tokenizer.from_file
```

Replace char level vocab and encode / decode

```(python)
  vocab_size      = tokenizer.get_vocab_size()
  pad_token_idx   = tokenizer.token_to_id('[PAD]')
  bos_token_idx   = tokenizer.token_to_id('[BOS]')
  eos_token_idx   = tokenizer.token_to_id('[EOS]')

  encode = lambda s: tokenizer.encode(s).ids
  decode = lambda ids: tokenizer.decode(ids)
```

Use tokenizer in batch generation

```(python)
  it_train = [tokenizer.encode(x).ids for x in it_full[:split_index]]
  en_train = [tokenizer.encode(x).ids for x in en_full[:split_index]]
```
