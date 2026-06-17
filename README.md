# Attention Replica

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
    * LR needs to be changed proportionally to batch size (Adam / AdamW is LR scales batch_size**0.5)
* Split model and related hyperparams in a separate file
* Implement wordpiece tokens (1609.08144)
* Need to try `torch.compile`

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
