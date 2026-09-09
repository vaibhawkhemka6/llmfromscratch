"""
BPE Tokenizer from scratch

Implements Byte Pair Encoding (BPE) ourselves -- the same algorithm behind
tiktoken's GPT-2 encoding (04_BPEtokenizer_tiktoken.py) -- instead of using
a pretrained library. This is how you'd actually *train* a BPE tokenizer
on your own corpus.

Algorithm:
  1. Start from a base vocabulary of the 256 possible raw bytes (0-255).
     Encoding UTF-8 bytes directly -- rather than characters/words -- means
     ANY text (any language, emoji, typos, made-up words) can always be
     represented from the start. No <|unk|> is ever needed.
  2. Repeatedly find the most frequent adjacent pair of tokens in the
     training data and "merge" it into a brand new token id appended to
     the vocabulary. Record the merge rule.
  3. Stop once we've done `vocab_size - 256` merges.

To encode new text: convert to raw bytes, then repeatedly apply the
learned merge rules in the order they were learned (earliest-learned
merge = highest priority), until no more merges apply.

To decode: look up each id's byte sequence in the vocab and concatenate.
"""

import os
from collections import Counter


def get_pair_counts(ids):
    """Count how often each adjacent pair of ids occurs in the sequence."""
    counts = Counter()
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


def merge_pair(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    merged = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


class BPETokenizerScratch:
    def __init__(self):
        # (id1, id2) -> merged_id, in the order they were learned. The
        # order IS the priority: earlier merges are applied first.
        self.merges = {}
        # id -> raw bytes it represents (base 256 ids = single raw bytes)
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256, "vocab_size must be at least 256 (one per raw byte)"
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = get_pair_counts(ids)
            if not stats:
                break  # nothing left to merge

            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = merge_pair(ids, pair, new_id)

            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {new_id} "
                    f"({vocab[new_id]!r}) had {stats[pair]} occurrences"
                )

        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_pair_counts(ids)
            # Among pairs present in the text, pick the one that was
            # learned EARLIEST during training (lowest merge id).
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no more applicable merges
            new_id = self.merges[pair]
            ids = merge_pair(ids, pair, new_id)
        return ids

    def decode(self, ids):
        raw_bytes = b"".join(self.vocab[idx] for idx in ids)
        return raw_bytes.decode("utf-8", errors="replace")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "the-verdict.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = BPETokenizerScratch()
    vocab_size = 500  # 256 base bytes + 244 learned merges
    print(f"Training BPE on the-verdict.txt (target vocab_size={vocab_size})...")
    tokenizer.train(raw_text, vocab_size=vocab_size, verbose=False)
    print(f"Learned {len(tokenizer.merges)} merges. Final vocab size: {len(tokenizer.vocab)}")

    print("\nFirst 10 learned merges:")
    for pair, new_id in list(tokenizer.merges.items())[:10]:
        print(f"  {pair} -> {new_id} ({tokenizer.vocab[new_id]!r})")

    # --- Test 1: round-trip on a short sample ---
    text = "Hello, do you like tea? In the sunlit terraces of someunknownPlace."
    ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(ids)

    print("\n--- Round-trip test ---")
    print(f"Input text:\n{text!r}")
    print(f"Encoded ids ({len(ids)} tokens): {ids}")
    print(f"Decoded text:\n{decoded_text!r}")
    print(f"Exact match: {text == decoded_text}")

    # --- Test 2: made-up words still encode fine (byte-level fallback) ---
    unknown_text = "Akwirw ier \U0001F600"  # includes an emoji too
    ids = tokenizer.encode(unknown_text)
    decoded_text = tokenizer.decode(ids)

    print("\n--- Unknown-word / unicode test (no <|unk|> needed) ---")
    print(f"Input text: {unknown_text!r}")
    print(f"Encoded ids: {ids}")
    print(f"Decoded text: {decoded_text!r}")
    print(f"Exact match: {unknown_text == decoded_text}")

    # --- Test 3: full-file round-trip + compression ratio ---
    ids = tokenizer.encode(raw_text)
    decoded_text = tokenizer.decode(ids)

    print("\n--- the-verdict.txt round-trip ---")
    print(f"Characters: {len(raw_text)}")
    print(f"Raw bytes (base tokens): {len(raw_text.encode('utf-8'))}")
    print(f"BPE tokens after merges: {len(ids)}")
    print(f"Compression ratio: {len(raw_text.encode('utf-8')) / len(ids):.2f}x")
    print(f"Exact match: {raw_text == decoded_text}")


if __name__ == "__main__":
    main()
