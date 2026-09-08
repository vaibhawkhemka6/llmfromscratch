"""
Simple Tokenizer V3

Extends SimpleTokenizerV2 by preserving whitespace as first-class tokens
instead of discarding it. This means decode(encode(text)) can reconstruct
the original text exactly (spacing, newlines, multiple spaces, etc.),
which V1/V2 could not do since they stripped whitespace and re-joined
tokens with a single space.

This mirrors how byte-level tokenizers (e.g. GPT-2 / tiktoken) treat
whitespace as meaningful, rather than something to throw away -- a useful
stepping stone before moving to a real BPE tokenizer.
"""

import os
import re

# Special tokens must be matched as whole, atomic units -- *before* falling
# back to the generic word/punctuation/whitespace rules below. Otherwise a
# literal "<|endoftext|>" in the input would get shredded into '<', '|',
# 'endoftext', '|', '>' since none of those characters are \w and each one
# would match the single-char punctuation rule on its own.
SPECIAL_TOKENS = ("<|endoftext|>", "<|unk|>")

_SPLIT_PATTERN = re.compile(
    "|".join(re.escape(tok) for tok in SPECIAL_TOKENS) + r"|\w+|[^\w\s]|\s"
)


class SimpleTokenizerV3:
    """A regex-based tokenizer that keeps whitespace runs as their own tokens."""

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    @staticmethod
    def split_text(text):
        # Capture special tokens (as whole units), word runs, single
        # punctuation/symbol chars, and *individual* whitespace characters
        # (not whole runs) as separate tokens. Nothing is stripped or
        # dropped, so the original text can be perfectly reconstructed by
        # simply concatenating tokens back together.
        #
        # Whitespace is split one character at a time (\s, not \s+) so that
        # any combination of spaces/newlines/tabs in new text can always be
        # represented -- even if that exact *run* (e.g. "  " or a lone "\n")
        # never appeared verbatim in the training text. A run like "\n\n"
        # just becomes two "\n" tokens back to back.
        preprocessed = _SPLIT_PATTERN.findall(text)
        return preprocessed

    def encode(self, text):
        preprocessed = self.split_text(text)
        # Replace any token that isn't in the vocab with <|unk|>
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        # No spaces are inserted here -- whitespace tokens already carry
        # their own spacing, so a plain join reconstructs the text exactly.
        text = "".join(self.int_to_str[i] for i in ids)
        return text


def build_vocab(tokens, special_tokens=SPECIAL_TOKENS):
    all_tokens = sorted(set(tokens))
    all_tokens.extend(special_tokens)
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    return vocab


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "the-verdict.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokens = SimpleTokenizerV3.split_text(raw_text)
    vocab = build_vocab(tokens)
    print(f"Vocabulary size (with whitespace + special tokens): {len(vocab)}")

    tokenizer = SimpleTokenizerV3(vocab)

    # --- Test 1: exact round-trip on text fully covered by the vocab ---
    # Built from whole tokens (not a raw character slice) so we don't
    # accidentally cut a word in half and create an artificial <|unk|>.
    sample_text = "".join(tokens[:60])
    ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(ids)

    print("\n--- Round-trip test (in-vocab text) ---")
    print(f"Sample text:\n{sample_text!r}")
    print(f"Decoded text:\n{decoded_text!r}")
    print(f"Exact match: {sample_text == decoded_text}")

    # --- Test 2: out-of-vocab words still fall back to <|unk|>, ---
    # --- while whitespace/newlines around them are preserved exactly ---
    text = "Hello,\ndo you like tea?  In the sunlit terraces."
    ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(ids)

    print("\n--- <|unk|> test (out-of-vocab words) ---")
    print(f"Input text:\n{text!r}")
    print(f"Encoded ids: {ids}")
    print(f"Decoded text:\n{decoded_text!r}")

    # --- Test 3: <|endoftext|> stays atomic, joining two chunks of text ---
    text1 = "".join(tokens[:20])
    text2 = "".join(tokens[20:40])
    joined = text1 + "<|endoftext|>" + text2

    ids = tokenizer.encode(joined)
    decoded_text = tokenizer.decode(ids)

    print("\n--- <|endoftext|> test (document boundary) ---")
    print(f"Input text:\n{joined!r}")
    print(f"'<|endoftext|>' kept as single token: "
          f"{vocab['<|endoftext|>'] in ids}")
    print(f"Decoded text:\n{decoded_text!r}")
    print(f"Exact match: {joined == decoded_text}")


if __name__ == "__main__":
    main()
