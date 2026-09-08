"""
Simple Tokenizer V2

Extends SimpleTokenizerV1 with special tokens:
  - <|unk|>        : substituted for any word not seen in the training vocabulary
  - <|endoftext|>   : marks a boundary between unrelated chunks of text
"""

import os
import re


class SimpleTokenizerV2:
    """A regex-based tokenizer with support for <|unk|> and <|endoftext|>."""

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    @staticmethod
    def split_text(text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        return [item.strip() for item in preprocessed if item.strip()]

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
        text = " ".join([self.int_to_str[i] for i in ids])
        # Remove extra whitespace before punctuation
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


def build_vocab(tokens, special_tokens=("<|endoftext|>", "<|unk|>")):
    all_tokens = sorted(set(tokens))
    all_tokens.extend(special_tokens)
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    return vocab


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "the-verdict.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokens = SimpleTokenizerV2.split_text(raw_text)
    vocab = build_vocab(tokens)
    print(f"Vocabulary size (with special tokens): {len(vocab)}")
    print(f"'<|unk|>' id: {vocab['<|unk|>']}, '<|endoftext|>' id: {vocab['<|endoftext|>']}")

    tokenizer = SimpleTokenizerV2(vocab)

    # Two unrelated snippets joined with <|endoftext|>, containing words
    # that are NOT in the-verdict.txt's vocabulary (to trigger <|unk|>).
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))

    print(f"\nInput text: {text}")

    ids = tokenizer.encode(text)
    print(f"Encoded ids: {ids}")

    decoded_text = tokenizer.decode(ids)
    print(f"Decoded text: {decoded_text}")


if __name__ == "__main__":
    main()
