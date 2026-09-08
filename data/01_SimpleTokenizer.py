"""
Simple Tokenizer

A basic tokenizer implementation as part of building an LLM from scratch.
"""

import os
import re


class SimpleTokenizerV1:
    """A basic regex-based tokenizer with a fixed vocabulary."""

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    @staticmethod
    def split_text(text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        return [item.strip() for item in preprocessed if item.strip()]

    def encode(self, text):
        preprocessed = self.split_text(text)
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Remove extra whitespace before punctuation
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


def build_vocab(tokens):
    all_words = sorted(set(tokens))
    vocab = {token: integer for integer, token in enumerate(all_words)}
    return vocab


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "the-verdict.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"Total number of characters: {len(raw_text)}")

    tokens = SimpleTokenizerV1.split_text(raw_text)
    print(f"Total number of tokens: {len(tokens)}")
    print(f"First 10 tokens: {tokens[:10]}")

    vocab = build_vocab(tokens)
    print(f"Vocabulary size: {len(vocab)}")

    tokenizer = SimpleTokenizerV1(vocab)

    # Test encode/decode roundtrip on a sample from the text
    sample_text = " ".join(tokens[:30])
    print(f"\nSample text: {sample_text}")

    ids = tokenizer.encode(sample_text)
    print(f"Encoded ids: {ids}")

    decoded_text = tokenizer.decode(ids)
    print(f"Decoded text: {decoded_text}")


if __name__ == "__main__":
    main()
