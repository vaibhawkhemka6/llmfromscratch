"""
BPE Tokenizer via tiktoken

Replaces our hand-rolled SimpleTokenizer (V1-V3) with OpenAI's tiktoken
library, which implements Byte Pair Encoding (BPE) -- the same tokenizer
used by GPT-2/GPT-3/GPT-4 style models.

Key differences from SimpleTokenizerV1-V3:
  - No fixed word-level vocabulary built from our tiny training text.
    tiktoken ships a pretrained ~50k-token BPE vocabulary (GPT-2: 50,257
    tokens), trained on a huge, diverse corpus.
  - No <|unk|> is needed. BPE operates on sub-word / byte-level units, so
    *any* string -- including made-up words, typos, or other languages --
    can always be encoded by falling back to smaller and smaller sub-word
    chunks, down to individual bytes if necessary. There is no
    out-of-vocabulary case.
  - <|endoftext|> is a real special token built into the GPT-2 vocabulary
    (id 50256), not something we have to bolt on ourselves.
"""

import os

import tiktoken


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"BPE vocab size: {tokenizer.n_vocab}")

    # --- Test 1: basic encode/decode round-trip ---
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    decoded_text = tokenizer.decode(ids)

    print("\n--- Round-trip test ---")
    print(f"Input text:\n{text!r}")
    print(f"Encoded ids ({len(ids)} tokens): {ids}")
    print(f"Decoded text:\n{decoded_text!r}")
    print(f"Exact match: {text == decoded_text}")

    # --- Test 2: BPE handles made-up / unknown words gracefully (no <|unk|>) ---
    unknown_text = "Akwirw ier"
    ids = tokenizer.encode(unknown_text)
    print("\n--- Unknown-word test (no <|unk|> needed) ---")
    print(f"Input text: {unknown_text!r}")
    print(f"Encoded ids: {ids}")
    for i in ids:
        print(f"  {i} -> {tokenizer.decode([i])!r}")
    print(f"Decoded text: {tokenizer.decode(ids)!r}")

    # --- Test 3: encode/decode the full the-verdict.txt file ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "the-verdict.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    ids = tokenizer.encode(raw_text)
    decoded_text = tokenizer.decode(ids)

    print("\n--- the-verdict.txt round-trip ---")
    print(f"Characters: {len(raw_text)}")
    print(f"BPE tokens: {len(ids)}")
    print(f"First 10 ids: {ids[:10]}")
    print(f"First 10 ids decoded: {tokenizer.decode(ids[:10])!r}")
    print(f"Exact match: {raw_text == decoded_text}")


if __name__ == "__main__":
    main()
