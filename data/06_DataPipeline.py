"""
End-to-end data pipeline for the-verdict.txt

Chains together everything built so far into the pipeline an LLM actually
trains on:

  1. BPE tokenization (tiktoken, GPT-2 encoding) -- raw text -> token ids
  2. Sliding-window sampling -- token ids -> (input, target) chunk pairs,
     wrapped in a PyTorch Dataset/DataLoader for batching
  3. Token embeddings -- token ids -> dense vectors (nn.Embedding lookup)
  4. Positional embeddings -- inject "where in the sequence" information,
     since token embeddings alone are position-agnostic (attention has no
     built-in sense of order)

  Final output: input_embeddings = token_embeddings + positional_embeddings
  This is exactly what gets fed into the first transformer block.
"""

import os

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# 1. Sliding-window dataset
# ---------------------------------------------------------------------------
class GPTDatasetV1(Dataset):
    """
    Turns a long stream of token ids into many fixed-length training
    examples using a sliding window.

    For window size `max_length` and step size `stride`:
      input_chunk  = token_ids[i : i+max_length]
      target_chunk = token_ids[i+1 : i+max_length+1]   (i.e. input shifted right by 1)

    The target is the input shifted by one position because the model's
    job at every position is simply "predict the next token".
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt, batch_size=4, max_length=256, stride=128,
    shuffle=True, drop_last=True, num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


def main():
    torch.manual_seed(123)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "the-verdict.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # -----------------------------------------------------------------
    # Step 1: BPE tokenization (sanity check, standalone)
    # -----------------------------------------------------------------
    tokenizer = tiktoken.get_encoding("gpt2")
    all_token_ids = tokenizer.encode(raw_text)
    vocab_size = tokenizer.n_vocab
    print("--- Step 1: BPE tokenization ---")
    print(f"Total characters: {len(raw_text)}")
    print(f"Total BPE tokens: {len(all_token_ids)}")
    print(f"Vocab size (GPT-2 BPE): {vocab_size}")
    print(f"First 10 token ids: {all_token_ids[:10]}")

    # -----------------------------------------------------------------
    # Step 2: Sliding-window sampling -> batches of (input, target) pairs
    # -----------------------------------------------------------------
    max_length = 4    # how many tokens per training example (context length)
    stride = max_length  # non-overlapping windows (stride == max_length)
    batch_size = 8

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False,
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("\n--- Step 2: Sliding-window sampling ---")
    print(f"max_length (context length): {max_length}, stride: {stride}, batch_size: {batch_size}")
    print(f"Inputs shape:  {inputs.shape}   (batch_size, max_length)")
    print(f"Targets shape: {targets.shape}   (batch_size, max_length)")
    print("First example in batch:")
    print(f"  input:  {inputs[0].tolist()} -> {tokenizer.decode(inputs[0].tolist())!r}")
    print(f"  target: {targets[0].tolist()} -> {tokenizer.decode(targets[0].tolist())!r}")
    print("  (target is input shifted right by 1 token -- 'predict the next token')")

    # -----------------------------------------------------------------
    # Step 3: Token embeddings
    # -----------------------------------------------------------------
    output_dim = 256  # size of each token's embedding vector
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(inputs)

    print("\n--- Step 3: Token embeddings ---")
    print(f"Embedding table shape: {tuple(token_embedding_layer.weight.shape)}   (vocab_size, output_dim)")
    print(f"Token embeddings shape for this batch: {tuple(token_embeddings.shape)}   (batch_size, max_length, output_dim)")

    # -----------------------------------------------------------------
    # Step 4: Positional embeddings
    # -----------------------------------------------------------------
    # Token embeddings alone carry no notion of ORDER -- the same token id
    # gets the exact same vector no matter where it appears in the
    # sequence. We add a second embedding, indexed purely by position
    # (0, 1, 2, ..., max_length-1), so the model can tell "1st token" apart
    # from "3rd token" even if they're the same word.
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))

    print("\n--- Step 4: Positional embeddings ---")
    print(f"Positional embedding table shape: {tuple(pos_embedding_layer.weight.shape)}   (context_length, output_dim)")
    print(f"Positional embeddings shape: {tuple(pos_embeddings.shape)}   (one vector per position, broadcasts over the batch)")

    # -----------------------------------------------------------------
    # Final: combine into the actual input to the transformer
    # -----------------------------------------------------------------
    input_embeddings = token_embeddings + pos_embeddings

    print("\n--- Final: input embeddings (token + positional) ---")
    print(f"input_embeddings shape: {tuple(input_embeddings.shape)}   (batch_size, max_length, output_dim)")
    print("This is what gets fed into the first transformer block.")


if __name__ == "__main__":
    main()
