import tiktoken
from datasets import load_dataset
import torch

SEQ_LEN = 256
MASK_TOKEN_ID = 50257

def main():
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("wikitext", "wikitext-103-v1")

    def tokenise_and_chunk(split):
        all_tokens = []
        for instance in ds[split]:
            text = instance["text"].strip()
            if text:
                all_tokens.extend(enc.encode(text))

        all_tokens = torch.tensor(all_tokens, dtype=torch.long)
        n_chunks = len(all_tokens) // SEQ_LEN
        all_tokens = all_tokens[:n_chunks * SEQ_LEN]
        chunks = all_tokens.view(n_chunks, SEQ_LEN)
        return chunks
    
    train = tokenise_and_chunk("train")
    val = tokenise_and_chunk("validation")

    torch.save({"train": train, "val": val}, "wikitext103_tokenised.pt")
    print(f"Train: {train.shape}, val: {val.shape}")
    
if __name__ == "__main__":
    main()