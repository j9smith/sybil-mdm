import os
import time
import torch
import torch.nn.functional as F
from model import SybilMDM
from params import (DEVICE, BATCH_SIZE, STEPS, MASK_TOKEN_ID,
                    VOCAB_SIZE, LOG_EVERY, CKPT_EVERY, N_TRANSFORMER_BLOCKS,
                    D_MODEL, N_ATTENTION_HEADS, FFN_DIMS, T_EMB_DIMS)

def main():
    os.makedirs("weights", exist_ok=True)
    data = torch.load("wikitext103_tokenised.pt")

    train_dataset = torch.utils.data.TensorDataset(data["train"][:1000])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = SybilMDM(
    vocab_size=VOCAB_SIZE,
    n_transformer_blocks=N_TRANSFORMER_BLOCKS,
    d_model=D_MODEL,
    n_attention_heads=N_ATTENTION_HEADS,
    ffn_dims=FFN_DIMS,
    t_emb_dims=T_EMB_DIMS
    ).to(DEVICE)

    optimiser = torch.optim.AdamW(model.parameters(), lr=4e-4)

    model.train()

    current_step = 0
    running_loss = 0.0
    start_time = time.time()

    it = iter(train_loader)
    while current_step <= STEPS:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        optimiser.zero_grad()

        x_original = batch[0].to(DEVICE)
        t = torch.rand(x_original.shape[0], device=DEVICE).unsqueeze(-1)
        r = torch.rand(x_original.shape, device=DEVICE)

        mask = (r > t)

        x_masked = x_original.clone()
        x_masked[mask] = MASK_TOKEN_ID

        logits = model(x_masked, t)
        loss = F.cross_entropy(logits[mask], x_original[mask])
        loss.backward()
        optimiser.step()

        running_loss += loss.item()
        current_step += 1

        if current_step % LOG_EVERY == 0:
            avg_loss = running_loss / LOG_EVERY
            elapsed = time.time() - start_time
            steps_per_sec = current_step / elapsed
            print(f"step {current_step}/{STEPS} | loss {avg_loss:.4f} | {steps_per_sec:.1f} steps/s")
            running_loss = 0.0

        if current_step % CKPT_EVERY == 0:
            torch.save({
                "step": current_step,
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
            }, f"weights/ckpt_{current_step}.pt")


if __name__ == "__main__":
    main()