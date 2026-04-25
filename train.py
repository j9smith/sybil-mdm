import os
import time
import argparse
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from model import SybilMDM
from params import (DEVICE, BATCH_SIZE, STEPS, MASK_TOKEN_ID, ACCUMULATION_STEPS,
                    VOCAB_SIZE, LOG_EVERY, CKPT_EVERY, N_TRANSFORMER_BLOCKS,
                    D_MODEL, N_ATTENTION_HEADS, FFN_DIMS, T_EMB_DIMS)

WARMUP_STEPS = 2000
MAX_LR = 4e-4

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=False)
    args = p.parse_args()

    os.makedirs("weights", exist_ok=True)
    writer = SummaryWriter("runs/sybil_mdm")

    data = torch.load("wikitext103_tokenised.pt")
    train_dataset = torch.utils.data.TensorDataset(data["train"])
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

    optimiser = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1)
    scheduler = LambdaLR(optimiser, lr_lambda=lambda step: min(1.0, (step + 1) / WARMUP_STEPS))

    current_step = 0
    starting_step = 0

    if args.ckpt is not None:
        print(f"Loading checkpoint: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location="cuda")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimiser.load_state_dict(checkpoint["optimiser"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        starting_step = checkpoint["step"]
        current_step = checkpoint["step"]
        print("Loaded.")

    model.train()

    running_loss = 0.0
    start_time = time.time()

    it = iter(train_loader)
    while current_step <= STEPS + starting_step:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        x_original = batch[0].to(DEVICE)
        t = torch.rand(x_original.shape[0], device=DEVICE).unsqueeze(-1)
        r = torch.rand(x_original.shape, device=DEVICE)

        mask = (r < t)

        x_masked = x_original.clone()
        x_masked[mask] = MASK_TOKEN_ID

        logits = model(x_masked, t)
        loss = F.cross_entropy(logits[mask], x_original[mask])
        loss = loss / ACCUMULATION_STEPS
        loss.backward()

        if current_step % ACCUMULATION_STEPS == 0:
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()

        running_loss += loss.item()
        current_step += 1

        writer.add_scalar("train/loss", loss.item(), current_step)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], current_step)

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
                "scheduler": scheduler.state_dict(),
            }, f"weights/ckpt_{current_step}.pt")

    writer.close()

if __name__ == "__main__":
    main()