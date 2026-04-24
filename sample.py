import tiktoken
import torch
import os, time
from model import SybilMDM
from params import (VOCAB_SIZE, DEVICE, ANSWER_LENGTH, N_ATTENTION_HEADS,
                    SAMPLING_STEPS, MASK_TOKEN_ID, FFN_DIMS,
                    N_TRANSFORMER_BLOCKS, D_MODEL, T_EMB_DIMS)

def main():
    enc = tiktoken.get_encoding("gpt2")

    model = SybilMDM(
    vocab_size=VOCAB_SIZE,
    n_transformer_blocks=N_TRANSFORMER_BLOCKS,
    d_model=D_MODEL,
    n_attention_heads=N_ATTENTION_HEADS,
    ffn_dims=FFN_DIMS,
    t_emb_dims=T_EMB_DIMS
    ).to(DEVICE)

    ckpt = torch.load("weights/ckpt_1000.pt")
    model.load_state_dict(ckpt["model"])
    model.eval()

    r_t = torch.full((ANSWER_LENGTH,), 50257).to(DEVICE)
    r0 = r_t.clone()

    t = torch.tensor(1.0, device=DEVICE)

    current_step = 0

    while t > 0:
        s = t - (1.0 / SAMPLING_STEPS)

        logits = model(r_t.unsqueeze(0), t.unsqueeze(0))
        r0 = torch.argmax(logits, dim=-1).squeeze(0)

        unmasked = (r_t != MASK_TOKEN_ID)
        masked = (r_t == MASK_TOKEN_ID)

        r0[unmasked] = r_t[unmasked]

        remask = torch.rand(r_t.shape, device=DEVICE) < (s / t)
        r0[masked & remask] = MASK_TOKEN_ID

        r_t = r0
        t = s

        os.system("clear")
        print(f"step {current_step}/{SAMPLING_STEPS}\n\n")

        tokens = r0.tolist()
        output = []
        for tok in tokens:
            if tok == MASK_TOKEN_ID:
                output.append("[MASK]")
            else:
                output.append(enc.decode([tok]))

        text = "".join(output)

        print(text)

        current_step += 1

if __name__ == "__main__":
    main()