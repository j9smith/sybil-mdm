# SybilMDM

A masked diffusion model for language modelling, based on the LLaDA paper (Nie et al., 2025).

## Architecture

Transformer encoder with modern components:

- **RoPE** for sequence position encoding (Su et al., 2023)
- **SwiGLU** feed-forward network (Shazeer, 2020)
- **AdaLN** for diffusion timestep conditioning (Peebles & Xie, 2022)
- **RMSNorm** with pre-norm residual connections

Timestep is encoded via sinusoidal positional encoding → MLP, then injected into each transformer block through adaptive layer normalization.

## Usage

**Preprocess**

```bash
python preprocess.py
```

Tokenises WikiText-103 with the GPT-2 tokenizer (tiktoken) and chunks into fixed-length sequences.

**Train**

```bash
python train.py
```

Checkpoints are saved to `weights/`.

**Sample**

```bash
python sample.py
```

Starts from a fully masked sequence and iteratively unmasks tokens using the random remasking strategy from LLaDA (Algorithm 4).

## Configuration

All hyperparameters are in `params.py`.

## References

- Nie et al. (2025) — *Large Language Diffusion Models*
- Peebles & Xie (2022) — *Scalable Diffusion Models with Transformers*
- Shazeer (2020) — *GLU Variants Improve Transformer*
- Su et al. (2023) — *RoFormer: Enhanced Transformer with Rotary Position Embedding*