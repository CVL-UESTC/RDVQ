# RDVQ Training Manual

> See [README.md](README.md) for environment setup, dataset preparation, and model architecture.

---

## 1. Training Pipeline Overview

```
Stage 1               Stage 2              Stage 3
Tokenizer             Entropy              Joint RD
Pretrain    ────→     Pretrain    ────→    Tuning
(256px)               (256px)              (256px, progressive λ)
                                            │
                                            ▼
                                     High-Res Fine-Tune
                                     OpenImage 512
                                     DF2K 512–2048
```

| Stage | Objective | Frozen | Data | Iters | BS | LR | Precision |
|:-----:|------|:---:|------|:---:|:---:|:---:|:---:|
| 1 | $\mathcal{L}_D$ | — | ImageNet 256 | 7×10⁵ | 32 | 1×10⁻⁴ | BF16 |
| 2 | $\mathcal{L}_R$ | tokenizer + codebook | ImageNet 256 | 4×10⁵ | 80 | 1×10⁻⁴ | BF16 |
| 3 | $\mathcal{L}_D + \lambda \mathcal{L}_R$ | codebook | ImageNet 256 | progressive | 8–16 | 1×10⁻⁵~1×10⁻⁴ | FP32 |

**Optimization objective**:

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{codebook}} + \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{LPIPS}} + 0.1\mathcal{L}_{\text{GAN}}}_{\mathcal{L}_D \text{ — distortion}} + \lambda \cdot \underbrace{\mathrm{CE}(p_{\text{soft}}, q_{\psi})}_{\mathcal{L}_R \text{ — rate}}
$$

---

## 2. Stage 1: Tokenizer Pretrain

### Objective

Train the VQ-VAE encoder, decoder, and codebook using only the distortion loss. No entropy model at this stage — focus entirely on learning a high-quality codebook and reconstruction.

### Command

```bash
conda activate RDVQ
cd /path/to/RDVQ_OpenSource

IMAGENET=/path/to/imagenet/train/
RESULTS=/path/to/results/

bash scripts/tokenizer/train_vq.sh \
  --data-path ${IMAGENET} \
  --image-size 256 \
  --vq-model VQ-16-32-64_quant_once \
  --dataset openimage \
  --global-batch-size 32 \
  --results-dir ${RESULTS}/s1_tokenizer \
  --codebook-size 4096 \
  --codebook-embed-dim 32 \
  --entropy-loss-ratio 0.0 \
  --entropy-loss-ratio-init 0.0 \
  --lr 1e-4 \
  --disc-lr 1e-4 \
  --wo-attn
```

### Key Parameters

| Parameter | Value | Description |
|------|:---:|------|
| `--entropy-loss-ratio` | 0.0 | ★ Disables rate loss |
| `--use-predictor` | (default false) | ★ No AR predictor created |
| `--wo-attn` | true | No attention in encoder/decoder |
| `--global-batch-size` | 32 | 8 per GPU with 4 GPUs |
| `--lr` / `--disc-lr` | 1e-4 | |
| `--mixed-precision` | bf16 (default) | |

### Expected Output

```
${RESULTS}/s1_tokenizer/000-VQ-16-32-64_quant_once/
├── checkpoints/
│   ├── 0050000.pt
│   ├── 0100000.pt
│   └── best.pt
├── tensorboard/
└── log.txt
```

---

## 3. Stage 2: Entropy Pretrain

### Objective

Freeze the tokenizer (encoder + decoder) and codebook. Train **only** the AR entropy predictor (12-layer Masked Transformer) to accurately model the conditional distribution over codebook indices. Only rate loss is needed — no GAN or discriminator.

### Command

```bash
S1_CKPT=${RESULTS}/s1_tokenizer/*/checkpoints/best.pt

bash scripts/tokenizer/train_vq.sh \
  --data-path ${IMAGENET} \
  --image-size 256 \
  --vq-model VQ-16-32-64_quant_once \
  --dataset openimage \
  --global-batch-size 80 \
  --results-dir ${RESULTS}/s2_entropy \
  --codebook-size 4096 \
  --codebook-embed-dim 32 \
  --entropy-loss-ratio 1.0 \
  --entropy-loss-ratio-init 1.0 \
  --vq-ckpt ${S1_CKPT} \
  --lr 1e-4 \
  --use-predictor \
  --pretrain-entropy \
  --freeze-codebook \
  --wo-attn \
  --not-load-strict
```

### Freeze Mechanism

`--pretrain-entropy` automatically applies the following freezing:

| Module | Status | Notes |
|------|:---:|------|
| `encoder` | frozen | |
| `decoder` | frozen | |
| `discriminator` | frozen | not used in this stage |
| `codebook` (`--freeze-codebook`) | frozen | |
| `condition_entropy_small` | **training** | ★ only this module is updated |

### Key Parameters

| Parameter | Value | Description |
|------|:---:|------|
| `--pretrain-entropy` | true | ★ Train only the entropy predictor |
| `--use-predictor` | true | ★ Create the AR predictor |
| `--freeze-codebook` | true | Freeze codebook |
| `--global-batch-size` | 80 | Large batch (no discriminator) |
| `--not-load-strict` | true | Allow missing entropy module weights in checkpoint |

---

## 4. Stage 3: Joint RD Tuning

### Objective

Freeze the codebook. Jointly optimize the encoder, decoder, discriminator, and entropy predictor. Different λ values control the **rate–distortion trade-off**, producing models at different bitrate levels.

### Piecewise RD Strategy

The softmax temperature τ controls the sharpness of $p_{\text{soft}}$: smaller τ → sharper distribution → stronger rate constraint.

| Bitrate Regime | τ | λ | Description |
|------|:---:|:---:|------|
| bpp < 0.025 | **0.1** | 4.8, 7.2, 12 | Smoother relaxation, more stable at low bitrates |
| bpp > 0.025 | **0.01** | 0.8, 1.2 | Sharper relaxation, precise rate constraint |

Training follows a **progressive λ curriculum** from high to low bitrates. All models use **FP32** precision from Stage 3 onward for stable optimization.

### 4.1 λ = 4.8 (Low Bitrate)

```bash
S2_CKPT=${RESULTS}/s2_entropy/*/checkpoints/best.pt

bash scripts/tokenizer/train_vq.sh \
  --data-path ${IMAGENET} \
  --image-size 256 \
  --vq-model VQ-16-32-64_quant_once \
  --dataset openimage \
  --global-batch-size 16 \
  --results-dir ${RESULTS}/s3_rd_lambda4.8 \
  --codebook-size 4096 \
  --codebook-embed-dim 32 \
  --entropy-loss-ratio 4.8 \
  --entropy-loss-ratio-init 4.8 \
  --vq-ckpt ${S2_CKPT} \
  --lr 1e-4 \
  --use-predictor \
  --wo-attn \
  --freeze-codebook \
  --warmup \
  --tau 0.1 \
  --mixed-precision none \
  --not-load-strict
```

### 4.2 λ = 1.2 (Medium-High Bitrate)

```bash
bash scripts/tokenizer/train_vq.sh \
  --data-path ${IMAGENET} \
  --image-size 256 \
  --vq-model VQ-16-32-64_quant_once \
  --dataset openimage \
  --global-batch-size 8 \
  --results-dir ${RESULTS}/s3_rd_lambda1.2 \
  --codebook-size 4096 \
  --codebook-embed-dim 32 \
  --entropy-loss-ratio 1.2 \
  --entropy-loss-ratio-init 1.2 \
  --vq-ckpt ${S2_CKPT} \
  --lr 1e-5 \
  --use-predictor \
  --wo-attn \
  --freeze-codebook \
  --mixed-precision none \
  --tau 0.01 \
  --not-load-strict
```

### 4.3 λ = 0.8 (High Bitrate)

```bash
bash scripts/tokenizer/train_vq.sh \
  --data-path ${IMAGENET} \
  --image-size 256 \
  --vq-model VQ-16-32-64_quant_once \
  --dataset openimage \
  --global-batch-size 16 \
  --results-dir ${RESULTS}/s3_rd_lambda0.8 \
  --codebook-size 4096 \
  --codebook-embed-dim 32 \
  --entropy-loss-ratio 0.8 \
  --entropy-loss-ratio-init 0.8 \
  --vq-ckpt ${S2_CKPT} \
  --lr 1e-4 \
  --use-predictor \
  --wo-attn \
  --freeze-codebook \
  --warmup \
  --tau 0.01 \
  --not-load-strict
```

### Stage 3 Freeze Summary

| Module | Status |
|------|:---:|
| codebook (`--freeze-codebook`) | frozen |
| encoder | training |
| decoder | training |
| discriminator | training |
| entropy predictor | training |
| `--warmup` | gradually increase entropy loss weight |

---

## 5. High-Resolution Fine-Tune

### Objective

Models trained solely on ImageNet 256×256 generalize poorly to high-resolution images. Fine-tuning on higher-resolution data improves cross-resolution RD performance.

### 5.1 OpenImage 512 Fine-Tune

```bash
S3_CKPT=${RESULTS}/s3_rd_lambda4.8/*/checkpoints/best.pt
OPENIMAGE=/path/to/openimages/train/

bash scripts/tokenizer/train_vq.sh \
  --data-path ${OPENIMAGE} \
  --image-size 512 \
  --vq-model VQ-16-32-64_quant_once \
  --dataset openimage \
  --global-batch-size 4 \
  --results-dir ${RESULTS}/hr_openimage_512 \
  --codebook-size 4096 \
  --codebook-embed-dim 32 \
  --entropy-loss-ratio 4.8 \
  --entropy-loss-ratio-init 4.8 \
  --vq-ckpt ${S3_CKPT} \
  --lr 1e-5 \
  --disc-lr 1e-5 \
  --use-predictor \
  --wo-attn \
  --freeze-codebook \
  --mixed-precision none \
  --tau 0.1 \
  --not-load-strict
```

| Parameter | Value |
|------|:---:|
| Resolution | 512×512 |
| Batch Size | 4 |
| LR | 1×10⁻⁵ |
| Iters | 4×10⁵ |
| GPU | 4× RTX 4090 |

### 5.2 DF2K Multi-Resolution Fine-Tune

```bash
HR1_CKPT=${RESULTS}/hr_openimage_512/*/checkpoints/best.pt
DF2K=/path/to/DF2K/train/

bash scripts/tokenizer/train_vq.sh \
  --data-path ${DF2K} \
  --image-size 2048 \
  --vq-model VQ-16-32-64_quant_once \
  --dataset openimage \
  --global-batch-size 1 \
  --results-dir ${RESULTS}/hr_df2k_mrs_2048 \
  --codebook-size 4096 \
  --codebook-embed-dim 32 \
  --entropy-loss-ratio 0.8 \
  --entropy-loss-ratio-init 0.8 \
  --vq-ckpt ${HR1_CKPT} \
  --lr 5e-6 \
  --disc-lr 5e-6 \
  --use-predictor \
  --wo-attn \
  --freeze-codebook \
  --MRS_tuning \
  --mixed-precision none \
  --not-load-strict
```

| Parameter | Value |
|------|:---:|
| Resolution | 512–2048 MRS crop schedule |
| Batch Size | 1 |
| LR | 5×10⁻⁶ |
| GPU | 1× RTX Pro 6000 (large resolution requires high VRAM) |

> **Note**: `--MRS_tuning` enables the 512-to-`--image-size` multi-resolution schedule. Without it, training uses the fixed `--image-size` crop.

---

## 6. Full Training Pipeline (End-to-End)

Complete script for training a low-bitrate RDVQ model ($\lambda=4.8$, $\tau=0.1$) from scratch:

```bash
#!/bin/bash
set -euo pipefail

conda activate RDVQ
cd /path/to/RDVQ_OpenSource

# ========== Paths ==========
IMAGENET=/path/to/imagenet/train/
OPENIMAGE=/path/to/openimages/train/
DF2K=/path/to/DF2K/train/
RESULTS=/path/to/results/

# ========== Stage 1: Tokenizer Pretrain ==========
bash scripts/tokenizer/train_vq.sh \
  --data-path ${IMAGENET} --image-size 256 \
  --vq-model VQ-16-32-64_quant_once --dataset openimage \
  --global-batch-size 32 --results-dir ${RESULTS}/s1_tokenizer \
  --codebook-size 4096 --codebook-embed-dim 32 \
  --entropy-loss-ratio 0.0 --entropy-loss-ratio-init 0.0 \
  --lr 1e-4 --disc-lr 1e-4 --wo-attn

S1_CKPT=$(ls -d ${RESULTS}/s1_tokenizer/*/checkpoints/best.pt | tail -1)

# ========== Stage 2: Entropy Pretrain ==========
bash scripts/tokenizer/train_vq.sh \
  --data-path ${IMAGENET} --image-size 256 \
  --vq-model VQ-16-32-64_quant_once --dataset openimage \
  --global-batch-size 80 --results-dir ${RESULTS}/s2_entropy \
  --codebook-size 4096 --codebook-embed-dim 32 \
  --entropy-loss-ratio 1.0 --entropy-loss-ratio-init 1.0 \
  --vq-ckpt ${S1_CKPT} --lr 1e-4 \
  --use-predictor --pretrain-entropy --freeze-codebook \
  --wo-attn --not-load-strict

S2_CKPT=$(ls -d ${RESULTS}/s2_entropy/*/checkpoints/best.pt | tail -1)

# ========== Stage 3: Joint RD Tuning (λ=4.8, low bitrate) ==========
bash scripts/tokenizer/train_vq.sh \
  --data-path ${IMAGENET} --image-size 256 \
  --vq-model VQ-16-32-64_quant_once --dataset openimage \
  --global-batch-size 16 --results-dir ${RESULTS}/s3_lambda4.8 \
  --codebook-size 4096 --codebook-embed-dim 32 \
  --entropy-loss-ratio 4.8 --entropy-loss-ratio-init 4.8 \
  --vq-ckpt ${S2_CKPT} --lr 1e-4 \
  --use-predictor --wo-attn --freeze-codebook --warmup \
  --tau 0.1 --mixed-precision none --not-load-strict

S3_CKPT=$(ls -d ${RESULTS}/s3_lambda4.8/*/checkpoints/best.pt | tail -1)

# ========== High-Res: OpenImage 512 ==========
bash scripts/tokenizer/train_vq.sh \
  --data-path ${OPENIMAGE} --image-size 512 \
  --vq-model VQ-16-32-64_quant_once --dataset openimage \
  --global-batch-size 4 --results-dir ${RESULTS}/hr_openimage \
  --codebook-size 4096 --codebook-embed-dim 32 \
  --entropy-loss-ratio 4.8 --entropy-loss-ratio-init 4.8 \
  --vq-ckpt ${S3_CKPT} --lr 1e-5 --disc-lr 1e-5 \
  --use-predictor --wo-attn --freeze-codebook \
  --tau 0.1 --mixed-precision none --not-load-strict

HR1_CKPT=$(ls -d ${RESULTS}/hr_openimage/*/checkpoints/best.pt | tail -1)

# ========== High-Res: DF2K Multi-Resolution ==========
bash scripts/tokenizer/train_vq.sh \
  --data-path ${DF2K} --image-size 2048 \
  --vq-model VQ-16-32-64_quant_once --dataset openimage \
  --global-batch-size 1 --results-dir ${RESULTS}/hr_df2k \
  --codebook-size 4096 --codebook-embed-dim 32 \
  --entropy-loss-ratio 0.8 --entropy-loss-ratio-init 0.8 \
  --vq-ckpt ${HR1_CKPT} --lr 5e-6 --disc-lr 5e-6 \
  --use-predictor --wo-attn --freeze-codebook \
  --MRS_tuning --mixed-precision none --not-load-strict

FINAL=$(ls -d ${RESULTS}/hr_df2k/*/checkpoints/best.pt | tail -1)
echo "Training complete. Final checkpoint: ${FINAL}"
```

---

## 7. Parameter Reference

### 7.1 Freeze Strategy Matrix

| Parameter | S1 | S2 | S3 | High-Res |
|------|:---:|:---:|:---:|:---:|
| `--freeze-codebook` | | ✅ | ✅ | ✅ |
| `--pretrain-entropy` | | ✅ | | |
| `--use-predictor` | | ✅ | ✅ | ✅ |
| `--warmup` | | | ✅ | ✅ |
| `--wo-attn` | ✅ | ✅ | ✅ | ✅ |

### 7.2 λ / τ Selection Matrix

| Target bpp | λ | τ | Typical Use |
|------|:---:|:---:|------|
| > 0.04 (high bitrate) | 0.8 | 0.01 | Quality priority |
| 0.03–0.04 | 1.2 | 0.01 | Balanced |
| 0.02–0.03 | 4.8 | 0.1 | Compression priority |
| 0.015–0.025 | 7.2 | 0.1 | Low bitrate |
| < 0.02 | 12 | 0.1 | Ultra-low bitrate |

> **Rule**: bpp < 0.025 → τ = 0.1, bpp > 0.025 → τ = 0.01

### 7.3 Default Parameter Values

| Parameter | Default | Description |
|------|:---:|------|
| `--vq-model` | VQ-16-32-64_quant_once | Only mainline model |
| `--codebook-size` | 4096 | Shared codebook size |
| `--codebook-embed-dim` | 32 | Codebook vector dimension |
| `--commit-loss-beta` | 0.25 | VQ commit loss coefficient |
| `--reconstruction-weight` | 1.0 | MSE weight |
| `--perceptual-weight` | 1.0 | LPIPS weight |
| `--disc-weight` | 0.1 | GAN loss coefficient |
| `--disc-start` | 25000 | Discriminator start step |
| `--disc-type` | patchgan | patchgan / stylegan |
| `--max-grad-norm` | 1.0 | Gradient clipping threshold |
| `--log-every` | 500 | Logging interval (steps) |
| `--ckpt-every` | 5000 | Checkpoint interval (steps) |
| `--max_steps` | 7,000,000 | Max training steps (early stop) |
| `--num-workers` | 8 | DataLoader workers |

---

## 8. train.sh Reference Index

`train.sh` preserves all historical experiment records. The mapping to training stages:

| Lines | train.sh Label | Stage |
|:---:|------|:---:|
| 1–33 | (active block) | High-Res OpenImage 512 |
| 140–153 | `Pretrain SOTA Model reconstruction` | Stage 1 |
| 171–183 | `Pretrain Entropy` | Stage 2 |
| 186–200 | `Tuning All (BF16)` | Stage 3 (λ=4.8) |
| 203–216 | `Tuning All (FP32)` | Stage 3 (λ=1.2) |
| 218–346 | Patch 512/768/1024/MRS/EMA, etc. | High-Res variants |
| 38–138 | VQ-8-16, VQ-4-8-16, soft VQ, VQGAN, etc. | Legacy experiments (non-mainline) |
