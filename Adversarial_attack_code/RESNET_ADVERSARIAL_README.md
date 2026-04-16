# ResNet-18 Adversarial Attack Framework

A Python framework for running adversarial attacks against a ResNet-18 image classifier. Supports four attack methods — FGSM, PGD, C&W, and Adversarial Sticker — all run automatically on a single input image.

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Pillow
- matplotlib
- numpy

Install dependencies:

```bash
pip install torch torchvision pillow matplotlib numpy
```

---

## Usage

```bash
python resnet_adversarial_framework.py --image <path_to_image> [options]
```

All four attacks (FGSM, PGD, C&W, Sticker) are run automatically in sequence. There is no need to specify an attack type.

### Minimal example

```bash
python resnet_adversarial_framework.py --image cat.jpg
```

### Full example with all options

```bash
python resnet_adversarial_framework.py \
  --image cat.jpg \
  --output-dir results \
  --epsilon 0.05 \
  --iterations 50 \
  --target-class "golden retriever" \
  --patch-size 64 \
  --patch-location top-left \
```

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--image` | *(required)* | Path to the input image |
| `--output-dir` | `resnet_adversarial_results` | Root folder for saving outputs |
| `--epsilon` | `0.03` | Max perturbation budget for FGSM and PGD (0–1 scale) |
| `--iterations` | varies | Iteration count override (default: 10 for PGD, 100 for C&W, 200 for Sticker) |
| `--target-class` | `None` | ImageNet class name to target (e.g. `"tabby"`) — runs a targeted attack |
| `--patch-size` | `50` | Width/height in pixels of the adversarial sticker patch |
| `--patch-location` | `center` | Where to place the sticker patch on the image |

### `--patch-location` options
`center`, `top-left`, `top-right`, `bottom-left`, `bottom-right`

---

## Output

Results are saved under:

```
<output-dir>/<image-name>/
```

For example, running on `cat.jpg` with the default output dir produces:

```
resnet_adversarial_results/
└── cat/
    ├── cat_adversarial_fgsm.png
    ├── cat_fgsm_comparison.png
    ├── cat_adversarial_pgd.png
    ├── cat_pgd_comparison.png
    ├── cat_adversarial_cw.png
    ├── cat_cw_comparison.png
    ├── cat_adversarial_sticker.png
    ├── cat_sticker_comparison.png
    └── cat_sticker_sticker.png
```

Each `_comparison.png` shows a side-by-side of the original image, the adversarial image, and (where applicable) the perturbation. The `_sticker_sticker.png` file contains just the isolated adversarial patch.

---

## Attack Methods

**FGSM** (Fast Gradient Sign Method) — Single-step attack. Fast but relatively weak. Perturbs the image by a single step in the direction of the loss gradient.

**PGD** (Projected Gradient Descent) — Iterative version of FGSM. Stronger than FGSM; each step is projected back into the epsilon ball around the original image.

**C&W** (Carlini & Wagner) — Optimization-based attack that minimizes L2 distortion while causing misclassification. Produces more subtle, harder-to-detect perturbations.

**Adversarial Sticker** — Places a visible circular patch on the image and optimizes its appearance to fool the classifier. Unlike the other attacks, this perturbation is intentionally localized and visible.

---

## Targeted vs. Untargeted Attacks

By default, all attacks are **untargeted** — they simply try to make the model predict any class other than the original.

To run a **targeted** attack, pass a valid ImageNet class name via `--target-class`:

```bash
python resnet_adversarial_framework.py --image dog.jpg --target-class "tabby"
```

The framework will attempt to make the model classify the image as that specific class. If the class name is not found in ImageNet, the attack falls back to untargeted mode with a warning.

---

## Example Output (console)

```
============================================================
RESNET-18 ADVERSARIAL ATTACK FRAMEWORK
============================================================

Loading ResNet-18...
Model loaded on cuda

Loading image: cat.jpg
Image size: (640, 480)

Original Classification:
  Top prediction: tabby (82.3%)
  Top-5:
    1. tabby: 82.3%
    2. tiger cat: 10.1%
    ...

============================================================
Running FGSM Attack
============================================================
  ...

============================================================
OVERALL ATTACK SUMMARY
============================================================
Original: tabby (82.3%)

  FGSM       → Egyptian cat              (41.2%)  ✓ CHANGED
  PGD        → Egyptian cat              (65.8%)  ✓ CHANGED
  CW         → tiger cat                 (38.4%)  ✓ CHANGED
  STICKER    → tabby                     (61.1%)  • unchanged
============================================================
```