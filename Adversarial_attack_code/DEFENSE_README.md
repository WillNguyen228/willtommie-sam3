# run_defense.py

A preprocessing-based adversarial defense script. Given a directory of adversarial images produced by the attack frameworks (`run_adversarial_attacks.py` or `run_resnet_adversarial_attacks.py`), it applies one or more image preprocessing techniques to each image before feeding it into the target model and saving the inference results.

---

## How It Works

For each adversarial image found in the input directory, the script:

1. Loads the adversarial image
2. Applies the selected preprocessing defense(s) in order
3. Saves the preprocessed image for reference
4. Runs the preprocessed image through the selected model
5. Saves the inference result visualization

Each run is automatically assigned a unique number so repeated runs never overwrite previous results.

---

## Requirements

**Core dependencies** (shared with the attack scripts):
- `torch` and `torchvision`
- `Pillow`
- `numpy`
- `matplotlib`

**For the `denoise` technique:**
```bash
pip install opencv-python
```

**For the `sam3` model type**, the `sam3` package must be installed and importable in your environment.

---

## Usage

```
python run_defense.py --directory PATH --model-type MODEL --techniques TECHNIQUE [TECHNIQUE ...]
```

All three arguments are required.

### Arguments

| Flag | Short | Description |
|------|-------|-------------|
| `--directory` | `-d` | Path to the adversarial results folder for one class (e.g., `resnet_adversarial_results/cat`) |
| `--model-type` | `-m` | Model to run inference with: `resnet` or `sam3` |
| `--techniques` | `-t` | One or more preprocessing techniques (see below). Space-separated. |

### Preprocessing Techniques

| Value | Description |
|-------|-------------|
| `denoise` | Non-local means denoising via OpenCV. Smooths high-frequency adversarial noise while preserving edges. |
| `resize` | Randomly scales the image between 90–110% and centre-crops back to the original dimensions. Breaks pixel-aligned perturbations. |
| `jpeg` | Round-trips the image through a JPEG buffer at quality 75. Lossy compression degrades fine-grained adversarial structure. |
| `all` | Shorthand to apply all three techniques in the order: `denoise` → `resize` → `jpeg`. |

Multiple techniques can be combined and are applied in the order specified.

---

## Examples

Apply only denoising, evaluate with ResNet:
```bash
python run_defense.py --directory resnet_adversarial_results/cat --model-type resnet --techniques denoise
```

Apply JPEG compression and random resize, evaluate with SAM3:
```bash
python run_defense.py --directory adversarial_results/dog --model-type sam3 --techniques jpeg resize
```

Apply all three defenses using the shorthand:
```bash
python run_defense.py --directory resnet_adversarial_results/cat --model-type resnet --techniques all
```

Using short flags:
```bash
python run_defense.py -d resnet_adversarial_results/cat -m resnet -t denoise resize
```

---

## Input Directory Structure

The script expects a directory that contains adversarial images produced by the attack scripts. It automatically detects files ending in any of these suffixes:

```
<name>_adversarial_fgsm.png
<name>_adversarial_pgd.png
<name>_adversarial_cw.png
<name>_adversarial_sticker.png
```

The class name (used as the detection prompt for SAM3) is inferred from the directory name. For example, passing `adversarial_results/cat` means the class prompt will be `"cat"`.

---

## Output

Results are saved under a defense results directory that is automatically created based on the model type:

```
resnet_defense_results/<class>/   (when --model-type resnet)
sam3_defense_results/<class>/     (when --model-type sam3)
```

For each adversarial image processed, two files are saved:

- `<stem>_preprocessed_<N>.png` — the image after defenses are applied
- `<stem>_defense_result_<N>.png` — inference visualization with predictions overlaid

`<N>` is a run counter that increments automatically each time the script is run, so previous outputs are never overwritten. For example, running the script three times on the same input produces `..._1.png`, `..._2.png`, and `..._3.png`.

### Example output layout

After running the script twice on `resnet_adversarial_results/cat`:

```
resnet_defense_results/
└── cat/
    ├── cat_adversarial_fgsm_preprocessed_1.png
    ├── cat_adversarial_fgsm_defense_result_1.png
    ├── cat_adversarial_pgd_preprocessed_1.png
    ├── cat_adversarial_pgd_defense_result_1.png
    ├── cat_adversarial_cw_preprocessed_1.png
    ├── cat_adversarial_cw_defense_result_1.png
    ├── cat_adversarial_sticker_preprocessed_1.png
    ├── cat_adversarial_sticker_defense_result_1.png
    ├── cat_adversarial_fgsm_preprocessed_2.png
    ├── cat_adversarial_fgsm_defense_result_2.png
    └── ...
```

---

## Project File Overview

| File | Description |
|------|-------------|
| `run_adversarial_attacks.py` | Generates adversarial images targeting the SAM3 model using FGSM, PGD, C&W, and sticker attacks |
| `run_resnet_adversarial_attacks.py` | Generates adversarial images targeting ResNet-18 using the same four attack methods |
| `run_defense.py` | Applies preprocessing defenses to generated adversarial images and re-evaluates them with the target model |