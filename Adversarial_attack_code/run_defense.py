"""
Adversarial Defense Framework
Applies preprocessing defenses to adversarial images before running model inference.

Supported defenses:
  - denoise     : Non-local means image denoising
  - resize      : Random resize and recrop
  - jpeg        : JPEG compression
  - all         : Apply all three techniques

Supported models:
  - resnet      : ResNet-18 classifier
  - sam3        : SAM3 grounding/segmentation model

Usage:
  python run_defense.py --directory <path> --model-type <model> --techniques <list>

  --directory    : Path to an adversarial results folder for one class
                   (e.g., resnet_adversarial_results/cat)
  --model-type   : "resnet" or "sam3"
  --techniques   : One or more of: denoise, resize, jpeg, all
                   Separate multiple values with spaces.

Examples:
  python run_defense.py --directory resnet_adversarial_results/cat --model-type resnet --techniques denoise
  python run_defense.py --directory adversarial_results/dog --model-type sam3 --techniques resize jpeg
  python run_defense.py --directory resnet_adversarial_results/cat --model-type resnet --techniques all
"""

import os
import sys
import argparse
import io
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing / defence helpers
# ─────────────────────────────────────────────────────────────────────────────

def defense_denoise(image: Image.Image) -> Image.Image:
    """
    Non-local means denoising via OpenCV.
    Reduces high-frequency adversarial noise while preserving edges.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for denoising: pip install opencv-python")

    img_np = np.array(image.convert("RGB"))
    # h=10, templateWindowSize=7, searchWindowSize=21 are standard defaults
    denoised = cv2.fastNlMeansDenoisingColored(img_np, None, h=10, hColor=10,
                                               templateWindowSize=7,
                                               searchWindowSize=21)
    return Image.fromarray(denoised)


def defense_resize_recrop(image: Image.Image,
                          scale_min: float = 0.9,
                          scale_max: float = 1.1) -> Image.Image:
    """
    Random resize followed by centre-recrop to the original dimensions.
    Disrupts pixel-aligned adversarial perturbations.
    """
    orig_w, orig_h = image.size
    scale = np.random.uniform(scale_min, scale_max)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Centre-crop back to original size
    left = max(0, (new_w - orig_w) // 2)
    top  = max(0, (new_h - orig_h) // 2)
    right  = left + orig_w
    bottom = top  + orig_h

    # If downscaled, paste onto a black canvas of original size
    if new_w < orig_w or new_h < orig_h:
        canvas = Image.new("RGB", (orig_w, orig_h), (0, 0, 0))
        canvas.paste(resized, (0, 0))
        return canvas

    return resized.crop((left, top, right, bottom))


def defense_jpeg(image: Image.Image, quality: int = 75) -> Image.Image:
    """
    Save to a JPEG byte buffer and reload.
    JPEG's lossy compression degrades fine-grained adversarial perturbations.
    """
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def apply_defenses(image: Image.Image, techniques: list) -> Image.Image:
    """Apply the requested defences in order."""
    for technique in techniques:
        if technique == "denoise":
            print(f"    Applying denoising...")
            image = defense_denoise(image)
        elif technique == "resize":
            print(f"    Applying random resize & recrop...")
            image = defense_resize_recrop(image)
        elif technique == "jpeg":
            print(f"    Applying JPEG compression...")
            image = defense_jpeg(image)
        else:
            print(f"    WARNING: Unknown technique '{technique}', skipping.")
    return image


# ─────────────────────────────────────────────────────────────────────────────
# Model runners
# ─────────────────────────────────────────────────────────────────────────────

def run_resnet(image: Image.Image, image_name: str, output_dir: str, run_num: int):
    """Run preprocessed image through ResNet-18 and save results."""
    import torch
    import torch.nn.functional as F
    from torchvision.models import resnet18, ResNet18_Weights
    from torchvision.transforms import v2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    model.eval()

    categories = ResNet18_Weights.DEFAULT.meta["categories"]
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    # Preprocess
    img_tensor = v2.functional.to_image(image.convert("RGB")).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    if img_tensor.shape[-2:] != (224, 224):
        img_tensor = F.interpolate(img_tensor, size=(224, 224),
                                   mode="bilinear", align_corners=False)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1).to(device)
    normalized = (img_tensor - mean_t) / std_t

    # Inference
    with torch.no_grad():
        logits = model(normalized)
        probs  = F.softmax(logits, dim=1)
        top5_probs, top5_idx = torch.topk(probs, 5, dim=1)

    predictions = [
        {"class": categories[idx.item()], "confidence": prob.item()}
        for prob, idx in zip(top5_probs[0], top5_idx[0])
    ]

    # Console output
    print(f"      Top-5 predictions:")
    for i, p in enumerate(predictions):
        print(f"        {i+1}. {p['class']}: {p['confidence']*100:.1f}%")

    # Save result image with prediction overlay
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(image)
    ax.axis("off")
    label = "\n".join(
        f"{i+1}. {p['class']}: {p['confidence']*100:.1f}%"
        for i, p in enumerate(predictions)
    )
    ax.set_title(f"Defense result\n{label}", fontsize=9, loc="left")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{image_name}_defense_result_{run_num}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {out_path}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return predictions


def run_sam3(image: Image.Image, prompt: str, image_name: str, output_dir: str, run_num: int):
    """Run preprocessed image through SAM3 and save results."""
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.visualization_utils import plot_results

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=0.1)

    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    boxes  = output["boxes"]
    scores = output["scores"]

    if boxes is not None and len(scores) > 0:
        print(f"      Detected {len(boxes)} '{prompt}'(s):")
        for idx, (box, score) in enumerate(zip(boxes, scores)):
            print(f"        #{idx+1}: confidence={score.item():.4f}")
    else:
        print(f"      No '{prompt}' detected.")

    plot_results(image, inference_state)
    out_path = os.path.join(output_dir, f"{image_name}_defense_result_{run_num}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"      Saved: {out_path}")

    del model
    del processor
    torch.cuda.empty_cache()

    return boxes, scores


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

ADVERSARIAL_SUFFIXES = [
    "_adversarial_fgsm.png",
    "_adversarial_pgd.png",
    "_adversarial_cw.png",
    "_adversarial_sticker.png",
]

ALL_TECHNIQUES = ["denoise", "resize", "jpeg"]


def find_adversarial_images(directory: str) -> list:
    """
    Return a list of (file_path, base_stem) for every adversarial image found
    in *directory*.  Looks for files ending in the four known attack suffixes.
    """
    found = []
    for fname in sorted(os.listdir(directory)):
        for suffix in ADVERSARIAL_SUFFIXES:
            if fname.endswith(suffix):
                found.append((os.path.join(directory, fname),
                               os.path.splitext(fname)[0]))
                break
    return found


def infer_prompt(directory: str) -> str:
    """
    Derive the object prompt from the directory name.
    e.g.  "resnet_adversarial_results/cat"  →  "cat"
    """
    return os.path.basename(os.path.normpath(directory))


def next_run_number(output_dir: str, stem: str) -> int:
    """
    Scan output_dir for existing files matching  <stem>_defense_result_<N>.png
    and return the next unused integer N (starting at 1).
    """
    existing = set()
    if os.path.isdir(output_dir):
        prefix = f"{stem}_defense_result_"
        for fname in os.listdir(output_dir):
            if fname.startswith(prefix) and fname.endswith(".png"):
                middle = fname[len(prefix):-len(".png")]
                if middle.isdigit():
                    existing.add(int(middle))
    n = 1
    while n in existing:
        n += 1
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply preprocessing defences to adversarial images and re-evaluate them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--directory", "-d",
        required=True,
        metavar="PATH",
        help="Path to adversarial results directory for one class "
             "(e.g., resnet_adversarial_results/cat)",
    )
    parser.add_argument(
        "--model-type", "-m",
        required=True,
        choices=["resnet", "sam3"],
        dest="model_type",
        metavar="MODEL",
        help='Model to evaluate with: "resnet" or "sam3"',
    )
    parser.add_argument(
        "--techniques", "-t",
        required=True,
        nargs="+",
        choices=["denoise", "resize", "jpeg", "all"],
        metavar="TECHNIQUE",
        help="One or more preprocessing techniques: denoise, resize, jpeg, all. "
             'Use "all" to apply every technique. '
             "Example: --techniques denoise resize",
    )
    args = parser.parse_args()

    # ── Validate directory ────────────────────────────────────────────────────
    if not os.path.isdir(args.directory):
        print(f"ERROR: '{args.directory}' is not a valid directory.")
        sys.exit(1)

    # ── Resolve techniques (expand "all") ────────────────────────────────────
    if "all" in args.techniques:
        techniques = ALL_TECHNIQUES
    else:
        # Preserve order while removing duplicates
        seen = set()
        techniques = []
        for t in args.techniques:
            if t not in seen:
                seen.add(t)
                techniques.append(t)

    # ── Determine output directory ────────────────────────────────────────────
    class_name   = infer_prompt(args.directory)
    results_root = (
        "resnet_defense_results" if args.model_type == "resnet"
        else "sam3_defense_results"
    )
    output_dir = os.path.join(results_root, class_name)
    os.makedirs(output_dir, exist_ok=True)

    # ── Find adversarial images ───────────────────────────────────────────────
    adv_images = find_adversarial_images(args.directory)
    if not adv_images:
        print(f"No adversarial images found in '{args.directory}'.")
        print("Expected files matching patterns like: *_adversarial_fgsm.png, etc.")
        sys.exit(1)

    # ── Banner ────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("ADVERSARIAL DEFENSE FRAMEWORK")
    print("=" * 70)
    print(f"  Directory   : {args.directory}")
    print(f"  Model       : {args.model_type}")
    print(f"  Techniques  : {', '.join(techniques)}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Images found: {len(adv_images)}")
    print("=" * 70)

    # ── Process each adversarial image ───────────────────────────────────────
    for img_path, stem in adv_images:
        print(f"\n  Processing: {os.path.basename(img_path)}")

        # Determine a unique run number for this stem before writing anything
        run_num = next_run_number(output_dir, stem)

        # Load
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"    ERROR loading image: {e}")
            continue

        # Apply defences
        print(f"  Applying defenses: {', '.join(techniques)}")
        defended = apply_defenses(image, techniques)

        # Save the preprocessed image for reference (also versioned)
        preprocessed_path = os.path.join(output_dir,
                                         f"{stem}_preprocessed_{run_num}.png")
        defended.save(preprocessed_path)
        print(f"    Saved preprocessed image: {preprocessed_path}")

        # Run model
        print(f"  Running {args.model_type.upper()} inference...")
        try:
            if args.model_type == "resnet":
                run_resnet(defended, stem, output_dir, run_num)
            else:  # sam3
                run_sam3(defended, class_name, stem, output_dir, run_num)
        except Exception as e:
            print(f"    ERROR during inference: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'=' * 70}")
    print(f"Defense evaluation complete.")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
