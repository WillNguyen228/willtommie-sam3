import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline, DDIMScheduler


def generate_images(pipe, num_images=4, num_steps=50, seed=None):
    """Generate images using the diffusion pipeline."""
    # Set seed=42 (or any integer) for reproducible results
    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None
    result = pipe(
        batch_size=num_images,
        generator=generator,
        num_inference_steps=num_steps,
    )
    return result.images


def show_images(images, titles=None, suptitle=None):
    """Display a row of PIL images."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis("off")
        if titles:
            axes[i].set_title(titles[i], fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.show()


def pil_to_tensor(pil_images):
    """Convert list of PIL images to a batch tensor (N, 3, 32, 32) in [0, 1]."""
    tensors = []
    for img in pil_images:
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        tensors.append(t)
    return torch.stack(tensors)


if __name__ == "__main__":
    # Load the diffusion model (unconditional DDPM trained on CIFAR-10)
    print("Loading diffusion model...")
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Checkpoint A: Generate and visualize images
    # TODO: generate 4 images using generate_images() with 50 steps
    # TODO: display them using show_images()
    images = generate_images(pipe, num_images=4, num_steps=50)

    print("Checkpoint A:")
    print(f"  Generated {len(images)} images")
    print(f"  Image size: {images[0].size}")
    show_images(images, suptitle="Checkpoint A: Generated CIFAR-10 Images")
    print()

    # Checkpoint B: Effect of number of denoising steps
    # Use a FIXED seed here so the same noise is used across all step counts.
    # Observe how image quality changes with more denoising steps.
    #
    # TODO: generate 1 image with steps=5, seed=0
    # TODO: generate 1 image with steps=10, seed=0
    # TODO: generate 1 image with steps=25, seed=0
    # TODO: generate 1 image with steps=50, seed=0
    # TODO: display all 4 in a row using show_images() with step counts as titles
    step_counts = [5, 10, 25, 50]
    step_images = []
    for steps in step_counts:
        img = generate_images(pipe, num_images=1, num_steps=steps, seed=0)
        step_images.append(img[0])

    print("Checkpoint B:")
    print(f"  Comparing {step_counts} denoising steps with same seed")
    show_images(step_images, titles=[f"{s} steps" for s in step_counts],
                suptitle="Checkpoint B: Quality vs. Number of Steps")
    print()

    # Checkpoint C: What does a classifier think these are?
    # Load a pretrained ImageNet classifier (ResNet-18) and run it on generated
    # images. The diffusion model generates 32x32 images, but ResNet expects
    # 224x224, so we upsample first.

    # IMPORTANT: Seeing if two models align or not if one predicts and one generated from the same same dataset
    #
    # TODO: load ResNet-18 with pretrained weights, set to eval mode
    #   from torchvision.models import resnet18, ResNet18_Weights
    #   classifier = resnet18(weights=ResNet18_Weights.DEFAULT)
    from torchvision.models import resnet18, ResNet18_Weights
    classifier = resnet18(weights=ResNet18_Weights.DEFAULT)
    classifier.eval()
    
    # TODO: convert your Checkpoint A images to a tensor with pil_to_tensor()
    batch = pil_to_tensor(images)
    
    # TODO: upsample from 32x32 to 224x224 using F.interpolate with mode="bilinear"
    batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
    
    # TODO: normalize with ImageNet stats: mean=(0.485, 0.456, 0.406),
    #       std=(0.229, 0.224, 0.225)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    batch = (batch - mean) / std
    
    # TODO: run the classifier (no grad needed), get predicted class and confidence
    #   logits = classifier(batch)
    #   probs = F.softmax(logits, dim=1)
    #   conf, pred = probs.max(dim=1)
    with torch.no_grad():
        logits = classifier(batch)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    
    # TODO: print the predicted class name and confidence for each image
    #   use ResNet18_Weights.DEFAULT.meta["categories"] for class names
    categories = ResNet18_Weights.DEFAULT.meta["categories"]
    print("Checkpoint C:")
    print("  Classifying generated images...")
    print("  Checkpoint A images:")
    for i, (c, p) in enumerate(zip(conf, pred)):
        print(f"    Image {i+1}: {categories[p]} ({c.item()*100:.1f}% confidence)")
    
    # TODO: repeat for the step_images from Checkpoint B.
    #       How does the classifier's confidence change with more steps?
    batch_steps = pil_to_tensor(step_images)
    batch_steps = F.interpolate(batch_steps, size=(224, 224), mode="bilinear", align_corners=False)
    batch_steps = (batch_steps - mean) / std
    
    with torch.no_grad():
        logits_steps = classifier(batch_steps)
    probs_steps = F.softmax(logits_steps, dim=1)
    conf_steps, pred_steps = probs_steps.max(dim=1)
    
    print("\n  Checkpoint B images (effect of steps):")
    for i, (steps, c, p) in enumerate(zip(step_counts, conf_steps, pred_steps)):
        print(f"    {steps} steps: {categories[p]} ({c.item()*100:.1f}% confidence)")
    
    print()

    # BONUS: generate 64 images and tally up the class distribution.
    # What ImageNet classes show up most? Do they match CIFAR-10 categories?
    #TODO: What is the question trying to ask here?
    # This one has no prompt so the model can choose whatever. 
    # We're doing this to see if the have biases

    print("BONUS: Class distribution analysis")
    print("  Generating 64 images...")
    bonus_images = generate_images(pipe, num_images=64, num_steps=50)
    
    print("  Classifying all images...")
    bonus_batch = pil_to_tensor(bonus_images)
    bonus_batch = F.interpolate(bonus_batch, size=(224, 224), mode="bilinear", align_corners=False)
    bonus_batch = (bonus_batch - mean) / std
    
    with torch.no_grad():
        bonus_logits = classifier(bonus_batch)
    bonus_probs = F.softmax(bonus_logits, dim=1)
    bonus_conf, bonus_pred = bonus_probs.max(dim=1)
    
    # Tally up the class distribution
    from collections import Counter
    class_counts = Counter()
    for pred in bonus_pred:
        class_counts[categories[pred.item()]] += 1
    
    print("\n  Top 10 predicted ImageNet classes:")
    for class_name, count in class_counts.most_common(10):
        print(f"    {class_name}: {count} images ({count/64*100:.1f}%)")
    
    # Image net is a lot bigger
    print("  Note: ImageNet has 1000 classes while CIFAR-10 has only 10.")
