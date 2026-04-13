"""
ResNet Adversarial Attack Framework
Supports FGSM, PGD, C&W, and Adversarial Sticker attacks on ResNet classifier
Adapted from SAM3 adversarial framework for image classification
"""

import torch
import torch.nn.functional as F
import torch.fft as fft
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import os
import sys
import argparse
import numpy as np


class ResNetAdversarialAttacker:
    """Base class for adversarial attacks on ResNet"""
    
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.eval()
        
        # ImageNet normalization stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        # Get class names
        self.categories = ResNet18_Weights.DEFAULT.meta["categories"]
        
    def preprocess_image(self, image):
        """Convert PIL image to tensor and apply transforms"""
        if isinstance(image, Image.Image):
            original_size = image.size  # (width, height)
            
            # Convert to tensor [0, 1]
            image_tensor = v2.functional.to_image(image).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Resize to 224x224 if needed
            if image_tensor.shape[-2:] != (224, 224):
                image_tensor = F.interpolate(image_tensor, size=(224, 224), 
                                            mode='bilinear', align_corners=False)
            
            # Normalize
            normalized = (image_tensor - self.mean) / self.std
            
            return normalized, image_tensor, original_size
        else:
            raise ValueError("Image must be a PIL image")
    
    def compute_loss(self, image_tensor, target_class=None, source_class=None):
        logits = self.model(image_tensor)
        
        if target_class is not None:
            loss = -logits[0, target_class]
        elif source_class is not None:
            loss = -logits[0, source_class]
        else:
            probs = F.softmax(logits, dim=1)
            top_prob = probs.max()
            loss = -top_prob
        
        return loss, logits
    
    def tensor_to_pil(self, tensor, original_size=None):
        """Convert normalized tensor back to PIL image"""
        tensor = tensor.squeeze(0)
        mean = self.mean.squeeze(0).view(3, 1, 1)
        std = self.std.squeeze(0).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        tensor = tensor.cpu()
        image_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        if original_size is not None:
            pil_image = pil_image.resize(original_size, Image.Resampling.LANCZOS)
        
        return pil_image
    
    def get_prediction(self, image_tensor):
        """Get top-k predictions for an image"""
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)
            
            top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
            
            results = []
            for prob, idx in zip(top5_probs[0], top5_indices[0]):
                results.append({
                    'class': self.categories[idx.item()],
                    'index': idx.item(),
                    'confidence': prob.item()
                })
            
            return results


class FGSM_ResNet(ResNetAdversarialAttacker):
    """Fast Gradient Sign Method for ResNet"""
    
    def __init__(self, model, epsilon=0.03, device="cuda"):
        super().__init__(model, device)
        self.epsilon = epsilon
        
    def attack(self, image, source_class=None, target_class=None):
        normalized_tensor, original_tensor, original_size = self.preprocess_image(image)
        normalized_tensor.requires_grad = True
        
        loss, _ = self.compute_loss(normalized_tensor, target_class, source_class)
        
        self.model.zero_grad()
        loss.backward()
        
        gradient = normalized_tensor.grad.data
        perturbation = self.epsilon * gradient.sign()
        perturbed_normalized = normalized_tensor.detach() + perturbation
        
        perturbed_01 = perturbed_normalized * self.std + self.mean
        perturbed_01 = torch.clamp(perturbed_01, 0, 1)
        
        adversarial_image = self.tensor_to_pil(perturbed_normalized, original_size)
        
        return adversarial_image, perturbation, loss.item()


class PGD_ResNet(ResNetAdversarialAttacker):
    """Projected Gradient Descent for ResNet"""
    
    def __init__(self, model, epsilon=0.03, alpha=0.01, iterations=10, device="cuda"):
        super().__init__(model, device)
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        
    def attack(self, image, source_class=None, target_class=None):
        original_normalized, original_01, original_size = self.preprocess_image(image)
        perturbed = original_normalized.clone().detach()
        
        print(f"  Running PGD attack with {self.iterations} iterations...")
        
        for i in range(self.iterations):
            perturbed.requires_grad = True
            
            loss, _ = self.compute_loss(perturbed, target_class, source_class)
            
            self.model.zero_grad()
            loss.backward()
            
            gradient = perturbed.grad.data
            
            with torch.no_grad():
                perturbed = perturbed.detach() + self.alpha * gradient.sign()
                
                perturbation = torch.clamp(
                    perturbed - original_normalized,
                    -self.epsilon,
                    self.epsilon
                )
                perturbed = original_normalized + perturbation
            
            if (i + 1) % 2 == 0:
                print(f"    Iteration {i+1}/{self.iterations}, Loss: {loss.item():.4f}")
        
        adversarial_image = self.tensor_to_pil(perturbed, original_size)
        final_perturbation = perturbed - original_normalized
        
        return adversarial_image, final_perturbation, loss.item()


class CW_ResNet(ResNetAdversarialAttacker):
    """Carlini & Wagner Attack for ResNet"""
    
    def __init__(self, model, c=10.0, iterations=100, learning_rate=0.01, 
                 device="cuda"):
        super().__init__(model, device)
        self.c = c
        self.iterations = iterations
        self.learning_rate = learning_rate
    
    def attack(self, image, source_class=None, target_class=None):
        original_normalized, original_01, original_size = self.preprocess_image(image)
        
        w = torch.zeros_like(original_normalized, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([w], lr=self.learning_rate)
        
        print(f"  Running C&W attack with {self.iterations} iterations...")
        
        best_perturbation = None
        best_loss = float('inf')
        
        for i in range(self.iterations):
            perturbed_01 = 0.5 * (torch.tanh(w) + 1)
            perturbed_normalized = (perturbed_01 - self.mean) / self.std
            
            if target_class is not None:
                logits = self.model(perturbed_normalized)
                target_logit = logits[0, target_class]
                other_logits = torch.cat([logits[0, :target_class], 
                                         logits[0, target_class+1:]])
                max_other = other_logits.max()
                classification_loss = torch.clamp(max_other - target_logit, min=0)
            elif source_class is not None:
                logits = self.model(perturbed_normalized)
                source_logit = logits[0, source_class]
                other_logits = torch.cat([logits[0, :source_class], 
                                         logits[0, source_class+1:]])
                max_other = other_logits.max()
                classification_loss = torch.clamp(source_logit - max_other, min=-10)
            else:
                classification_loss, _ = self.compute_loss(perturbed_normalized, 
                                                          target_class, source_class)
            
            l2_dist = torch.norm(perturbed_01 - original_01)
            total_loss = l2_dist + self.c * classification_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_perturbation = perturbed_normalized.detach().clone()
            
            if (i + 1) % 20 == 0:
                print(f"    Iteration {i+1}/{self.iterations}, Loss: {total_loss.item():.4f}, "
                      f"L2: {l2_dist.item():.4f}")
        
        if best_perturbation is not None:
            adversarial_image = self.tensor_to_pil(best_perturbation, original_size)
            final_perturbation = best_perturbation - original_normalized
        else:
            adversarial_image = self.tensor_to_pil(perturbed_normalized, original_size)
            final_perturbation = perturbed_normalized - original_normalized
        
        return adversarial_image, final_perturbation, best_loss


class AdversarialSticker_ResNet(ResNetAdversarialAttacker):
    """Adversarial Sticker/Patch Attack for ResNet"""
    
    def __init__(self, model, patch_size=50, location='center', 
                 c=10.0, iterations=200, learning_rate=0.1, device="cuda"):
        super().__init__(model, device)
        self.patch_size = patch_size
        self.location = location
        self.c = c
        self.iterations = iterations
        self.learning_rate = learning_rate
    
    def create_circular_mask(self, size, device):
        y, x = torch.meshgrid(torch.arange(size, device=device),
                             torch.arange(size, device=device), indexing='ij')
        center = size / 2.0
        dist = torch.sqrt((x - center + 0.5) ** 2 + (y - center + 0.5) ** 2)
        radius = size / 2.0
        edge_width = 2.0
        mask = torch.clamp((radius - dist) / edge_width + 0.5, 0, 1)
        return mask
    
    def get_patch_location(self, image_shape, patch_size):
        _, _, h, w = image_shape
        
        if self.location == 'center':
            x = (w - patch_size) // 2
            y = (h - patch_size) // 2
        elif self.location == 'top-left':
            x, y = 20, 20
        elif self.location == 'top-right':
            x, y = w - patch_size - 20, 20
        elif self.location == 'bottom-left':
            x, y = 20, h - patch_size - 20
        elif self.location == 'bottom-right':
            x, y = w - patch_size - 20, h - patch_size - 20
        else:
            x = (w - patch_size) // 2
            y = (h - patch_size) // 2
        
        return x, y
    
    def attack(self, image, source_class=None, target_class=None):
        original_normalized, original_01, original_size = self.preprocess_image(image)
        
        circular_mask = self.create_circular_mask(self.patch_size, self.device)
        x, y = self.get_patch_location(original_01.shape, self.patch_size)
        
        full_mask = torch.zeros_like(original_01)
        full_mask[:, :, y:y+self.patch_size, x:x+self.patch_size] = circular_mask.unsqueeze(0).unsqueeze(0)
        
        patch = torch.zeros(1, 3, self.patch_size, self.patch_size, 
                           device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([patch], lr=self.learning_rate)
        
        print(f"  Running Sticker attack with {self.iterations} iterations...")
        print(f"  Patch location: ({x}, {y}), size: {self.patch_size}x{self.patch_size}")
        
        best_patch = None
        best_loss = float('inf')
        
        for i in range(self.iterations):
            patch_01 = 0.5 * (torch.tanh(patch) + 1)
            
            perturbed_01 = original_01.clone()
            patch_region = perturbed_01[:, :, y:y+self.patch_size, x:x+self.patch_size]
            patch_mask = circular_mask.unsqueeze(0).unsqueeze(0)
            perturbed_01[:, :, y:y+self.patch_size, x:x+self.patch_size] = (
                patch_01 * patch_mask + patch_region * (1 - patch_mask)
            )
            
            perturbed_normalized = (perturbed_01 - self.mean) / self.std
            
            if target_class is not None:
                logits = self.model(perturbed_normalized)
                target_logit = logits[0, target_class]
                other_logits = torch.cat([logits[0, :target_class], 
                                         logits[0, target_class+1:]])
                max_other = other_logits.max()
                classification_loss = torch.clamp(max_other - target_logit, min=0)
            elif source_class is not None:
                logits = self.model(perturbed_normalized)
                source_logit = logits[0, source_class]
                other_logits = torch.cat([logits[0, :source_class], 
                                         logits[0, source_class+1:]])
                max_other = other_logits.max()
                classification_loss = torch.clamp(source_logit - max_other, min=-10)
            else:
                classification_loss, _ = self.compute_loss(perturbed_normalized,
                                                          target_class, source_class)
            
            dx = torch.abs(patch[:, :, :, 1:] - patch[:, :, :, :-1])
            dy = torch.abs(patch[:, :, 1:, :] - patch[:, :, :-1, :])
            smoothness_loss = dx.mean() + dy.mean()
            
            total_loss = self.c * classification_loss + 0.01 * smoothness_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_patch = patch_01.detach().clone()
            
            if (i + 1) % 40 == 0:
                print(f"    Iteration {i+1}/{self.iterations}, Loss: {total_loss.item():.4f}")
        
        if best_patch is not None:
            final_01 = original_01.clone()
            patch_region = final_01[:, :, y:y+self.patch_size, x:x+self.patch_size]
            patch_mask = circular_mask.unsqueeze(0).unsqueeze(0)
            final_01[:, :, y:y+self.patch_size, x:x+self.patch_size] = (
                best_patch * patch_mask + patch_region * (1 - patch_mask)
            )
            final_normalized = (final_01 - self.mean) / self.std
            adversarial_image = self.tensor_to_pil(final_normalized, original_size)
            sticker_image = self.tensor_to_pil((best_patch - self.mean) / self.std)
        else:
            adversarial_image = self.tensor_to_pil(perturbed_normalized, original_size)
            sticker_image = None
        
        return adversarial_image, sticker_image, best_loss


def visualize_results(original_img, adversarial_img, original_preds, adv_preds, 
                     attack_name, output_name, perturbation=None, amplify_factor=1.0):
    """Visualize attack results"""
    fig, axes = plt.subplots(1, 3 if perturbation is not None else 2, 
                            figsize=(15 if perturbation is not None else 10, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title(f"Original\n{original_preds[0]['class']}\n"
                     f"{original_preds[0]['confidence']*100:.1f}% confidence",
                     fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(adversarial_img)
    axes[1].set_title(f"After {attack_name}\n{adv_preds[0]['class']}\n"
                     f"{adv_preds[0]['confidence']*100:.1f}% confidence",
                     fontsize=12)
    axes[1].axis('off')
    
    if perturbation is not None:
        pert_np = perturbation.squeeze().cpu().detach().numpy()
        pert_vis = np.transpose(pert_np, (1, 2, 0))
        
        if amplify_factor == 1.0:
            pert_vis = np.clip((pert_vis * 0.5) + 0.5, 0, 1)
            title_suffix = "(raw, no amplification)"
        else:
            if amplify_factor < 0:
                pert_vis = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min() + 1e-8)
                title_suffix = "(auto-scaled)"
            else:
                pert_vis = np.clip((pert_vis * amplify_factor) + 0.5, 0, 1)
                title_suffix = f"(amplified {amplify_factor}x)"
        
        axes[2].imshow(pert_vis)
        axes[2].set_title(f"Perturbation\n{title_suffix}", fontsize=12)
        axes[2].axis('off')
    
    plt.suptitle(f"{attack_name} Attack Results", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization: {output_name}")
    plt.close()


def run_attack(attack_name, model, device, args, original_image, source_class,
               target_class_idx, original_preds, output_dir, image_name):
    """Run a single attack and save its results. Returns a summary dict."""
    print(f"\n{'=' * 60}")
    print(f"Running {attack_name.upper()} Attack")
    print(f"{'=' * 60}")

    if attack_name == 'fgsm':
        attacker = FGSM_ResNet(model, epsilon=args.epsilon, device=device)
        adversarial_image, perturbation, loss = attacker.attack(
            original_image, source_class, target_class_idx
        )

    elif attack_name == 'pgd':
        attacker = PGD_ResNet(model, epsilon=args.epsilon,
                              iterations=args.iterations or 10, device=device)
        adversarial_image, perturbation, loss = attacker.attack(
            original_image, source_class, target_class_idx
        )

    elif attack_name == 'cw':
        attacker = CW_ResNet(model, iterations=args.iterations or 100, device=device)
        adversarial_image, perturbation, loss = attacker.attack(
            original_image, source_class, target_class_idx
        )

    elif attack_name == 'sticker':
        attacker = AdversarialSticker_ResNet(
            model, patch_size=args.patch_size, location=args.patch_location,
            iterations=args.iterations or 200, device=device
        )
        adversarial_image, sticker_image, loss = attacker.attack(
            original_image, source_class, target_class_idx
        )
        perturbation = None
        if sticker_image:
            sticker_path = os.path.join(output_dir, f"{image_name}_sticker_sticker.png")
            sticker_image.save(sticker_path)
            print(f"  Saved sticker: {sticker_path}")

    # Get adversarial prediction
    normalized_adv, _, _ = attacker.preprocess_image(adversarial_image)
    adv_preds = attacker.get_prediction(normalized_adv)

    print(f"\nAdversarial Classification ({attack_name.upper()}):")
    print(f"  Top prediction: {adv_preds[0]['class']} "
          f"({adv_preds[0]['confidence']*100:.1f}%)")
    print(f"  Top-5:")
    for i, pred in enumerate(adv_preds):
        print(f"    {i+1}. {pred['class']}: {pred['confidence']*100:.1f}%")

    # Save adversarial image
    output_path = os.path.join(output_dir, f"{image_name}_adversarial_{attack_name}.png")
    adversarial_image.save(output_path)
    print(f"\n  Saved adversarial image: {output_path}")

    # Save visualization
    viz_path = os.path.join(output_dir, f"{image_name}_{attack_name}_comparison.png")
    visualize_results(original_image, adversarial_image,
                      original_preds, adv_preds,
                      attack_name.upper(), viz_path, perturbation,
                      amplify_factor=args.amplify_perturbation)

    summary = {
        'attack': attack_name,
        'adv_class': adv_preds[0]['class'],
        'adv_confidence': adv_preds[0]['confidence'],
        'adv_index': adv_preds[0]['index'],
        'loss': loss,
    }

    # Free attacker and any lingering tensors before the next attack
    del attacker
    if device == 'cuda':
        torch.cuda.empty_cache()
    print(f"  Memory cleared after {attack_name.upper()} attack.")

    return summary


class Tee:
    """Mirrors stdout to both the console and a log file simultaneously."""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def main():
    parser = argparse.ArgumentParser(description='Adversarial Attacks on ResNet-18')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output-dir', type=str, default='resnet_adversarial_results',
                       help='Output directory')
    parser.add_argument('--epsilon', type=float, default=0.03,
                       help='Perturbation budget for FGSM/PGD')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Number of iterations (default: 10 for PGD, 100 for C&W, 200 for sticker)')
    parser.add_argument('--target-class', type=str, default="horse",
                       help='Target class name for targeted attack')
    parser.add_argument('--patch-size', type=int, default=50,
                       help='Sticker patch size')
    parser.add_argument('--patch-location', type=str, default='center',
                       choices=['center', 'top-left', 'top-right', 'bottom-left', 'bottom-right'],
                       help='Sticker location')
    parser.add_argument('--amplify-perturbation', type=float, default=1.0,
                       help='Amplification factor for perturbation visualization '
                            '(1.0=no amplification, -1=auto-scale, >1=amplify by factor)')

    args = parser.parse_args()

    print("=" * 60)
    print("RESNET-18 ADVERSARIAL ATTACK FRAMEWORK")
    print("=" * 60)

    # Load model
    print("\nLoading ResNet-18...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Load image
    print(f"\nLoading image: {args.image}")
    try:
        original_image = Image.open(args.image).convert('RGB')
        print(f"Image size: {original_image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Derive image name and create output directory scoped to that name
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    output_dir = os.path.join(args.output_dir, image_name)
    os.makedirs(output_dir, exist_ok=True)

    # Mirror all stdout to a log file in the output directory
    log_path = os.path.join(output_dir, f"{image_name}_log.txt")
    tee = Tee(log_path)
    sys.stdout = tee
    print(f"Logging output to: {log_path}")

    # Get original prediction using a base attacker instance
    base_attacker = ResNetAdversarialAttacker(model, device)
    normalized, _, _ = base_attacker.preprocess_image(original_image)
    original_preds = base_attacker.get_prediction(normalized)
    source_class = original_preds[0]['index']

    print("\nOriginal Classification:")
    print(f"  Top prediction: {original_preds[0]['class']} "
          f"({original_preds[0]['confidence']*100:.1f}%)")
    print(f"  Top-5:")
    for i, pred in enumerate(original_preds):
        print(f"    {i+1}. {pred['class']}: {pred['confidence']*100:.1f}%")

    # Handle target class
    target_class_idx = None
    if args.target_class:
        try:
            target_class_idx = base_attacker.categories.index(args.target_class)
            print(f"\nTarget class: {args.target_class} (index {target_class_idx})")
        except ValueError:
            print(f"\nWarning: '{args.target_class}' not found in ImageNet classes")
            print("Running untargeted attacks instead")

    # Free base attacker before running attacks
    del base_attacker
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Run all attacks in sequence
    all_attacks = ['fgsm', 'pgd', 'cw', 'sticker']
    summaries = []

    for attack_name in all_attacks:
        summary = run_attack(
            attack_name, model, device, args,
            original_image, source_class, target_class_idx,
            original_preds, output_dir, image_name
        )
        summaries.append(summary)

    # Print combined summary table
    print(f"\n{'=' * 60}")
    print("OVERALL ATTACK SUMMARY")
    print(f"{'=' * 60}")
    print(f"Original: {original_preds[0]['class']} ({original_preds[0]['confidence']*100:.1f}%)\n")

    for s in summaries:
        if target_class_idx is not None and s['adv_index'] == target_class_idx:
            status = "✓ TARGETED SUCCESS"
        elif s['adv_index'] != source_class:
            status = "✓ CHANGED"
        else:
            status = "• unchanged"
        print(f"  {s['attack'].upper():<10} → {s['adv_class']:<30} "
              f"({s['adv_confidence']*100:.1f}%)  {status}")

    print(f"{'=' * 60}\n")

    # Restore stdout and close the log file
    sys.stdout = tee.terminal
    tee.close()
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()