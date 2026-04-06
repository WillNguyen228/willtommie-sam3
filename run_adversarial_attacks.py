"""
SAM3 Adversarial Attack Framework
Supports FGSM, PGD, and C&W attacks on SAM3 model
"""

import torch
import torch.nn.functional as F
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np


class AdversarialAttacker:
    """Base class for adversarial attacks on SAM3"""
    
    def __init__(self, model, processor, device="cuda", preserve_aspect_ratio=True):
        self.model = model
        self.processor = processor
        self.device = device
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.resolution = processor.resolution
        self.model.eval()
        
        # Use the processor's transform (which resizes to square)
        # We'll handle preserving aspect ratio by resizing back to original after attack
        self.custom_transform = processor.transform
        
    def preprocess_image(self, image):
        """Convert PIL image to tensor and apply transforms"""
        if isinstance(image, Image.Image):
            width, height = image.size
            original_size = (width, height)
        else:
            raise ValueError("Image must be a PIL image")
        
        # Convert to tensor
        image_tensor = v2.functional.to_image(image).to(self.device)
        image_tensor = self.custom_transform(image_tensor).unsqueeze(0)
        
        return image_tensor, original_size
    
    def compute_loss(self, image_tensor, target_prompt, original_boxes=None):
        """
        Compute loss for adversarial attack
        For untargeted attack: we want to MINIMIZE detection confidence
        So we return NEGATIVE loss (gradient descent will reduce detections)
        """
        # Forward pass through backbone
        backbone_out = self.model.backbone.forward_image(image_tensor)
        
        # Process text prompt
        text_outputs = self.model.backbone.forward_text([target_prompt], device=self.device)
        backbone_out.update(text_outputs)
        
        # Get dummy geometric prompt
        geometric_prompt = self.model._get_dummy_prompt()
        
        # Forward through grounding head
        outputs = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=self.processor.find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )
        
        # Get predictions
        out_logits = outputs["pred_logits"]
        presence_logit = outputs["presence_logit_dec"]
        
        # Loss: we want to MINIMIZE detection confidence (untargeted attack)
        # Return NEGATIVE loss so that gradient descent reduces detections
        detection_loss = out_logits.sigmoid().max()
        presence_loss = presence_logit.sigmoid()
        
        # Combined loss - negative because we want to minimize detections
        total_loss = -(detection_loss + presence_loss)
        
        return total_loss, outputs
    
    def apply_perturbation(self, original_image, perturbation, epsilon):
        """Apply perturbation and clip to valid range"""
        perturbed = original_image + perturbation
        # Clip to valid range after normalization [-1, 1] for normalized images
        # The transform normalizes to mean=0.5, std=0.5, so range is roughly [-1, 1]
        perturbed = torch.clamp(perturbed, -1, 1)
        return perturbed
    
    def tensor_to_pil(self, tensor, original_size=None):
        """Convert tensor back to PIL image and optionally resize to original size"""
        # Denormalize: reverse the normalization (mean=0.5, std=0.5)
        tensor = tensor.squeeze(0)
        tensor = tensor * 0.5 + 0.5  # Denormalize
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL
        tensor = tensor.cpu()
        image_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        # Resize back to original size if specified
        if original_size is not None:
            pil_image = pil_image.resize(original_size, Image.Resampling.LANCZOS)
        
        return pil_image


class FGSM(AdversarialAttacker):
    """Fast Gradient Sign Method"""
    
    def __init__(self, model, processor, epsilon=0.03, device="cuda"):
        super().__init__(model, processor, device)
        self.epsilon = epsilon
        
    def attack(self, image, target_prompt):
        """
        Perform FGSM attack
        Args:
            image: PIL Image
            target_prompt: text prompt for detection
        Returns:
            adversarial_image: PIL Image with adversarial perturbation
            perturbation: the actual perturbation added
        """
        # Preprocess image
        image_tensor, original_size = self.preprocess_image(image)
        
        # Enable gradients
        image_tensor.requires_grad = True
        
        # Compute loss
        loss, original_outputs = self.compute_loss(image_tensor, target_prompt)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Get gradient
        gradient = image_tensor.grad.data
        
        # Create perturbation: epsilon * sign(gradient)
        # For untargeted attack, we want to go in direction of gradient to maximize loss
        perturbation = self.epsilon * gradient.sign()
        
        # Apply perturbation
        perturbed_tensor = self.apply_perturbation(image_tensor.detach(), perturbation, self.epsilon)
        
        # Convert back to PIL and resize to original dimensions
        adversarial_image = self.tensor_to_pil(perturbed_tensor, original_size)
        
        return adversarial_image, perturbation, loss.item()


class PGD(AdversarialAttacker):
    """Projected Gradient Descent (iterative FGSM)"""
    
    def __init__(self, model, processor, epsilon=0.03, alpha=0.01, iterations=10, device="cuda"):
        super().__init__(model, processor, device)
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        
    def attack(self, image, target_prompt):
        """
        Perform PGD attack
        Args:
            image: PIL Image
            target_prompt: text prompt for detection
        Returns:
            adversarial_image: PIL Image with adversarial perturbation
            perturbation: the actual perturbation added
        """
        # Preprocess image
        original_tensor, original_size = self.preprocess_image(image)
        perturbed_tensor = original_tensor.clone().detach()
        
        print(f"  Running PGD attack with {self.iterations} iterations...")
        
        for i in range(self.iterations):
            perturbed_tensor.requires_grad = True
            
            # Compute loss
            loss, _ = self.compute_loss(perturbed_tensor, target_prompt)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Get gradient
            gradient = perturbed_tensor.grad.data
            
            # Update perturbation
            with torch.no_grad():
                # Take a step in the direction of gradient
                perturbed_tensor = perturbed_tensor.detach() + self.alpha * gradient.sign()
                
                # Project back to epsilon ball around original image
                perturbation = torch.clamp(
                    perturbed_tensor - original_tensor, 
                    -self.epsilon, 
                    self.epsilon
                )
                perturbed_tensor = torch.clamp(
                    original_tensor + perturbation, 
                    -1, 1
                )
            
            if (i + 1) % 2 == 0:
                print(f"    Iteration {i+1}/{self.iterations}, Loss: {loss.item():.4f}")
        
        # Convert back to PIL and resize to original dimensions
        adversarial_image = self.tensor_to_pil(perturbed_tensor, original_size)
        final_perturbation = perturbed_tensor - original_tensor
        
        return adversarial_image, final_perturbation, loss.item()


class CW(AdversarialAttacker):
    """Carlini & Wagner L2 Attack"""
    
    def __init__(self, model, processor, c=1.0, kappa=0, iterations=100, learning_rate=0.01, device="cuda"):
        super().__init__(model, processor, device)
        self.c = c
        self.kappa = kappa
        self.iterations = iterations
        self.learning_rate = learning_rate
        
    def attack(self, image, target_prompt):
        """
        Perform C&W attack
        Args:
            image: PIL Image
            target_prompt: text prompt for detection
        Returns:
            adversarial_image: PIL Image with adversarial perturbation
            perturbation: the actual perturbation added
        """
        # Preprocess image
        original_tensor, original_size = self.preprocess_image(image)
        
        # Initialize perturbation variable in tanh space for unconstrained optimization
        w = torch.zeros_like(original_tensor, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([w], lr=self.learning_rate)
        
        print(f"  Running C&W attack with {self.iterations} iterations...")
        
        best_perturbation = None
        best_loss = float('inf')
        
        for i in range(self.iterations):
            # Convert from tanh space to image space
            # This ensures the perturbed image stays in valid range
            perturbed_tensor = 0.5 * (torch.tanh(w) + 1)
            # Normalize to SAM3's expected range
            perturbed_tensor = (perturbed_tensor - 0.5) / 0.5
            
            # Compute loss
            detection_loss, _ = self.compute_loss(perturbed_tensor, target_prompt)
            
            # L2 distance penalty
            l2_dist = torch.norm(perturbed_tensor - original_tensor)
            
            # Combined loss: C&W formulation
            # We want to minimize detection while keeping perturbation small
            total_loss = l2_dist + self.c * detection_loss
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track best perturbation
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_perturbation = perturbed_tensor.detach().clone()
            
            if (i + 1) % 20 == 0:
                print(f"    Iteration {i+1}/{self.iterations}, Loss: {total_loss.item():.4f}, "
                      f"L2: {l2_dist.item():.4f}, Detection: {detection_loss.item():.4f}")
        
        # Use best perturbation found and resize to original dimensions
        if best_perturbation is not None:
            adversarial_image = self.tensor_to_pil(best_perturbation, original_size)
            final_perturbation = best_perturbation - original_tensor
        else:
            adversarial_image = self.tensor_to_pil(perturbed_tensor, original_size)
            final_perturbation = perturbed_tensor - original_tensor
        
        return adversarial_image, final_perturbation, best_loss


def test_adversarial_attack(attack_method, image_path, animal_name, output_dir):
    """Test adversarial attack on a single image"""
    
    # Create subfolder for this animal
    animal_output_dir = os.path.join(output_dir, animal_name)
    os.makedirs(animal_output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Processing: {animal_name} ({image_path})")
    print(f"Attack: {attack_method}")
    print(f"Output folder: {animal_output_dir}")
    print('='*70)
    
    # Load model
    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=0.1)
    
    # Load image
    try:
        original_image = Image.open(image_path).convert('RGB')
        print(f"Image size: {original_image.size[0]}x{original_image.size[1]}")
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return
    
    # Test original image first
    print("\n--- Testing Original Image ---")
    inference_state = processor.set_image(original_image)
    processor.reset_all_prompts(inference_state)
    output = processor.set_text_prompt(state=inference_state, prompt=animal_name)
    
    boxes_orig = output["boxes"]
    scores_orig = output["scores"]
    
    if boxes_orig is not None and len(scores_orig) > 0:
        print(f"Original: Detected {len(boxes_orig)} {animal_name}(s)")
        for idx, (box, score) in enumerate(zip(boxes_orig, scores_orig)):
            print(f"  {animal_name} #{idx+1}: score={score.item():.4f}")
    else:  
        print(f"Original: No {animal_name} detected.")
    
    # Plot and save original
    plot_results(original_image, inference_state)
    orig_output_path = f"{animal_output_dir}/{animal_name}_original.png"
    plt.savefig(orig_output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    # Create attacker
    print(f"\n--- Applying {attack_method.upper()} Attack ---")
    
    if attack_method.lower() == 'fgsm':
        attacker = FGSM(model, processor, epsilon=0.03)
    elif attack_method.lower() in ['pgd', 'pgm']:
        attacker = PGD(model, processor, epsilon=0.03, alpha=0.01, iterations=10)
    elif attack_method.lower() == 'cw':
        attacker = CW(model, processor, c=1.0, iterations=100, learning_rate=0.01)
    else:
        print(f"Unknown attack method: {attack_method}")
        return
    
    # Perform attack
    adversarial_image, perturbation, loss_value = attacker.attack(original_image, animal_name)
    print(f"Attack completed. Loss: {loss_value:.4f}")
    
    # Save adversarial image
    adv_image_path = f"{animal_output_dir}/{animal_name}_adversarial_{attack_method}.png"
    adversarial_image.save(adv_image_path)
    print(f"Saved adversarial image to: {adv_image_path}")
    
    # Test adversarial image
    print("\n--- Testing Adversarial Image ---")
    inference_state_adv = processor.set_image(adversarial_image)
    processor.reset_all_prompts(inference_state_adv)
    output_adv = processor.set_text_prompt(state=inference_state_adv, prompt=animal_name)
    
    boxes_adv = output_adv["boxes"]
    scores_adv = output_adv["scores"]
    
    if boxes_adv is not None and len(scores_adv) > 0:
        print(f"Adversarial: Detected {len(boxes_adv)} {animal_name}(s)")
        for idx, (box, score) in enumerate(zip(boxes_adv, scores_adv)):
            print(f"  {animal_name} #{idx+1}: score={score.item():.4f}")
    else:
        print(f"Adversarial: No {animal_name} detected.")
    
    # Plot and save adversarial result
    plot_results(adversarial_image, inference_state_adv)
    adv_output_path = f"{animal_output_dir}/{animal_name}_adversarial_{attack_method}_result.png"
    plt.savefig(adv_output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    # Visualize perturbation
    perturbation_np = perturbation.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    perturbation_vis = np.abs(perturbation_np)
    perturbation_vis = (perturbation_vis - perturbation_vis.min()) / (perturbation_vis.max() - perturbation_vis.min() + 1e-8)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(perturbation_vis)
    plt.title(f'Perturbation ({attack_method.upper()})')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(adversarial_image)
    plt.title('Adversarial')
    plt.axis('off')
    
    comparison_path = f"{animal_output_dir}/{animal_name}_{attack_method}_comparison.png"
    plt.savefig(comparison_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved comparison to: {comparison_path}")
    
    # Print summary
    print("\n--- Attack Summary ---")
    print(f"Original detections: {len(scores_orig) if scores_orig is not None else 0}")
    print(f"Adversarial detections: {len(scores_adv) if scores_adv is not None else 0}")
    if scores_orig is not None and len(scores_orig) > 0:
        print(f"Original max confidence: {scores_orig.max().item():.4f}")
    if scores_adv is not None and len(scores_adv) > 0:
        print(f"Adversarial max confidence: {scores_adv.max().item():.4f}")
    
    attack_success = (scores_orig is not None and len(scores_orig) > 0 and 
                     (scores_adv is None or len(scores_adv) == 0 or 
                      scores_adv.max().item() < scores_orig.max().item() * 0.5))
    print(f"Attack successful: {attack_success}")


def main():
    parser = argparse.ArgumentParser(description='SAM3 Adversarial Attack Framework')
    parser.add_argument('--attack', type=str, default='fgsm', 
                       choices=['fgsm', 'pgd', 'pgm', 'cw'],
                       help='Attack method: fgsm, pgd/pgm, or cw')
    parser.add_argument('--image', type=str, default='data/cat.jpg',
                       help='Path to input image')
    parser.add_argument('--prompt', type=str, default='cat',
                       help='Text prompt for detection')
    parser.add_argument('--output-dir', type=str, default='adversarial_results',
                       help='Output directory for results')
    parser.add_argument('--epsilon', type=float, default=0.03,
                       help='Perturbation budget for FGSM/PGD')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations for PGD/CW')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run attack
    test_adversarial_attack(args.attack, args.image, args.prompt, args.output_dir)
    
    print(f"\n{'='*70}")
    print(f"All results saved in '{args.output_dir}/{args.prompt}/' folder")
    print('='*70)


if __name__ == "__main__":
    main()
