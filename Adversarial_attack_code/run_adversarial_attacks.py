"""
SAM3 Adversarial Attack Framework
Supports FGSM, PGD, C&W, Adversarial Sticker, Score-Based, and Decision-Based attacks on SAM3 model
"""

# Add parent directory to path so we can import sam3 module
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import torch.fft as fft
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import gc


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
    
    def compute_targeted_loss(self, image_tensor, source_prompt, target_prompt):
        """
        Compute loss for TARGETED adversarial attack
        Goal: Minimize detection of source_prompt AND maximize detection of target_prompt
        
        Args:
            image_tensor: Input image tensor
            source_prompt: Original class to hide (e.g., "cat")
            target_prompt: Target class to fake (e.g., "dog")
        
        Returns:
            total_loss: Combined loss that encourages target class detection
            source_outputs: Outputs for source prompt
            target_outputs: Outputs for target prompt
        """
        # Get predictions for SOURCE class (we want to minimize this)
        backbone_out_source = self.model.backbone.forward_image(image_tensor)
        text_outputs_source = self.model.backbone.forward_text([source_prompt], device=self.device)
        backbone_out_source.update(text_outputs_source)
        geometric_prompt = self.model._get_dummy_prompt()
        
        source_outputs = self.model.forward_grounding(
            backbone_out=backbone_out_source,
            find_input=self.processor.find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )
        
        # Get predictions for TARGET class (we want to maximize this)
        backbone_out_target = self.model.backbone.forward_image(image_tensor)
        text_outputs_target = self.model.backbone.forward_text([target_prompt], device=self.device)
        backbone_out_target.update(text_outputs_target)
        
        target_outputs = self.model.forward_grounding(
            backbone_out=backbone_out_target,
            find_input=self.processor.find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )
        
        # Source class: minimize ALL detection confidence
        source_logits = source_outputs["pred_logits"].sigmoid()
        source_presence = source_outputs["presence_logit_dec"].sigmoid()
        # Use max to focus on suppressing the strongest detection
        source_confidence = source_logits.max() + source_presence
        
        # Target class: maximize STRONGEST detections, but penalize too many
        target_logits = target_outputs["pred_logits"].sigmoid()
        target_presence = target_outputs["presence_logit_dec"].sigmoid()
        
        # Focus on top-k strongest detections (e.g., top 5)
        k = min(5, target_logits.numel())
        top_k_logits = torch.topk(target_logits.flatten(), k).values
        
        # Maximize the strongest detections
        target_confidence = top_k_logits.mean() + target_presence
        
        # Add sparsity penalty: penalize having too many medium-confidence detections
        # This encourages fewer, higher-confidence detections
        threshold = 0.5
        num_above_threshold = (target_logits > threshold).sum().float()
        sparsity_loss = torch.relu(num_above_threshold - 3.0) * 0.1  # Penalty if more than 3 detections
        
        # Combined loss: minimize source, maximize target, encourage sparsity
        total_loss = source_confidence - target_confidence + sparsity_loss
        
        return total_loss, source_outputs, target_outputs
    
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
        
        # Clear gradients to free memory
        image_tensor.grad = None
        torch.cuda.empty_cache()
        
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
            
            # Clear gradients to free memory
            perturbed_tensor.grad = None
            torch.cuda.empty_cache()
            
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
    """Carlini & Wagner L2 Attack with Perceptual Improvements"""
    
    def __init__(self, model, processor, c=10.0, kappa=0, iterations=100, learning_rate=0.05, 
                 target_class=None, perceptual_weight=0.1, frequency_weight=0.05, device="cuda"):
        super().__init__(model, processor, device)
        self.c = c  # Increased from 1.0 to 10.0 to prioritize attack success over perturbation size
        self.kappa = kappa
        self.iterations = iterations
        self.learning_rate = learning_rate  # Increased from 0.01 to 0.05 for faster convergence
        self.target_class = target_class  # If None, untargeted attack
        self.perceptual_weight = perceptual_weight  # Weight for perceptual loss
        self.frequency_weight = frequency_weight  # Weight for high-frequency penalty
    
    def compute_perceptual_loss(self, original, perturbed):
        """
        Compute perceptual loss to penalize visually noticeable differences.
        Uses color space and local contrast metrics.
        """
        # L2 loss in LAB color space (more perceptually uniform than RGB)
        # Approximate LAB conversion using weighted channels
        # Convert from normalized space back to [0, 1]
        orig_01 = (original * 0.5 + 0.5).clamp(0, 1)
        pert_01 = (perturbed * 0.5 + 0.5).clamp(0, 1)
        
        # Simple perceptual weighting: human eye is more sensitive to changes in brightness (Y channel)
        # than in chrominance (U, V channels)
        orig_y = 0.299 * orig_01[:, 0] + 0.587 * orig_01[:, 1] + 0.114 * orig_01[:, 2]
        pert_y = 0.299 * pert_01[:, 0] + 0.587 * pert_01[:, 1] + 0.114 * pert_01[:, 2]
        
        # Luminance loss (most perceptible)
        luminance_loss = F.mse_loss(orig_y, pert_y)
        
        # Color loss (less perceptible, so lower weight)
        color_loss = F.mse_loss(orig_01, pert_01)
        
        return 2.0 * luminance_loss + 0.5 * color_loss
    
    def compute_frequency_loss(self, perturbation):
        """
        Penalize high-frequency components in perturbation.
        High-frequency noise is more visible to humans.
        """
        # Apply 2D FFT to perturbation
        fft_pert = fft.fft2(perturbation)
        fft_magnitude = torch.abs(fft_pert)
        
        # Create high-frequency mask (emphasize outer frequencies)
        h, w = fft_magnitude.shape[-2], fft_magnitude.shape[-1]
        center_h, center_w = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h, device=perturbation.device), 
                              torch.arange(w, device=perturbation.device), indexing='ij')
        
        # Distance from center (low frequencies)
        dist = torch.sqrt(((y - center_h) ** 2 + (x - center_w) ** 2).float())
        high_freq_mask = (dist > min(h, w) * 0.3).float()  # Outer 70% are high frequencies
        
        # Penalize high-frequency energy
        high_freq_energy = (fft_magnitude * high_freq_mask).mean()
        
        return high_freq_energy
        
    def attack(self, image, target_prompt):
        """
        Perform C&W attack
        Args:
            image: PIL Image
            target_prompt: text prompt for detection (source class for targeted attack)
        Returns:
            adversarial_image: PIL Image with adversarial perturbation
            perturbation: the actual perturbation added
        """
        # Preprocess image
        original_tensor, original_size = self.preprocess_image(image)
        
        # Initialize perturbation variable in tanh space for unconstrained optimization
        w = torch.zeros_like(original_tensor, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([w], lr=self.learning_rate)
        
        if self.target_class:
            print(f"  Running TARGETED C&W attack with {self.iterations} iterations...")
            print(f"  Source class: '{target_prompt}' -> Target class: '{self.target_class}'")
        else:
            print(f"  Running C&W attack with {self.iterations} iterations...")
        
        best_perturbation = None
        best_loss = float('inf')
        
        for i in range(self.iterations):
            # Convert from tanh space to image space
            # This ensures the perturbed image stays in valid range
            perturbed_tensor = 0.5 * (torch.tanh(w) + 1)
            # Normalize to SAM3's expected range
            perturbed_tensor = (perturbed_tensor - 0.5) / 0.5
            
            # Compute loss based on attack type
            if self.target_class:
                # TARGETED attack: fool model to detect target_class instead of target_prompt
                detection_loss, source_outputs, target_outputs = self.compute_targeted_loss(
                    perturbed_tensor, target_prompt, self.target_class
                )
            else:
                # UNTARGETED attack: minimize detection of target_prompt
                detection_loss, _ = self.compute_loss(perturbed_tensor, target_prompt)
            
            # L2 distance penalty
            l2_dist = torch.norm(perturbed_tensor - original_tensor)
            
            # Perceptual loss to reduce visible artifacts
            perceptual_loss = self.compute_perceptual_loss(original_tensor, perturbed_tensor)
            
            # High-frequency penalty to make perturbations smoother
            perturbation = perturbed_tensor - original_tensor
            frequency_loss = self.compute_frequency_loss(perturbation)
            
            # Combined loss: C&W formulation with perceptual improvements
            total_loss = (l2_dist + 
                         self.c * detection_loss + 
                         self.perceptual_weight * perceptual_loss + 
                         self.frequency_weight * frequency_loss)
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Periodic memory cleanup
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
            
            # Track best perturbation
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_perturbation = perturbed_tensor.detach().clone()
            
            if (i + 1) % 20 == 0:
                if self.target_class:
                    # For targeted attacks, also show source and target confidences
                    with torch.no_grad():
                        _, src_out, tgt_out = self.compute_targeted_loss(
                            perturbed_tensor, target_prompt, self.target_class
                        )
                        src_conf = src_out["pred_logits"].sigmoid().max().item()
                        tgt_conf = tgt_out["pred_logits"].sigmoid().max().item()
                        # Count high-confidence target detections
                        num_high_conf = (tgt_out["pred_logits"].sigmoid() > 0.5).sum().item()
                        print(f"    Iteration {i+1}/{self.iterations}, Loss: {total_loss.item():.4f}, "
                              f"L2: {l2_dist.item():.4f}, SrcMax: {src_conf:.4f}, TgtMax: {tgt_conf:.4f}, #Tgt>0.5: {num_high_conf}")
                else:
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


class AdversarialSticker(AdversarialAttacker):
    """Adversarial Patch/Sticker Attack - Creates a small localized circular perturbation"""
    
    def __init__(self, model, processor, patch_size=100, location='center', 
                 c=10.0, iterations=200, learning_rate=0.1, target_class=None, device="cuda"):
        super().__init__(model, processor, device)
        self.patch_size = patch_size  # Diameter of circular sticker
        self.location = location  # 'center', 'random', 'top-left', etc.
        self.c = c
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.target_class = target_class
    
    def create_circular_mask(self, size, device):
        """Create a circular mask for the sticker"""
        # Create coordinate grid
        y, x = torch.meshgrid(torch.arange(size, device=device), 
                             torch.arange(size, device=device), indexing='ij')
        
        # Center of the patch
        center = size / 2.0
        
        # Distance from center
        dist = torch.sqrt((x - center + 0.5) ** 2 + (y - center + 0.5) ** 2)
        
        # Create circular mask (1 inside circle, 0 outside)
        radius = size / 2.0
        mask = (dist <= radius).float()
        
        # Add smooth edges with anti-aliasing
        edge_width = 2.0
        mask = torch.clamp((radius - dist) / edge_width + 0.5, 0, 1)
        
        return mask
    
    def get_patch_location(self, image_shape, patch_size):
        """Determine where to place the patch"""
        _, _, h, w = image_shape
        
        if self.location == 'center':
            x = (w - patch_size) // 2
            y = (h - patch_size) // 2
        elif self.location == 'top-left':
            x = 0
            y = 0
        elif self.location == 'top-right':
            x = w - patch_size
            y = 0
        elif self.location == 'bottom-left':
            x = 0
            y = h - patch_size
        elif self.location == 'bottom-right':
            x = w - patch_size
            y = h - patch_size
        elif self.location == 'random':
            x = np.random.randint(0, max(1, w - patch_size))
            y = np.random.randint(0, max(1, h - patch_size))
        else:
            # Default to center
            x = (w - patch_size) // 2
            y = (h - patch_size) // 2
        
        # Ensure within bounds
        x = max(0, min(x, w - patch_size))
        y = max(0, min(y, h - patch_size))
        
        return x, y
    
    def apply_patch(self, image, patch, x, y, mask):
        """Apply circular patch to image at location (x, y) using mask"""
        _, _, h, w = image.shape
        patch_h, patch_w = patch.shape[-2], patch.shape[-1]
        
        # Ensure patch fits
        patch_h = min(patch_h, h - y)
        patch_w = min(patch_w, w - x)
        
        # Trim mask if needed
        trimmed_mask = mask[:patch_h, :patch_w]
        
        # Create modified image with circular blending
        patched_image = image.clone()
        
        # Extract the region
        original_region = patched_image[:, :, y:y+patch_h, x:x+patch_w]
        patch_region = patch[:, :, :patch_h, :patch_w]
        
        # Apply mask: blend patch with original using circular mask
        # mask is shape (H, W), need to broadcast for channels
        mask_3d = trimmed_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        blended = patch_region * mask_3d + original_region * (1 - mask_3d)
        
        patched_image[:, :, y:y+patch_h, x:x+patch_w] = blended
        
        return patched_image
    
    def attack(self, image, target_prompt):
        """
        Perform adversarial sticker attack
        Args:
            image: PIL Image
            target_prompt: text prompt for detection (source class for targeted attack)
        Returns:
            adversarial_image: PIL Image with adversarial sticker
            patch: the sticker/patch tensor
            location: (x, y) coordinates of patch
        """
        # Preprocess image (transforms to square for SAM3)
        original_tensor, original_size = self.preprocess_image(image)
        
        # Determine patch size (scale with SAM3's resolution)
        _, _, img_h, img_w = original_tensor.shape
        patch_size = min(self.patch_size, img_h // 4, img_w // 4)  # Max 1/4 of image
        
        # Create circular mask for the patch
        circular_mask = self.create_circular_mask(patch_size, self.device)
        print(f"  Created circular sticker with diameter: {patch_size} pixels")
        print(f"  Working in SAM3 space: {img_w}x{img_h}, Original: {original_size[0]}x{original_size[1]}")
        
        # Get patch location in SAM3 space
        patch_x, patch_y = self.get_patch_location(original_tensor.shape, patch_size)
        print(f"  Sticker location: ({patch_x}, {patch_y})")
        
        # Initialize patch in tanh space for unconstrained optimization
        patch_shape = (1, 3, patch_size, patch_size)
        w = torch.zeros(patch_shape, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([w], lr=self.learning_rate)
        
        if self.target_class:
            print(f"  Running TARGETED Adversarial Sticker attack with {self.iterations} iterations...")
            print(f"  Source class: '{target_prompt}' -> Target class: '{self.target_class}'")
        else:
            print(f"  Running Adversarial Sticker attack with {self.iterations} iterations...")
        
        best_patch = None
        best_loss = float('inf')
        
        for i in range(self.iterations):
            # Convert patch from tanh space to valid pixel range
            patch = 0.5 * (torch.tanh(w) + 1)
            patch = (patch - 0.5) / 0.5  # Normalize to SAM3's expected range
            
            # Apply circular mask to patch (only optimize circular region)
            mask_3d = circular_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            masked_patch = patch * mask_3d
            
            # Apply patch to SAM3-transformed image
            patched_image = self.apply_patch(original_tensor, masked_patch, patch_x, patch_y, circular_mask)
            
            # Compute detection loss
            if self.target_class:
                detection_loss, _, _ = self.compute_targeted_loss(
                    patched_image, target_prompt, self.target_class
                )
            else:
                detection_loss, _ = self.compute_loss(patched_image, target_prompt)
            
            # Patch regularization (encourage small, smooth patches)
            # Only regularize within the circular region
            patch_norm = torch.norm(masked_patch)
            
            # Total variation loss for smoothness
            tv_loss = (torch.sum(torch.abs(patch[:, :, :, :-1] - patch[:, :, :, 1:])) +
                      torch.sum(torch.abs(patch[:, :, :-1, :] - patch[:, :, 1:, :])))
            
            # Combined loss
            total_loss = self.c * detection_loss + 0.01 * patch_norm + 0.001 * tv_loss
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Periodic memory cleanup
            if (i + 1) % 20 == 0:
                torch.cuda.empty_cache()
            
            # Track best patch
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_patch = masked_patch.detach().clone()
            
            if (i + 1) % 40 == 0:
                if self.target_class:
                    with torch.no_grad():
                        _, src_out, tgt_out = self.compute_targeted_loss(
                            patched_image, target_prompt, self.target_class
                        )
                        src_conf = src_out["pred_logits"].sigmoid().max().item()
                        tgt_conf = tgt_out["pred_logits"].sigmoid().max().item()
                        print(f"    Iteration {i+1}/{self.iterations}, Loss: {total_loss.item():.4f}, "
                              f"SrcMax: {src_conf:.4f}, TgtMax: {tgt_conf:.4f}")
                else:
                    print(f"    Iteration {i+1}/{self.iterations}, Loss: {total_loss.item():.4f}, "
                          f"Detection: {detection_loss.item():.4f}")
        
        # Use best patch found
        if best_patch is not None:
            final_patched = self.apply_patch(original_tensor, best_patch, patch_x, patch_y, circular_mask)
        else:
            final_patched = patched_image
        
        # For the adversarial result, save it at SAM3's resolution first
        # This ensures the sticker stays in the exact right location when tested
        adversarial_image_sam3 = self.tensor_to_pil(final_patched, None)
        
        # Also create a version at original resolution for visualization
        adversarial_image_display = self.tensor_to_pil(final_patched, original_size)
        
        # Create visualization of the circular sticker with transparency
        # Convert patch to [0, 1] range
        patch_to_save = best_patch if best_patch is not None else masked_patch
        patch_01 = (patch_to_save.squeeze(0) * 0.5 + 0.5).clamp(0, 1).cpu()
        
        # Create RGBA image with alpha channel based on circular mask
        patch_rgba = torch.zeros(4, patch_size, patch_size)
        patch_rgba[:3, :, :] = patch_01
        patch_rgba[3, :, :] = circular_mask.cpu()  # Alpha channel
        
        # Convert to PIL Image with transparency
        patch_rgba_np = (patch_rgba.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        patch_pil = Image.fromarray(patch_rgba_np, mode='RGBA')
        
        # Return both SAM3-resolution version (for testing) and display version (for saving)
        return adversarial_image_sam3, adversarial_image_display, patch_pil, (patch_x, patch_y), best_loss


class ScoreBased(AdversarialAttacker):
    """Score-Based Attack using Natural Evolution Strategies (NES) for gradient estimation
    
    This is a black-box attack that only requires access to model output scores (logits/probabilities),
    not gradients. It estimates gradients by sampling random perturbations and using the model's
    output scores to compute a weighted combination that approximates the true gradient direction.
    """
    
    def __init__(self, model, processor, epsilon=0.1, iterations=50, num_samples=20, 
                 sigma=0.001, learning_rate=0.01, target_class=None, device="cuda"):
        super().__init__(model, processor, device)
        self.epsilon = epsilon  # Maximum perturbation magnitude
        self.iterations = iterations  # Number of optimization steps
        self.num_samples = num_samples  # Number of random samples per gradient estimation
        self.sigma = sigma  # Sampling standard deviation for perturbations
        self.learning_rate = learning_rate  # Step size for updates
        self.target_class = target_class  # If None, untargeted attack
        
    def estimate_gradient_nes(self, image_tensor, target_prompt):
        """
        Estimate gradient using Natural Evolution Strategies (NES)
        
        NES samples random directions, evaluates the loss in those directions,
        and computes a weighted average to approximate the gradient.
        
        For UNTARGETED attacks: We want to DECREASE detection confidence.
        So we define attack_success = -detection_confidence (higher = better attack)
        
        Args:
            image_tensor: Current adversarial image tensor
            target_prompt: Text prompt for detection
            
        Returns:
            gradient_estimate: Approximation of the gradient
        """
        # Sample random perturbation directions from Gaussian distribution
        # Shape: (num_samples, channels, height, width)
        perturbations = torch.randn(
            self.num_samples, *image_tensor.shape[1:], 
            device=self.device
        )
        
        # Evaluate attack success for each perturbed sample
        # Higher success = better attack = lower detection
        attack_scores = []
        
        for i in range(self.num_samples):
            # Apply perturbation
            perturbed = image_tensor + self.sigma * perturbations[i:i+1]
            perturbed = torch.clamp(perturbed, -1, 1)
            
            # Compute attack success score (no gradients needed - score-based only)
            with torch.no_grad():
                if self.target_class:
                    # For targeted: minimize source, maximize target
                    _, src_out, tgt_out = self.compute_targeted_loss(
                        perturbed, target_prompt, self.target_class
                    )
                    src_conf = src_out["pred_logits"].sigmoid().max().item()
                    tgt_conf = tgt_out["pred_logits"].sigmoid().max().item()
                    # Attack success using log-loss formulation (like cross-entropy)
                    # Higher = better attack
                    score = -torch.log(torch.tensor(src_conf + 1e-10)).item() + torch.log(torch.tensor(tgt_conf + 1e-10)).item()
                else:
                    # For untargeted: minimize detection
                    # Use negative log-probability (like cross-entropy loss)
                    _, outputs = self.compute_loss(perturbed, target_prompt)
                    detection_conf = outputs["pred_logits"].sigmoid().max().item()
                    # Attack success = -log(detection_confidence)
                    # When detection_conf is high (0.95), score is low (-0.05)
                    # When detection_conf is low (0.1), score is high (-(-2.3) = 2.3)
                    # Higher score = better attack
                    score = -torch.log(torch.tensor(detection_conf + 1e-10)).item()
                
                attack_scores.append(score)
        
        # Convert to tensor
        scores_tensor = torch.tensor(attack_scores, device=self.device)
        
        # Debug: print score statistics
        if torch.rand(1).item() < 0.1:  # Print 10% of the time to avoid spam
            print(f"      [NES Debug] Score range: [{scores_tensor.min().item():.4f}, {scores_tensor.max().item():.4f}], "
                  f"mean: {scores_tensor.mean().item():.4f}, std: {scores_tensor.std().item():.6f}")
        
        # Standard NES with baseline subtraction
        # Directions with higher-than-average attack success get positive weight
        scores_mean = scores_tensor.mean()
        scores_centered = scores_tensor - scores_mean
        
        # Estimate gradient: directions with better attack scores (higher) contribute positively
        # Formula: ĝ ≈ (1/Nσ) Σ (score_k - score_mean) * u_k
        gradient_estimate = torch.zeros_like(image_tensor)
        
        for i in range(self.num_samples):
            gradient_estimate += scores_centered[i] * perturbations[i:i+1]
        
        # Scale by (1 / (N * sigma)) as per NES formula
        gradient_estimate = gradient_estimate / (self.num_samples * self.sigma)
        
        return gradient_estimate
    
    def attack(self, image, target_prompt):
        """
        Perform score-based adversarial attack using NES gradient estimation
        
        Args:
            image: PIL Image
            target_prompt: text prompt for detection (source class for targeted attack)
            
        Returns:
            adversarial_image: PIL Image with adversarial perturbation
            perturbation: the actual perturbation added
            final_loss: final loss value
        """
        # Preprocess image
        original_tensor, original_size = self.preprocess_image(image)
        perturbed_tensor = original_tensor.clone().detach()
        
        if self.target_class:
            print(f"  Running TARGETED Score-Based attack with {self.iterations} iterations...")
            print(f"  Source class: '{target_prompt}' -> Target class: '{self.target_class}'")
            print(f"  Using {self.num_samples} samples per gradient estimation")
        else:
            print(f"  Running Score-Based attack with {self.iterations} iterations...")
            print(f"  Using {self.num_samples} samples per gradient estimation")
        
        best_perturbation = None
        best_score = float('-inf')  # Track HIGHEST attack success score
        
        # Total queries counter
        total_queries = 0
        
        for i in range(self.iterations):
            # Estimate gradient using NES (requires num_samples queries)
            gradient_estimate = self.estimate_gradient_nes(perturbed_tensor, target_prompt)
            total_queries += self.num_samples
            
            # Debug: check gradient magnitude
            grad_norm = torch.norm(gradient_estimate).item()
            grad_max = torch.abs(gradient_estimate).max().item()
            
            # Update perturbed image in the direction of estimated gradient
            with torch.no_grad():
                # Normalize gradient to unit norm for stable updates
                grad_norm = torch.norm(gradient_estimate)
                if grad_norm > 1e-8:
                    gradient_normalized = gradient_estimate / grad_norm
                else:
                    gradient_normalized = gradient_estimate
                
                # Take a step using normalized gradient (preserves direction, controlled magnitude)
                perturbed_tensor = perturbed_tensor + self.learning_rate * gradient_normalized
                
                # Project back to epsilon ball around original image
                perturbation = perturbed_tensor - original_tensor
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                
                # Apply perturbation and clip to valid range
                perturbed_tensor = torch.clamp(original_tensor + perturbation, -1, 1)
            
            # Evaluate current solution (1 additional query for monitoring)
            with torch.no_grad():
                if self.target_class:
                    _, src_out, tgt_out = self.compute_targeted_loss(
                        perturbed_tensor, target_prompt, self.target_class
                    )
                    src_conf = src_out["pred_logits"].sigmoid().max().item()
                    tgt_conf = tgt_out["pred_logits"].sigmoid().max().item()
                    current_score = -torch.log(torch.tensor(src_conf + 1e-10)).item() + torch.log(torch.tensor(tgt_conf + 1e-10)).item()
                else:
                    _, outputs = self.compute_loss(perturbed_tensor, target_prompt)
                    detection_conf = outputs["pred_logits"].sigmoid().max().item()
                    current_score = -torch.log(torch.tensor(detection_conf + 1e-10)).item()
                
                total_queries += 1
            
            # Periodic memory cleanup
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
            
            # Track best perturbation (HIGHEST attack score = lowest detection)
            if current_score > best_score:
                best_score = current_score
                best_perturbation = perturbed_tensor.clone()
            
            # Print progress
            if (i + 1) % 10 == 0 or i == 0:
                if self.target_class:
                    print(f"    Iteration {i+1}/{self.iterations}, Score: {current_score:.4f}, "
                          f"SrcMax: {src_conf:.4f}, TgtMax: {tgt_conf:.4f}, Queries: {total_queries}, "
                          f"GradNorm: {grad_norm:.6f}, GradMax: {grad_max:.6f}")
                else:
                    print(f"    Iteration {i+1}/{self.iterations}, Score: {current_score:.4f}, "
                          f"DetectionMax: {detection_conf:.4f}, Queries: {total_queries}, "
                          f"GradNorm: {grad_norm:.6f}, GradMax: {grad_max:.6f}")
        
        print(f"  Total queries: {total_queries}")
        
        # Use best perturbation found
        if best_perturbation is not None:
            final_perturbed = best_perturbation
        else:
            final_perturbed = perturbed_tensor
        
        # Convert back to PIL and resize to original dimensions
        adversarial_image = self.tensor_to_pil(final_perturbed, original_size)
        final_perturbation = final_perturbed - original_tensor
        
        return adversarial_image, final_perturbation, best_score


class DecisionBased(AdversarialAttacker):
    """Decision-Based Attack (Boundary Attack) using only hard-label predictions
    
    This is the most restrictive black-box attack that only requires access to the final
    prediction label (e.g., "detected" or "not detected"), not gradients or confidence scores.
    
    The Boundary Attack:
    1. Starts with an initial adversarial example (could be random noise or from transfer attack)
    2. Iteratively walks along the decision boundary towards the original image
    3. At each step, proposes a new candidate by:
       - Taking a random step roughly orthogonal to the line connecting current point to original
       - Checking if it remains adversarial (only using hard labels)
       - Checking if it's closer to the original image
    4. Accepts the step only if both conditions are met
    
    This approach requires many queries but works when only binary decisions are available.
    """
    
    def __init__(self, model, processor, iterations=1000, initial_num_evals=100,
                 max_num_evals=10000, step_adapt=0.9, spherical_step_size=0.01, 
                 source_step_size=0.01, target_class=None, device="cuda"):
        super().__init__(model, processor, device)
        self.iterations = iterations
        self.initial_num_evals = initial_num_evals  # Evals for initial adversarial
        self.max_num_evals = max_num_evals  # Max total queries
        self.step_adapt = step_adapt  # Step size adaptation rate
        self.spherical_step_size = spherical_step_size  # Initial orthogonal step size
        self.source_step_size = source_step_size  # Initial step size towards source
        self.target_class = target_class
        
    def is_adversarial(self, image_tensor, target_prompt):
        """
        Check if image is adversarial using ONLY hard labels (decision-based)
        
        Returns True if:
        - Untargeted: target_prompt is NOT detected (or very weakly detected)
        - Targeted: target_class IS detected AND target_prompt is NOT detected
        
        This function only uses binary decision outcomes, not confidence scores.
        """
        with torch.no_grad():
            # Get predictions for target_prompt (source class)
            backbone_out = self.model.backbone.forward_image(image_tensor)
            text_outputs = self.model.backbone.forward_text([target_prompt], device=self.device)
            backbone_out.update(text_outputs)
            geometric_prompt = self.model._get_dummy_prompt()
            
            outputs = self.model.forward_grounding(
                backbone_out=backbone_out,
                find_input=self.processor.find_stage,
                geometric_prompt=geometric_prompt,
                find_target=None,
            )
            
            # Hard decision: detected if max confidence > threshold
            source_logits = outputs["pred_logits"].sigmoid()
            source_detected = source_logits.max().item() > 0.3  # Binary decision
            
            if self.target_class:
                # For targeted attack, also check target class
                backbone_out_target = self.model.backbone.forward_image(image_tensor)
                text_outputs_target = self.model.backbone.forward_text([self.target_class], device=self.device)
                backbone_out_target.update(text_outputs_target)
                
                target_outputs = self.model.forward_grounding(
                    backbone_out=backbone_out_target,
                    find_input=self.processor.find_stage,
                    geometric_prompt=geometric_prompt,
                    find_target=None,
                )
                
                target_logits = target_outputs["pred_logits"].sigmoid()
                target_detected = target_logits.max().item() > 0.3  # Binary decision
                
                # Adversarial if source NOT detected AND target IS detected
                return (not source_detected) and target_detected
            else:
                # Untargeted: adversarial if source NOT detected
                return not source_detected
    
    def generate_initial_adversarial(self, original_tensor, target_prompt, original_size):
        """
        Generate an initial adversarial example to start the boundary attack.
        
        Strategy:
        1. Try random noise
        2. If that fails, try uniform noise at different scales
        3. As a last resort, use a heavily perturbed version of the original
        """
        print("  Searching for initial adversarial example...")
        num_attempts = 0
        
        # Strategy 1: Random uniform noise
        for attempt in range(self.initial_num_evals // 4):
            # Generate random noise image in normalized range [-1, 1]
            random_image = torch.rand_like(original_tensor) * 2 - 1
            num_attempts += 1
            
            if self.is_adversarial(random_image, target_prompt):
                print(f"  ✓ Found initial adversarial (random noise) after {num_attempts} attempts")
                return random_image, num_attempts
        
        # Strategy 2: Blend random noise with original at different ratios
        for blend_ratio in [0.2, 0.4, 0.6, 0.8, 1.0]:
            for attempt in range(self.initial_num_evals // 10):
                random_image = torch.rand_like(original_tensor) * 2 - 1
                blended = original_tensor * (1 - blend_ratio) + random_image * blend_ratio
                blended = torch.clamp(blended, -1, 1)
                num_attempts += 1
                
                if self.is_adversarial(blended, target_prompt):
                    print(f"  ✓ Found initial adversarial (blend ratio={blend_ratio:.1f}) after {num_attempts} attempts")
                    return blended, num_attempts
        
        # Strategy 3: Heavy uniform perturbation
        for magnitude in [0.3, 0.5, 0.8, 1.0]:
            perturbation = torch.rand_like(original_tensor) * 2 - 1
            perturbed = original_tensor + magnitude * perturbation
            perturbed = torch.clamp(perturbed, -1, 1)
            num_attempts += 1
            
            if self.is_adversarial(perturbed, target_prompt):
                print(f"  ✓ Found initial adversarial (magnitude={magnitude:.1f}) after {num_attempts} attempts")
                return perturbed, num_attempts
        
        print(f"  ✗ Failed to find initial adversarial after {num_attempts} attempts")
        print(f"  Using heavily perturbed original as fallback (may not be adversarial)")
        # Return heavily perturbed version even if not confirmed adversarial
        perturbation = torch.rand_like(original_tensor) * 2 - 1
        fallback = torch.clamp(original_tensor + perturbation, -1, 1)
        return fallback, num_attempts
    
    def attack(self, image, target_prompt):
        """
        Perform Boundary Attack (Decision-Based)
        
        Args:
            image: PIL Image
            target_prompt: text prompt for detection (source class to hide)
            
        Returns:
            adversarial_image: PIL Image with minimal adversarial perturbation
            perturbation: the actual perturbation added
            total_queries: total number of model queries used
        """
        # Preprocess image
        original_tensor, original_size = self.preprocess_image(image)
        
        if self.target_class:
            print(f"  Running TARGETED Decision-Based (Boundary) Attack...")
            print(f"  Source class: '{target_prompt}' -> Target class: '{self.target_class}'")
        else:
            print(f"  Running Decision-Based (Boundary) Attack...")
        
        print(f"  WARNING: This attack is query-intensive. Max queries: {self.max_num_evals}")
        
        # Step 1: Find initial adversarial example
        adversarial_tensor, init_queries = self.generate_initial_adversarial(
            original_tensor, target_prompt, original_size
        )
        total_queries = init_queries
        
        # Verify it's adversarial
        if not self.is_adversarial(adversarial_tensor, target_prompt):
            print("  WARNING: Initial example is not adversarial. Attack may fail.")
        
        # Track best adversarial example (closest to original)
        best_adversarial = adversarial_tensor.clone()
        best_distance = torch.norm(best_adversarial - original_tensor).item()
        
        # Adaptive step sizes
        spherical_step = self.spherical_step_size
        source_step = self.source_step_size
        
        print(f"  Initial L2 distance: {best_distance:.4f}")
        print(f"  Starting boundary walk with {self.iterations} iterations...")
        
        # Step 2: Iteratively walk along decision boundary
        for iteration in range(self.iterations):
            if total_queries >= self.max_num_evals:
                print(f"  Reached maximum query limit ({self.max_num_evals})")
                break
            
            # Unnormalized direction from adversarial to original
            direction_to_original = original_tensor - adversarial_tensor
            distance_to_original = torch.norm(direction_to_original)
            
            if distance_to_original < 1e-6:
                print(f"  Converged: adversarial example very close to original")
                break
            
            # Normalized direction
            direction_to_original_normalized = direction_to_original / (distance_to_original + 1e-10)
            
            # Step 2a: Take step towards original (reduce perturbation)
            # This is the "source step" in boundary attack terminology
            candidate_source = adversarial_tensor + source_step * distance_to_original * direction_to_original_normalized
            candidate_source = torch.clamp(candidate_source, -1, 1)
            total_queries += 1
            
            # Check if still adversarial and closer to original
            if self.is_adversarial(candidate_source, target_prompt):
                distance_candidate = torch.norm(candidate_source - original_tensor).item()
                if distance_candidate < best_distance:
                    adversarial_tensor = candidate_source
                    best_distance = distance_candidate
                    best_adversarial = adversarial_tensor.clone()
                    # Increase step size (successful step)
                    source_step = min(source_step / self.step_adapt, 0.1)
                else:
                    # Decrease step size (didn't improve distance)
                    source_step = source_step * self.step_adapt
            else:
                # Not adversarial anymore, decrease step size
                source_step = source_step * self.step_adapt
            
            # Step 2b: Take orthogonal (spherical) step to explore boundary
            # Sample random direction orthogonal to direction_to_original
            random_direction = torch.randn_like(original_tensor)
            
            # Project random direction to be orthogonal to direction_to_original
            # Using Gram-Schmidt: v_perp = v - (v·u)u where u is direction_to_original_normalized
            dot_product = (random_direction * direction_to_original_normalized).sum()
            orthogonal_direction = random_direction - dot_product * direction_to_original_normalized
            orthogonal_direction_normalized = orthogonal_direction / (torch.norm(orthogonal_direction) + 1e-10)
            
            # Take spherical step
            candidate_spherical = adversarial_tensor + spherical_step * distance_to_original * orthogonal_direction_normalized
            candidate_spherical = torch.clamp(candidate_spherical, -1, 1)
            total_queries += 1
            
            # Check if still adversarial
            if self.is_adversarial(candidate_spherical, target_prompt):
                distance_candidate = torch.norm(candidate_spherical - original_tensor).item()
                if distance_candidate < best_distance:
                    adversarial_tensor = candidate_spherical
                    best_distance = distance_candidate
                    best_adversarial = adversarial_tensor.clone()
                    # Increase step size
                    spherical_step = min(spherical_step / self.step_adapt, 0.1)
                else:
                    # Still adversarial but didn't improve, keep it anyway (explore boundary)
                    adversarial_tensor = candidate_spherical
                    spherical_step = spherical_step * self.step_adapt
            else:
                # Not adversarial, decrease step size
                spherical_step = spherical_step * self.step_adapt
            
            # Periodic memory cleanup and progress reporting
            if (iteration + 1) % 100 == 0 or iteration == 0:
                torch.cuda.empty_cache()
                print(f"    Iteration {iteration+1}/{self.iterations}, L2: {best_distance:.4f}, "
                      f"Queries: {total_queries}, Steps: [src={source_step:.6f}, sph={spherical_step:.6f}]")
        
        print(f"  Total queries: {total_queries}")
        print(f"  Final L2 distance: {best_distance:.4f}")
        
        # Verify final result is adversarial
        final_is_adversarial = self.is_adversarial(best_adversarial, target_prompt)
        print(f"  Final result is adversarial: {final_is_adversarial}")
        
        # Convert back to PIL and resize to original dimensions
        adversarial_image = self.tensor_to_pil(best_adversarial, original_size)
        final_perturbation = best_adversarial - original_tensor
        
        return adversarial_image, final_perturbation, total_queries


def test_adversarial_attack(attack_method, image_path, animal_name, output_dir, target_class=None, 
                           stealthy=False, patch_size=150, patch_location='center', max_size=None, 
                           epsilon=0.1, iterations=10):
    """Test adversarial attack on a single image"""
    
    # Create subfolder for this animal
    animal_output_dir = os.path.join(output_dir, animal_name)
    os.makedirs(animal_output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Processing: {animal_name} ({image_path})")
    print(f"Attack: {attack_method}")
    if target_class:
        print(f"Target Class: {target_class} (TARGETED ATTACK)")
    print(f"Output folder: {animal_output_dir}")
    print('='*70)
    
    # Load model
    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    # Use higher confidence threshold to filter out weak detections
    confidence_thresh = 0.5 if target_class else 0.1
    processor = Sam3Processor(model, confidence_threshold=confidence_thresh)
    print(f"Using confidence threshold: {confidence_thresh}")
    
    # Load image
    try:
        original_image = Image.open(image_path).convert('RGB')
        print(f"Image size: {original_image.size[0]}x{original_image.size[1]}")
        
        # Downsample very large images to prevent OOM during gradient computation (only if max_size is set)
        if max_size is not None:
            width, height = original_image.size
            if max(width, height) > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                print(f"Downsampling to {new_width}x{new_height} (--max-size={max_size})")
                original_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"Downsampled image size: {original_image.size[0]}x{original_image.size[1]}")
        
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
        attacker = FGSM(model, processor, epsilon=epsilon)
    elif attack_method.lower() in ['pgd', 'pgm']:
        # Alpha is typically epsilon/iterations for good convergence
        alpha = epsilon / iterations
        attacker = PGD(model, processor, epsilon=epsilon, alpha=alpha, iterations=iterations)
    elif attack_method.lower() == 'cw':
        # Use stronger parameters for targeted attacks
        c_param = 10.0 if target_class else 1.0
        lr_param = 0.05 if target_class else 0.01
        # Perceptual parameters: higher values = more imperceptible but potentially less effective
        if stealthy:
            perceptual_w = 0.5   # Higher weight for very imperceptible perturbations
            frequency_w = 0.3    # Strong smoothness constraint
        else:
            perceptual_w = 0.2   # Weight for perceptual loss
            frequency_w = 0.1    # Weight for frequency smoothness
        attacker = CW(model, processor, c=c_param, iterations=100, learning_rate=lr_param, 
                     target_class=target_class, perceptual_weight=perceptual_w, 
                     frequency_weight=frequency_w)
    elif attack_method.lower() == 'sticker':
        # Adversarial sticker/patch attack
        attacker = AdversarialSticker(model, processor, patch_size=patch_size, location=patch_location,
                                     c=10.0, iterations=200, learning_rate=0.1, 
                                     target_class=target_class)
    elif attack_method.lower() == 'scorebased':
        # Score-based attack using Natural Evolution Strategies (NES)
        # Query-efficient black-box attack that only needs model output scores
        num_samples = 100  # Increased for more accurate gradient estimates
        sigma = 0.2  # Larger sampling radius for stronger signal (was 0.1)
        lr_scorebased = 0.05  # Larger step size for more aggressive updates (was 0.01)
        print(f"  Score-based attack parameters: samples={num_samples}, sigma={sigma}, lr={lr_scorebased}")
        print(f"  Note: Score-based attacks are query-intensive. Using {num_samples} samples per iteration.")
        attacker = ScoreBased(model, processor, epsilon=epsilon, iterations=iterations,
                             num_samples=num_samples, sigma=sigma, learning_rate=lr_scorebased,
                             target_class=target_class)
    elif attack_method.lower() == 'decision':
        # Decision-based attack (Boundary Attack) using only hard-label predictions
        # Most restrictive black-box setting - only uses final prediction labels
        boundary_iterations = 1000  # More iterations needed for boundary walking
        max_queries = 10000  # Maximum queries allowed
        print(f"  Decision-based attack parameters: iterations={boundary_iterations}, max_queries={max_queries}")
        print(f"  Note: Decision-based attacks require MANY queries (often thousands).")
        print(f"        This is the most restrictive attack - only uses hard labels (detected/not detected).")
        attacker = DecisionBased(model, processor, iterations=boundary_iterations,
                                initial_num_evals=100, max_num_evals=max_queries,
                                step_adapt=0.9, spherical_step_size=0.01, source_step_size=0.01,
                                target_class=target_class)
    else:
        print(f"Unknown attack method: {attack_method}")
        return
    
    # Perform attack (different return values for sticker vs others)
    if attack_method.lower() == 'sticker':
        adversarial_image_test, adversarial_image_display, patch_image, patch_location, loss_value = attacker.attack(original_image, animal_name)
        print(f"Attack completed. Loss: {loss_value:.4f}")
        print(f"Patch location: {patch_location}")
        
        # Save patch separately with appropriate naming
        if target_class:
            patch_filename = f"sticker_{target_class}.png"
        else:
            patch_filename = f"sticker_{animal_name}.png"
        
        patch_path = f"{animal_output_dir}/{patch_filename}"
        patch_image.save(patch_path)
        print(f"Saved sticker to: {patch_path}")
        
        # Use test version for evaluation, display version for saving
        adversarial_image = adversarial_image_test
        adversarial_image_tosave = adversarial_image_display
        perturbation = None  # No full perturbation for sticker
    else:
        adversarial_image, perturbation, loss_value = attacker.attack(original_image, animal_name)
        adversarial_image_tosave = adversarial_image
        print(f"Attack completed. Loss: {loss_value:.4f}")
    
    # Save adversarial image (use display version for sticker)
    adv_image_path = f"{animal_output_dir}/{animal_name}_adversarial_{attack_method}.png"
    adversarial_image_tosave.save(adv_image_path)
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
    
    # If targeted attack, also test for target class detection
    if target_class:
        print(f"\n--- Testing for Target Class '{target_class}' ---")
        processor.reset_all_prompts(inference_state_adv)
        output_target = processor.set_text_prompt(state=inference_state_adv, prompt=target_class)
        
        boxes_target = output_target["boxes"]
        scores_target = output_target["scores"]
        
        if boxes_target is not None and len(scores_target) > 0:
            print(f"Adversarial: Detected {len(boxes_target)} {target_class}(s)")
            for idx, (box, score) in enumerate(zip(boxes_target, scores_target)):
                print(f"  {target_class} #{idx+1}: score={score.item():.4f}")
        else:
            print(f"Adversarial: No {target_class} detected.")
        
        # Plot and save target class result
        plot_results(adversarial_image, inference_state_adv)
        target_output_path = f"{animal_output_dir}/{animal_name}_adversarial_{attack_method}_target_{target_class}.png"
        plt.savefig(target_output_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved target class detection to: {target_output_path}")
    
    # Visualize perturbation (skip for sticker attacks as we saved the patch separately)
    if perturbation is not None:
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
        plt.imshow(adversarial_image_tosave)
        plt.title('Adversarial')
        plt.axis('off')
        
        comparison_path = f"{animal_output_dir}/{animal_name}_{attack_method}_comparison.png"
        plt.savefig(comparison_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved comparison to: {comparison_path}")
    else:
        # For sticker attack, show original vs adversarial side by side
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(adversarial_image_tosave)
        plt.title('With Adversarial Sticker')
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
    
    if target_class:
        # Targeted attack success: source class suppressed AND target class detected
        source_suppressed = (scores_adv is None or len(scores_adv) == 0 or 
                           scores_adv.max().item() < 0.3)
        target_detected = (boxes_target is not None and len(boxes_target) > 0 and 
                          scores_target.max().item() > 0.3)
        attack_success = source_suppressed and target_detected
        print(f"Source class suppressed: {source_suppressed}")
        print(f"Target class detected: {target_detected}")
    else:
        # Untargeted attack success: just suppress detections
        attack_success = (scores_orig is not None and len(scores_orig) > 0 and 
                         (scores_adv is None or len(scores_adv) == 0 or 
                          scores_adv.max().item() < scores_orig.max().item() * 0.5))
    print(f"Attack successful: {attack_success}")
    
    # Clean up GPU memory after attack
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description='SAM3 Adversarial Attack Framework')
    parser.add_argument('--attack', type=str, default='all', 
                       choices=['fgsm', 'pgd', 'pgm', 'cw', 'sticker', 'scorebased', 'decision', 'all'],
                       help='Attack method: fgsm, pgd/pgm, cw, sticker, scorebased, decision, or all (default: all)')
    parser.add_argument('--image', type=str, default='data/cat.jpg',
                       help='Path to input image')
    parser.add_argument('--prompt', type=str, default='cat',
                       help='Text prompt for detection (source class)')
    parser.add_argument('--target', type=str, default=None,
                       help='Target class for targeted attack (e.g., "dog" to make cat->dog). Works with C&W, sticker, score-based, and decision-based attacks.')
    parser.add_argument('--output-dir', type=str, default='adversarial_results',
                       help='Output directory for results')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Perturbation budget for FGSM/PGD (default: 0.1, range: 0.01-0.3)')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations for PGD/CW')
    parser.add_argument('--stealthy', action='store_true',
                       help='Use higher perceptual weights for less visible perturbations (C&W only)')
    parser.add_argument('--patch-size', type=int, default=150,
                       help='Size of adversarial sticker/patch (sticker attack only)')
    parser.add_argument('--patch-location', type=str, default='center',
                       choices=['center', 'random', 'top-left', 'top-right', 'bottom-left', 'bottom-right'],
                       help='Location to place adversarial sticker (sticker attack only)')
    parser.add_argument('--max-size', type=int, default=None,
                       help='Maximum image dimension (if set, images larger than this will be downsampled to prevent OOM errors)')
    
    args = parser.parse_args()
    
    # Determine which attacks to run
    if args.attack == 'all':
        attacks_to_run = ['fgsm', 'pgd', 'cw', 'sticker', 'scorebased', 'decision']
    else:
        attacks_to_run = [args.attack]
    
    # Validate targeted attack
    if args.target:
        for attack in attacks_to_run:
            if attack not in ['cw', 'sticker', 'scorebased', 'decision']:
                print(f"WARNING: Targeted attacks (--target) only supported for C&W, sticker, score-based, and decision-based attacks.")
                print(f"         Will skip targeted mode for {attack.upper()}.")
        # Filter to only attacks that support targeting
        if args.attack == 'all':
            attacks_to_run_with_target = ['cw', 'sticker', 'scorebased', 'decision']
            attacks_to_run_without_target = [a for a in attacks_to_run if a not in attacks_to_run_with_target]
        else:
            attacks_to_run_with_target = [a for a in attacks_to_run if a in ['cw', 'sticker', 'scorebased', 'decision']]
            attacks_to_run_without_target = [a for a in attacks_to_run if a not in ['cw', 'sticker', 'scorebased', 'decision']]
    else:
        attacks_to_run_with_target = []
        attacks_to_run_without_target = attacks_to_run
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run attacks without targeting first
    for attack_method in attacks_to_run_without_target:
        test_adversarial_attack(attack_method, args.image, args.prompt, args.output_dir, 
                               None, args.stealthy, args.patch_size, args.patch_location, args.max_size, 
                               args.epsilon, args.iterations)
    
    # Run attacks with targeting
    for attack_method in attacks_to_run_with_target:
        test_adversarial_attack(attack_method, args.image, args.prompt, args.output_dir, 
                               args.target, args.stealthy, args.patch_size, args.patch_location, args.max_size, 
                               args.epsilon, args.iterations)
    
    print(f"\n{'='*70}")
    print(f"All results saved in '{args.output_dir}/{args.prompt}/' folder")
    print('='*70) 

if __name__ == "__main__":
    main()
