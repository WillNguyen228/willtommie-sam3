"""
ResNet-18 Animal Classification Script
- Classifies animals in images using pretrained ResNet-18
- Provides top-5 predictions with confidence scores
- Saves visualizations with predictions
"""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs("animal_results_resnet", exist_ok=True)

# Load ResNet-18 model
print("Loading ResNet-18 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
model.eval()

# Get ImageNet categories
categories = ResNet18_Weights.DEFAULT.meta["categories"]

# ImageNet normalization constants
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Define images and their expected animals
images_to_process = [
    ("data/cat.jpg", "cat"),
    ("data/dog.png", "dog"),
    ("data/panda.png", "panda"),
    ("data/rabbit.jpg", "rabbit"),
    ("data/tiger.png", "tiger")
]

def preprocess_image(image_pil):
    """Preprocess image for ResNet-18."""
    # Convert to tensor and normalize to [0, 1]
    img_tensor = v2.functional.to_image(image_pil.convert("RGB")).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Resize to 224x224 if needed
    if img_tensor.shape[-2:] != (224, 224):
        img_tensor = F.interpolate(img_tensor, size=(224, 224),
                                   mode="bilinear", align_corners=False)
    
    # Normalize using ImageNet stats
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    normalized = (img_tensor - mean_t) / std_t
    
    return normalized

def classify_image(image_tensor):
    """Run classification and return top-5 predictions."""
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        top5_probs, top5_idx = torch.topk(probs, 5, dim=1)
    
    predictions = [
        {"class": categories[idx.item()], "confidence": prob.item()}
        for prob, idx in zip(top5_probs[0], top5_idx[0])
    ]
    
    return predictions

def visualize_results(image_pil, predictions, animal_name, output_path):
    """Create visualization with image and predictions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Display image
    ax1.imshow(image_pil)
    ax1.axis('off')
    ax1.set_title(f"Input: {animal_name}", fontsize=14, fontweight='bold')
    
    # Display predictions as bar chart
    classes = [p["class"] for p in predictions]
    confidences = [p["confidence"] * 100 for p in predictions]
    colors = ['green' if animal_name.lower() in c.lower() else 'steelblue' 
              for c in classes]
    
    y_pos = np.arange(len(classes))
    ax2.barh(y_pos, confidences, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.invert_yaxis()
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Top-5 Predictions', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 100])
    
    # Add confidence values on bars
    for i, v in enumerate(confidences):
        ax2.text(v + 1, i, f'{v:.2f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

# Process each image
for image_path, animal_name in images_to_process:
    print(f"\n{'='*60}")
    print(f"Processing: {animal_name} ({image_path})")
    print('='*60)
    
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Image size: {image.size[0]}x{image.size[1]}")
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        continue
    
    # Preprocess
    image_tensor = preprocess_image(image)
    
    # Classify
    predictions = classify_image(image_tensor)
    
    # Display results
    print(f"Top-5 predictions for {animal_name}:")
    for idx, pred in enumerate(predictions, 1):
        marker = "✓" if animal_name.lower() in pred["class"].lower() else " "
        print(f"  {marker} #{idx}: {pred['class']:30s} - {pred['confidence']*100:6.2f}%")
    
    # Check if correct
    top1_class = predictions[0]["class"].lower()
    if animal_name.lower() in top1_class:
        print(f"✓ Correctly classified as {predictions[0]['class']}!")
    else:
        # Check if animal is in top-5
        found_in_top5 = any(animal_name.lower() in p["class"].lower() for p in predictions)
        if found_in_top5:
            print(f"⚠ Expected '{animal_name}' found in top-5, but not top-1")
        else:
            print(f"✗ Expected '{animal_name}' NOT found in top-5 predictions")
    
    # Save visualization
    output_path = f"animal_results_resnet/{animal_name}_classified.png"
    visualize_results(image, predictions, animal_name, output_path)
    print(f"Saved visualization to: {output_path}")

print(f"\n{'='*60}")
print("All images processed! Results saved in 'animal_results_resnet/' folder")
print('='*60)
