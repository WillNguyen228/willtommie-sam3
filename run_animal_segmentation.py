# --- SAM3 Animal Segmentation Script ---
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs("animal_results", exist_ok=True)

# Load model
print("Loading SAM3 model...")
model = build_sam3_image_model()
processor = Sam3Processor(model, confidence_threshold=0.1)

# Define images and their prompts
images_to_process = [
    ("data/cat.jpg", "cat"),
    ("data/dog_adversarial_pgd.png", "dog"),
    ("data/panda.png", "panda"),
    ("data/rabbit.jpg", "rabbit"),
    ("data/tiger.png", "tiger")
]
#("data/dog.png", "dog"),

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
    
    # Set image for processing
    inference_state = processor.set_image(image)
    
    # Reset prompts and segment with text prompt
    processor.reset_all_prompts(inference_state)
    output = processor.set_text_prompt(
        state=inference_state,
        prompt=animal_name
    )
    
    # Get detection info
    boxes = output["boxes"]
    scores = output["scores"]
    
    if boxes is not None and len(scores) > 0:
        print(f"Detected {len(boxes)} {animal_name}(s):")
        for idx, (box, score) in enumerate(zip(boxes, scores)):
            box_list = [round(v.item(), 2) for v in box]
            print(f"  {animal_name} #{idx+1}: score={score.item():.4f}, bbox={box_list}")
    else:
        print(f"No {animal_name} detected.")
    
    # Plot and save the segmentation result
    plot_results(image, inference_state)
    output_path = f"animal_results/{animal_name}_segmented.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved segmentation to: {output_path}")

print(f"\n{'='*60}")
print("All images processed! Results saved in 'animal_results/' folder")
print('='*60)
