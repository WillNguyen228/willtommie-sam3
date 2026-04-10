# ResNet Adversarial Attack Framework

This framework implements the same 4 adversarial attack techniques from SAM3, adapted for ResNet-18 image classification:

1. **FGSM** (Fast Gradient Sign Method) - Fast single-step attack
2. **PGD** (Projected Gradient Descent) - Iterative multi-step attack  
3. **C&W** (Carlini & Wagner) - Optimization-based attack
4. **Adversarial Sticker** - Localized patch attack

## Quick Start

### Basic Usage

```bash
# Download a test image first
wget https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg -O cat.jpg

# FGSM attack
python run_resnet_adversarial_attacks.py --attack fgsm --image cat.jpg

# PGD attack
python run_resnet_adversarial_attacks.py --attack pgd --image cat.jpg

# C&W attack
python run_resnet_adversarial_attacks.py --attack cw --image cat.jpg

# Adversarial sticker
python run_resnet_adversarial_attacks.py --attack sticker --image cat.jpg
```

### Targeted Attacks

Make ResNet misclassify a cat as a dog:

```bash
python run_resnet_adversarial_attacks.py \
    --attack cw \
    --image cat.jpg \
    --target-class dog \
    --iterations 150
```

Sticker attack to fool classifier:

```bash
python run_resnet_adversarial_attacks.py \
    --attack sticker \
    --image cat.jpg \
    --target-class dog \
    --patch-size 60 \
    --patch-location center
```

## Attack Methods Comparison

### FGSM (Fast Gradient Sign Method)
- **Speed**: ⚡ Very fast (single step)
- **Effectiveness**: ⭐⭐ Moderate
- **Use case**: Quick testing, baseline attacks
- **Parameters**: `--epsilon` (perturbation strength, default: 0.03)

```bash
python run_resnet_adversarial_attacks.py \
    --attack fgsm \
    --image cat.jpg \
    --epsilon 0.05
```

### PGD (Projected Gradient Descent)  
- **Speed**: ⚡⚡ Moderate (iterative)
- **Effectiveness**: ⭐⭐⭐ Good
- **Use case**: Stronger attacks than FGSM, good balance
- **Parameters**: 
  - `--epsilon` (perturbation budget, default: 0.03)
  - `--iterations` (number of steps, default: 10)

```bash
python run_resnet_adversarial_attacks.py \
    --attack pgd \
    --image dog.jpg \
    --epsilon 0.03 \
    --iterations 20
```

### C&W (Carlini & Wagner)
- **Speed**: ⚡⚡⚡ Slow (optimization-based)
- **Effectiveness**: ⭐⭐⭐⭐ Very strong
- **Use case**: Finding minimal perturbations, strongest attacks
- **Parameters**: `--iterations` (optimization steps, default: 100)

```bash
python run_resnet_adversarial_attacks.py \
    --attack cw \
    --image bird.jpg \
    --iterations 150 \
    --target-class airplane
```

### Adversarial Sticker
- **Speed**: ⚡⚡⚡ Slow (optimization-based)
- **Effectiveness**: ⭐⭐⭐⭐ Very strong
- **Use case**: Physical-world attacks, localized perturbations
- **Special**: Creates a **printable sticker** that can fool classifiers
- **Parameters**:
  - `--patch-size` (sticker diameter in pixels, default: 50)
  - `--patch-location` (center/top-left/top-right/bottom-left/bottom-right)
  - `--iterations` (default: 200)

```bash
python run_resnet_adversarial_attacks.py \
    --attack sticker \
    --image person.jpg \
    --target-class dog \
    --patch-size 80 \
    --patch-location center \
    --iterations 300
```

**Print the sticker!** The sticker image is saved separately and can be printed to test physical-world attacks.

## Parameters

### Required
- `--attack`: Attack method (`fgsm`, `pgd`, `cw`, `sticker`)
- `--image`: Path to input image

### Optional
- `--output-dir`: Output directory (default: `resnet_adversarial_results`)
- `--epsilon`: Perturbation budget for FGSM/PGD (default: 0.03)
- `--iterations`: Number of iterations (default varies by attack)
- `--target-class`: Target class name for targeted attack (e.g., `dog`, `airplane`, `toaster`)
- `--patch-size`: Sticker size in pixels (default: 50)
- `--patch-location`: Sticker placement (default: `center`)

## Output Files

Results are saved in `resnet_adversarial_results/` (or your custom `--output-dir`):

1. `{image}_adversarial_{attack}.png` - Adversarial image
2. `{image}_{attack}_comparison.png` - Side-by-side visualization
3. `{image}_sticker_{attack}.png` - Just the sticker (for sticker attacks)

## Examples by Use Case

### 1. Quick untargeted attack
```bash
python run_resnet_adversarial_attacks.py --attack fgsm --image cat.jpg
```

### 2. Strong targeted misclassification
```bash
python run_resnet_adversarial_attacks.py \
    --attack cw \
    --image cat.jpg \
    --target-class dog \
    --iterations 200
```

### 3. Physical-world printable sticker
```bash
python run_resnet_adversarial_attacks.py \
    --attack sticker \
    --image stop_sign.jpg \
    --target-class "speed limit" \
    --patch-size 100 \
    --iterations 400
```

### 4. Compare all 4 attacks
```bash
#!/bin/bash
IMAGE="cat.jpg"

python run_resnet_adversarial_attacks.py --attack fgsm --image $IMAGE
python run_resnet_adversarial_attacks.py --attack pgd --image $IMAGE --iterations 20
python run_resnet_adversarial_attacks.py --attack cw --image $IMAGE --iterations 100
python run_resnet_adversarial_attacks.py --attack sticker --image $IMAGE --patch-size 60
```

## Understanding Results

### Success Metrics

**Untargeted Attack Success:**
- ✓ Original and adversarial predictions are different
- ✓ Confidence in original class drops significantly

**Targeted Attack Success:**
- ✓ Adversarial prediction matches target class
- ✓ Target class has high confidence (>50%)

### Example Output

```
Original Classification:
  Top prediction: tabby cat (85.3%)
  
Adversarial Classification:
  Top prediction: dog (72.1%)

✓ ATTACK SUCCESSFUL - Classification changed!
✓ TARGETED ATTACK SUCCESSFUL - Misclassified as 'dog'!
```

## ImageNet Classes

ResNet-18 is trained on ImageNet with 1000 classes. Common classes include:

**Animals**: `cat`, `dog`, `bird`, `fish`, `horse`, `elephant`, `bear`, `tiger`, `lion`
**Vehicles**: `car`, `truck`, `airplane`, `boat`, `bicycle`, `motorcycle`
**Objects**: `chair`, `table`, `bottle`, `cup`, `phone`, `laptop`, `keyboard`
**Food**: `pizza`, `burger`, `apple`, `banana`, `strawberry`

See full list: [ImageNet Classes](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)

## Comparison with SAM3 Attacks

| Feature | SAM3 Attacks | ResNet Attacks |
|---------|--------------|----------------|
| Task | Segmentation | Classification |
| Model | SAM3 | ResNet-18 |
| Input | Text prompts + images | Images only |
| Output | Masks/detections | Class labels |
| Attack goal | Hide/show objects | Misclassify images |
| Same techniques | FGSM, PGD, C&W, Sticker | ✓ |

## Tips for Best Results

1. **Start with FGSM** for quick testing
2. **Use PGD** for better results with reasonable speed
3. **Use C&W** when you need guaranteed success
4. **Use Sticker** for physical-world applications

### Epsilon Guidelines
- Small (0.01-0.02): Subtle, might not fool model
- Medium (0.03-0.05): Good balance (recommended)
- Large (0.1+): Very noticeable, but highly effective

### Iteration Guidelines
- **PGD**: 10-40 iterations
- **C&W**: 100-200 iterations
- **Sticker**: 200-500 iterations

## Research Applications

This framework is useful for:
- **Adversarial robustness testing**
- **Model debugging and understanding**
- **Security research**
- **Physical-world attack experiments**

## Advanced: Creating Universal Adversarial Patches

You can use the sticker attack to create patches that fool the classifier on MULTIPLE images:

```bash
# Create sticker that makes any image look like a toaster
python run_resnet_adversarial_attacks.py \
    --attack sticker \
    --image random_image.jpg \
    --target-class toaster \
    --patch-size 100 \
    --iterations 500
```

Then test the sticker on different images by overlaying it!

## Troubleshooting

**Attack not working?**
- Increase `--iterations`
- Increase `--epsilon` (for FGSM/PGD)
- Try a different attack method
- Make sure target class is in ImageNet

**Out of memory?**
- Use smaller images
- Reduce `--patch-size` (for sticker)
- Use CPU instead of GPU (slower)

## See Also

- `run_adversarial_attacks.py` - Original SAM3 adversarial attacks
- `diffusion_lab.py` - Example of using ResNet for classification
