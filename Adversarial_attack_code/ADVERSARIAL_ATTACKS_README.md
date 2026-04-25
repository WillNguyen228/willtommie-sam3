# SAM3 Adversarial Attack Framework

This framework provides implementations of six adversarial attack methods on SAM3:
- **FGSM** (Fast Gradient Sign Method) - Fast single-step attack
- **PGD** (Projected Gradient Descent) - Iterative multi-step attack  
- **C&W** (Carlini & Wagner) - Optimization-based attack
- **Sticker** - Localized patch-based attack
- **Score-Based** - Query-efficient black-box attack
- **Decision-Based** - Boundary attack for minimal perturbations

**Key Features:**
- Supports both **untargeted** and **targeted** attacks
- Preserves original image dimensions (no square resizing in output)
- Organized output by animal/object name in separate folders
- Comprehensive visualization with comparison images

## Usage

### Basic Usage

Run an attack using keywords: `fgsm`, `pgd` (or `pgm`), `cw`, `sticker`, `scorebased`, `decision`, or `all`

```bash
# FGSM attack on cat image
python run_adversarial_attacks.py --attack fgsm --image ../data/cat.jpg --prompt cat

# PGD attack on dog image  
python run_adversarial_attacks.py --attack pgd --image ../data/dog.png --prompt dog

# C&W attack on panda
python run_adversarial_attacks.py --attack cw --image ../data/panda.png --prompt panda

# Sticker attack (places adversarial patch on image)
python run_adversarial_attacks.py --attack sticker --image ../data/cat.jpg --prompt cat

# Score-based attack (query-efficient black-box)
python run_adversarial_attacks.py --attack scorebased --image ../data/cat.jpg --prompt cat

# Decision-based attack (minimal boundary perturbations)
python run_adversarial_attacks.py --attack decision --image ../data/cat.jpg --prompt cat

# Run all attacks
python run_adversarial_attacks.py --attack all --image ../data/cat.jpg --prompt cat
```

### Targeted Attacks

Add `--target` to perform targeted attacks that make the model detect a different class:

```bash
# Targeted FGSM: Make cat image be detected as dog
python run_adversarial_attacks.py --attack fgsm --image ../data/cat.jpg --prompt cat --target dog

# Targeted C&W: Make tiger image be detected as horse
python run_adversarial_attacks.py --attack cw --image ../data/tiger.png --prompt tiger --target horse

# Targeted sticker attack
python run_adversarial_attacks.py --attack sticker --image ../data/cat.jpg --prompt cat --target dog
```

### Advanced Options

```bash
# FGSM with custom epsilon (perturbation strength)
python run_adversarial_attacks.py --attack fgsm --image ../data/cat.jpg --prompt cat --epsilon 0.05

# PGD with more iterations and custom epsilon
python run_adversarial_attacks.py --attack pgd --image ../data/dog.png --prompt dog --epsilon 0.03 --iterations 20

# C&W with custom iterations
python run_adversarial_attacks.py --attack cw --image ../data/tiger.png --prompt tiger --iterations 150

# C&W with stealthy mode (less visible perturbations)
python run_adversarial_attacks.py --attack cw --image ../data/cat.jpg --prompt cat --stealthy

# Sticker attack with custom patch size and location
python run_adversarial_attacks.py --attack sticker --image ../data/cat.jpg --prompt cat --patch-size 200 --patch-location top-left

# Score-based attack with size limit for large images
python run_adversarial_attacks.py --attack scorebased --image ../data/cat.jpg --prompt cat --max-size 1024

# Custom output directory
python run_adversarial_attacks.py --attack fgsm --image ../data/rabbit.jpg --prompt rabbit --output-dir my_results
```

## Parameters

- `--attack`: Attack method (`fgsm`, `pgd`, `pgm`, `cw`, `sticker`, `scorebased`, `decision`, or `all`)
- `--image`: Path to input image
- `--prompt`: Text prompt for object detection (source class)
- `--target`: Target class for targeted attack (e.g., "dog" to make cat->dog). Works with all attacks
- `--output-dir`: Directory to save results (default: `adversarial_results`)
- `--epsilon`: Perturbation budget for FGSM/PGD (default: 0.1, range: 0.01-0.3)
- `--iterations`: Number of iterations for PGD/CW (default: 10)
- `--stealthy`: Use higher perceptual weights for less visible perturbations (C&W only)
- `--patch-size`: Size of adversarial sticker/patch (sticker attack only, default: 150)
- `--patch-location`: Location to place adversarial sticker (`center`, `random`, `top-left`, `top-right`, `bottom-left`, `bottom-right`)
- `--max-size`: Maximum image dimension for downsampling large images (recommended for score-based attacks)

## Attack Methods

### FGSM (Fast Gradient Sign Method)
- **Speed**: Very fast (single step)
- **Effectiveness**: Moderate
- **Best for**: Quick testing, baseline attacks
- **Parameters**: `--epsilon` controls perturbation strength

### PGD (Projected Gradient Descent)  
- **Speed**: Moderate (iterative)
- **Effectiveness**: Good
- **Best for**: Stronger attacks than FGSM, balanced speed/effectiveness
- **Parameters**: `--epsilon` (perturbation budget), `--iterations` (number of steps)

### C&W (Carlini & Wagner)
- **Speed**: Slow (optimization-based)
- **Effectiveness**: Very strong
- **Best for**: Finding minimal perturbations, strongest attacks
- **Parameters**: `--iterations` (optimization steps), `--stealthy` (less visible perturbations)

### Sticker Attack
- **Speed**: Moderate
- **Effectiveness**: Good for localized attacks
- **Best for**: Physical-world attacks, patch-based adversarial examples
- **Parameters**: `--patch-size`, `--patch-location`

### Score-Based Attack
- **Speed**: Moderate
- **Effectiveness**: Good for black-box scenarios
- **Best for**: Query-efficient attacks without gradient access
- **Parameters**: `--max-size` (for large images), `--iterations`

### Decision-Based Attack
- **Speed**: Slow
- **Effectiveness**: Very strong
- **Best for**: Minimal perturbations, boundary exploration
- **Parameters**: `--iterations`

## Output Files

Results are organized by animal/object name. For each attack, the following files are generated in `{output_dir}/{animal_name}/`:

### Untargeted Attacks:
1. `{animal}_original.png` - Original image with detections
2. `{animal}_adversarial_{attack}.png` - Raw adversarial image (perturbation applied)
3. `{animal}_adversarial_{attack}_result.png` - Adversarial image with detections for source class
4. `{animal}_{attack}_comparison.png` - Side-by-side comparison showing original, perturbation, and adversarial

### Targeted Attacks (when `--target` is specified):
1. All files from untargeted attacks, plus:
2. `{animal}_adversarial_{attack}_target_{target_class}.png` - Adversarial image with detections for target class

**Example folder structure:**
```
adversarial_results/
├── cat/
│   ├── cat_original.png
│   ├── cat_adversarial_fgsm.png
│   ├── cat_adversarial_fgsm_result.png
│   ├── cat_adversarial_fgsm_target_dog.png  # Targeted
│   ├── cat_fgsm_comparison.png
│   ├── cat_adversarial_pgd.png
│   ├── cat_adversarial_pgd_result.png
│   ├── cat_adversarial_pgd_target_dog.png   # Targeted
│   ├── cat_pgd_comparison.png
│   ├── cat_adversarial_cw.png
│   ├── cat_adversarial_cw_result.png
│   ├── cat_adversarial_cw_target_dog.png    # Targeted
│   ├── cat_cw_comparison.png
│   ├── cat_adversarial_sticker.png
│   ├── cat_adversarial_sticker_result.png
│   ├── cat_adversarial_sticker_target_dog.png
│   ├── cat_sticker_comparison.png
│   ├── cat_adversarial_scorebased.png
│   ├── cat_adversarial_scorebased_result.png
│   ├── cat_adversarial_scorebased_target_horse.png
│   ├── cat_scorebased_comparison.png
│   ├── cat_adversarial_decision.png
│   ├── cat_adversarial_decision_result.png
│   ├── cat_adversarial_decision_target_horse.png
│   └── cat_decision_comparison.png
```

## Example Workflow

```bash
# Test all three attacks on the same image (untargeted)
python run_adversarial_attacks.py --attack fgsm --image ../data/cat.jpg --prompt cat
python run_adversarial_attacks.py --attack pgd --image ../data/cat.jpg --prompt cat  
python run_adversarial_attacks.py --attack cw --image ../data/cat.jpg --prompt cat

# Test targeted attacks (make cat appear as dog)
python run_adversarial_attacks.py --attack fgsm --image ../data/cat.jpg --prompt cat --target dog
python run_adversarial_attacks.py --attack cw --image ../data/cat.jpg --prompt cat --target dog

# Test sticker attack
python run_adversarial_attacks.py --attack sticker --image ../data/cat.jpg --prompt cat --target dog

# Run all attacks (both targeted and untargeted)
python run_adversarial_attacks.py --attack all --image ../data/cat.jpg --prompt cat --target dog

# Results will show:
# - Number of detections before/after attack
# - Confidence scores for source class
# - For targeted attacks: Confidence scores for target class
# - Whether attack was successful
```

## Understanding Results

**Untargeted Attack Success**: Reduces detection confidence or number of detections significantly for the source class
- Original detections > Adversarial detections
- Original confidence > Adversarial confidence  

**Targeted Attack Success**: Reduces detection of source class AND increases detection of target class
- Source class: Original confidence > Adversarial confidence
- Target class: Adversarial confidence > Threshold (e.g., >0.5)

**Attack Success Criteria**: 
- **Untargeted**: No detections on adversarial image, OR max confidence reduced by >50%
- **Targeted**: Source confidence reduced by >50% AND target confidence >0.3

## Tips

1. **Start with FGSM**: Fastest way to test if model is vulnerable
2. **Tune epsilon**: Higher values = stronger but more visible perturbations (try 0.01-0.3)
3. **Use PGD for stronger attacks**: More effective than FGSM with minimal speed cost
4. **Use C&W for publication-quality results**: Finds minimal perturbations but slower
5. **Try targeted attacks**: Use `--target` to test if attacks can cause misclassification to specific classes
6. **Use sticker attacks for physical scenarios**: Good for real-world adversarial patches
7. **Score-based attacks for black-box**: When you don't have gradient access to the model
8. **Decision-based for minimal perturbations**: When you want the smallest possible changes
9. **Use --stealthy with C&W**: For less perceptible perturbations
10. **Downsample large images**: Use `--max-size` with score-based attacks to prevent memory issues
