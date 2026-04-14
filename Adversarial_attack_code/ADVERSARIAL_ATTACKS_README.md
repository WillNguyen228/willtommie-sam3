# SAM3 Adversarial Attack Framework

This framework provides implementations of three adversarial attack methods on SAM3:
- **FGSM** (Fast Gradient Sign Method) - Fast single-step attack
- **PGD** (Projected Gradient Descent) - Iterative multi-step attack  
- **C&W** (Carlini & Wagner) - Optimization-based attack

**Key Features:**
- Preserves original image dimensions (no square resizing in output)
- Organized output by animal/object name in separate folders
- Comprehensive visualization with comparison images

## Usage

### Basic Usage

Run an attack using keywords: `fgsm`, `pgd` (or `pgm`), or `cw`

```bash
# FGSM attack on cat image
python run_adversarial_attacks.py --attack fgsm --image ../data/cat.jpg --prompt cat

# PGD attack on dog image  
python run_adversarial_attacks.py --attack pgd --image ../data/dog.png --prompt dog

# C&W attack on panda
python run_adversarial_attacks.py --attack cw --image ../data/panda.png --prompt panda
```

### Advanced Options

```bash
# FGSM with custom epsilon (perturbation strength)
python run_adversarial_attacks.py --attack fgsm --image ../data/cat.jpg --prompt cat --epsilon 0.05

# PGD with more iterations and custom epsilon
python run_adversarial_attacks.py --attack pgd --image ../data/dog.png --prompt dog --epsilon 0.03 --iterations 20

# C&W with custom iterations
python run_adversarial_attacks.py --attack cw --image ../data/tiger.png --prompt tiger --iterations 150

# Custom output directory
python run_adversarial_attacks.py --attack fgsm --image ../data/rabbit.jpg --prompt rabbit --output-dir my_results
```

## Parameters

- `--attack`: Attack method (`fgsm`, `pgd`, `pgm`, or `cw`)
- `--image`: Path to input image
- `--prompt`: Text prompt for object detection
- `--output-dir`: Directory to save results (default: `adversarial_results`)
- `--epsilon`: Perturbation budget for FGSM/PGD (default: 0.03)
- `--iterations`: Number of iterations for PGD/CW (default: 10 for PGD, 100 for C&W)

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
- **Parameters**: `--iterations` (optimization steps)

## Output Files

Results are organized by animal/object name. For each attack, the following files are generated in `{output_dir}/{animal_name}/`:

1. `{animal}_original.png` - Original image with detections
2. `{animal}_adversarial_{attack}.png` - Adversarial image (raw)
3. `{animal}_adversarial_{attack}_result.png` - Adversarial image with detections
4. `{animal}_{attack}_comparison.png` - Side-by-side comparison showing original, perturbation, and adversarial

**Example folder structure:**
```
adversarial_results/
├── cat/
│   ├── cat_original.png
│   ├── cat_adversarial_fgsm.png
│   ├── cat_adversarial_fgsm_result.png
│   └── cat_fgsm_comparison.png
├── dog/
│   ├── dog_original.png
│   ├── dog_adversarial_pgd.png
│   ├── dog_adversarial_pgd_result.png
│   └── dog_pgd_comparison.png
└── tiger/
    ├── tiger_original.png
    ├── tiger_adversarial_cw.png
    ├── tiger_adversarial_cw_result.png
    └── tiger_cw_comparison.png
```

## Example Workflow

```bash
# Test all three attacks on the same image
python run_adversarial_attacks.py --attack fgsm --image ../data/cat.jpg --prompt cat
python run_adversarial_attacks.py --attack pgd --image ../data/cat.jpg --prompt cat  
python run_adversarial_attacks.py --attack cw --image ../data/cat.jpg --prompt cat

# Results will show:
# - Number of detections before/after attack
# - Confidence scores
# - Whether attack was successful
```

## Understanding Results

**Successful Attack**: Reduces detection confidence or number of detections significantly
- Original detections > Adversarial detections
- Original confidence > Adversarial confidence  

**Attack Success Criteria**: 
- No detections on adversarial image, OR
- Max confidence reduced by >50%

## Tips

1. **Start with FGSM**: Fastest way to test if model is vulnerable
2. **Tune epsilon**: Higher values = stronger but more visible perturbations (try 0.01-0.1)
3. **Use PGD for stronger attacks**: More effective than FGSM with minimal speed cost
4. **Use C&W for publication-quality results**: Finds minimal perturbations but slower
