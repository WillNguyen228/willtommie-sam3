# How to Run Adversarial Attacks from Subfolder

Since the adversarial attack scripts have been moved to the `Adversarial_attack_code/` subfolder, here's how to run them:

## Quick Start

### Option 1: Run from the subfolder (Recommended)

```bash
# Navigate to the subfolder
cd ~/sam3/Adversarial_attack_code

# Activate the conda environment
conda activate sam3_1

# Run the attack (note: use ../data/ for image paths)
python run_adversarial_attacks.py --attack sticker --image ../data/cat.jpg --prompt cat --target dog
```

### Option 2: Use the full Python path (no conda activation needed)

```bash
cd ~/sam3/Adversarial_attack_code
/home/will/miniconda3/envs/sam3_1/bin/python run_adversarial_attacks.py --attack sticker --image ../data/cat.jpg --prompt cat --target dog
```

### Option 3: Use the demo scripts

```bash
cd ~/sam3/Adversarial_attack_code

# For SAM3 attacks
bash demo_adversarial_attacks.sh

# For ResNet attacks
bash demo_resnet_attacks.sh
```

## Important Notes

1. **Image paths**: When running from the subfolder, use `../data/` instead of `data/` for images
2. **Environment**: Make sure you're using the `sam3_1` conda environment
3. **Import fix**: The scripts have been updated to automatically add the parent directory to Python's path so the `sam3` module can be imported

## Example Commands

```bash
# Sticker attack (targeted)
python run_adversarial_attacks.py --attack sticker --image ../data/cat.jpg --prompt cat --target dog

# FGSM attack
python run_adversarial_attacks.py --attack fgsm --image ../data/cat.jpg --prompt cat --epsilon 0.05

# C&W attack (targeted)
python run_adversarial_attacks.py --attack cw --image ../data/cat.jpg --prompt cat --target dog --iterations 100

# Run all attacks
python run_adversarial_attacks.py --attack all --image ../data/cat.jpg --prompt cat
```

## Troubleshooting

If you get `ModuleNotFoundError: No module named 'sam3'`:
- Make sure you're using the `sam3_1` conda environment
- Verify the parent directory path fix is in the script (check lines 6-9 of run_adversarial_attacks.py)

If you get `FileNotFoundError` for the image:
- Use `../data/cat.jpg` instead of `data/cat.jpg` when running from the subfolder
- Or use absolute paths: `/home/will/sam3/data/cat.jpg`
