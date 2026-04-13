#!/bin/bash
# Demo script to test all three adversarial attacks on SAM3

echo "========================================="
echo "SAM3 Adversarial Attacks Demo"
echo "========================================="
echo ""

# Activate environment
source /home/will/miniconda3/bin/activate sam3_1

# Create output directory
mkdir -p adversarial_demo_results

echo "1. Testing FGSM Attack..."
echo "-----------------------------------------"
python run_adversarial_attacks.py \
    --attack fgsm \
    --image data/cat.jpg \
    --prompt cat \
    --epsilon 0.05 \
    --output-dir adversarial_demo_results

echo ""
echo ""
echo "2. Testing PGD Attack..."
echo "-----------------------------------------"
python run_adversarial_attacks.py \
    --attack pgd \
    --image data/cat.jpg \
    --prompt cat \
    --epsilon 0.1 \
    --iterations 20 \
    --output-dir adversarial_demo_results

echo ""
echo ""
echo "3. Testing C&W Attack..."
echo "-----------------------------------------"
python run_adversarial_attacks.py \
    --attack cw \
    --image data/cat.jpg \
    --prompt cat \
    --iterations 50 \
    --output-dir adversarial_demo_results

echo ""
echo ""
echo "========================================="
echo "Demo Complete!"
echo "Results saved in: adversarial_demo_results/"
echo "========================================="
