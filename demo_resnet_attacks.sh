#!/bin/bash
# Demo script for ResNet Adversarial Attacks
# Runs all 4 attack methods on a sample image

echo "========================================"
echo "ResNet Adversarial Attack Demo"
echo "========================================"
echo ""

# Check if we have a test image
if [ ! -f "cat_test.jpg" ]; then
    echo "Downloading test image..."
    wget -q https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg -O cat_test.jpg
    echo "✓ Downloaded cat_test.jpg"
fi

echo ""
echo "Running 4 adversarial attack methods on cat_test.jpg..."
echo ""

# FGSM Attack
echo "1/4 - Running FGSM (Fast Attack)..."
python run_resnet_adversarial_attacks.py \
    --attack fgsm \
    --image cat_test.jpg \
    --epsilon 0.03

echo ""
echo "2/4 - Running PGD (Iterative Attack)..."
python run_resnet_adversarial_attacks.py \
    --attack pgd \
    --image cat_test.jpg \
    --epsilon 0.03 \
    --iterations 15

echo ""
echo "3/4 - Running C&W (Optimization Attack)..."
python run_resnet_adversarial_attacks.py \
    --attack cw \
    --image cat_test.jpg \
    --iterations 100

echo ""
echo "4/4 - Running Adversarial Sticker..."
python run_resnet_adversarial_attacks.py \
    --attack sticker \
    --image cat_test.jpg \
    --patch-size 60 \
    --patch-location center \
    --iterations 200

echo ""
echo "========================================"
echo "Demo Complete!"
echo "========================================"
echo ""
echo "Results saved in: resnet_adversarial_results/"
echo ""
echo "Check these files:"
echo "  - cat_test_fgsm_comparison.png"
echo "  - cat_test_pgd_comparison.png"
echo "  - cat_test_cw_comparison.png"
echo "  - cat_test_sticker_comparison.png"
echo ""
echo "To run targeted attacks (e.g., make cat → dog):"
echo "  python run_resnet_adversarial_attacks.py --attack cw --image cat_test.jpg --target-class dog"
echo ""
