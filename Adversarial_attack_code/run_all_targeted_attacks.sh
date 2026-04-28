#!/bin/bash
# Run all targeted adversarial attacks with horse as target
# Saves output logs for analysis

OUTPUT_LOG="attack_results_log.txt"
echo "Starting targeted attack experiments at $(date)" > "$OUTPUT_LOG"
echo "=============================================" >> "$OUTPUT_LOG"

# Function to run attack and log output
run_attack() {
    local image=$1
    local prompt=$2
    local attack=$3
    local extra_args=$4
    
    echo "" >> "$OUTPUT_LOG"
    echo "========================================" >> "$OUTPUT_LOG"
    echo "IMAGE: $prompt | ATTACK: $attack" >> "$OUTPUT_LOG"
    echo "========================================" >> "$OUTPUT_LOG"
    
    python run_adversarial_attacks.py --attack "$attack" --image "$image" --prompt "$prompt" --target horse $extra_args 2>&1 | tee -a "$OUTPUT_LOG"
    
    # Clear GPU memory between attacks
    python -c "import torch, gc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null
    sleep 2
    
    echo "Completed: $prompt - $attack"
}

# Array of images and prompts
declare -a images=("../data/cat.jpg" "../data/dog.png" "../data/human.jpg" "../data/panda.png" "../data/rabbit.jpg" "../data/tiger.png")
declare -a prompts=("cat" "dog" "human" "panda" "rabbit" "tiger")

# Run all attacks for each image
for i in "${!images[@]}"; do
    image="${images[$i]}"
    prompt="${prompts[$i]}"
    
    echo ""
    echo "================================================"
    echo "Processing: $prompt"
    echo "================================================"
    
    # FGSM
    run_attack "$image" "$prompt" "fgsm" "--epsilon 0.9"
    
    # PGD
    run_attack "$image" "$prompt" "pgd" "--epsilon 0.1 --iterations 20"
    
    # C&W
    run_attack "$image" "$prompt" "cw" ""
    
    # Score-based
    run_attack "$image" "$prompt" "scorebased" ""
    
    # Decision-based
    run_attack "$image" "$prompt" "decision" ""
    
    # Sticker
    run_attack "$image" "$prompt" "sticker" ""
    
    echo "Finished all attacks for $prompt"
    echo ""
    
    # Extra cleanup between images
    python -c "import torch, gc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null
    sleep 3
done

echo "" >> "$OUTPUT_LOG"
echo "=============================================" >> "$OUTPUT_LOG"
echo "All experiments completed at $(date)" >> "$OUTPUT_LOG"
echo "" >> "$OUTPUT_LOG"

echo ""
echo "All attacks completed! Results saved to $OUTPUT_LOG"
echo "Run analyze_attack_results.py to generate visualizations"
