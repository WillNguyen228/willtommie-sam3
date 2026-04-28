#!/bin/bash

# Run all defense tests on all adversarial results
# Logs output to defense_results_log.txt

LOG_FILE="defense_results_log.txt"

echo "Starting defense evaluation at $(date)" > $LOG_FILE
echo "=============================================" >> $LOG_FILE
echo "" >> $LOG_FILE

IMAGES=("cat" "dog" "human" "panda" "rabbit" "tiger")

for img in "${IMAGES[@]}"; do
    echo "" >> $LOG_FILE
    echo "========================================" >> $LOG_FILE
    echo "IMAGE: $img | DEFENSE: all techniques" >> $LOG_FILE
    echo "========================================" >> $LOG_FILE
    
    python run_defense.py -d adversarial_results/$img -m sam3 -t all 2>&1 | tee -a $LOG_FILE
    
    echo "" >> $LOG_FILE
done

echo "" >> $LOG_FILE
echo "=============================================" >> $LOG_FILE
echo "All defense tests completed at $(date)" >> $LOG_FILE
echo "=============================================" >> $LOG_FILE

echo ""
echo "All defense tests complete!"
echo "Results logged to: $LOG_FILE"
