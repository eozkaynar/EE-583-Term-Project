#!/bin/bash

# Define hyperparameter ranges
learning_rates=(1e-4 1e-3 1e-5)           # Learning ratesk
patch_sizes=("4,4,4" "6,6,6" "8,8,8")     # Patch sizes
projection_dims=(64 128 256)              # Projection dimensions
num_heads=(4 8)                           # Number of attention heads
num_layers=(6 8)                          # Number of transformer layers

# Output directory
OUTPUT_DIR="hyperparameter_outputs"
mkdir -p "$OUTPUT_DIR"  # Create the output directory if it doesn't exist

# Record the start time
start_time=$(date)

# Hyperparameter tuning loop

for lr in "${learning_rates[@]}"; do
  for patch_size in "${patch_sizes[@]}"; do
    for projection_dim in "${projection_dims[@]}"; do
      for num_head in "${num_heads[@]}"; do
        for num_layer in "${num_layers[@]}"; do
        
          # Generate a unique filename for the current hyperparameter combination
          OUTPUT_FILE="${OUTPUT_DIR}/lr_${lr}_patch_${patch_size}_pd_${projection_dim}_nh_${num_head}_nl_${num_layer}.log"
          
          echo "Starting: , lr=$lr, patch_size=$patch_size, projection_dim=$projection_dim, num_heads=$num_head, num_layers=$num_layer"
          echo "Output will be saved to: $OUTPUT_FILE"
          
          # Run the Python training script and redirect output to the log file
          python3 /home/eda/Desktop/EE583/ViViT/utils/classification.py \
            --hyperparameter true \
            --lr "$lr" \
            --patch_size "$patch_size" \
            --projection_dim "$projection_dim" \
            --num_heads "$num_head" \
            --num_layers "$num_layer" > "$OUTPUT_FILE"
          
          # Log completion status
          echo "Completed:  lr=$lr, patch_size=$patch_size, projection_dim=$projection_dim, num_heads=$num_head, num_layers=$num_layer. Output saved to $OUTPUT_FILE."
        
        done
      done
    done
  done
done


# Record the end time
end_time=$(date)

# Summary
echo "Hyperparameter tuning completed."
echo "Log files are saved in the $OUTPUT_DIR directory."
echo "Start time: $start_time"
echo "End time: $end_time"

