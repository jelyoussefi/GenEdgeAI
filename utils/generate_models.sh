#!/bin/bash

# Check if required parameters are provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <HF_TOKEN> \"<MODELS>\" \"<PRECISIONS>\" <OUTPUT_DIR>"
    echo "Example: $0 your_huggingface_token \"deepseek-ai/DeepSeek-R1-Distill-Llama-8B TinyLlama/TinyLlama-1.1B-Chat-v1.0 meta-llama/Meta-Llama-3-8B-Instruct\" \"INT4 INT8\" /opt/models"
    exit 1
fi

# Assign parameters to variables
HF_TOKEN="$1"
MODELS="$2"
PRECISIONS="$3"
OUTPUT_DIR="$4"

# Validate HF_TOKEN
[ -z "$HF_TOKEN" ] && { echo "Error: Hugging Face token (HF_TOKEN) is required."; exit 1; }

# Validate MODELS and PRECISIONS
[ -z "$MODELS" ] && { echo "Error: MODELS is required."; exit 1; }
[ -z "$PRECISIONS" ] && { echo "Error: PRECISIONS is required."; exit 1; }

# Create output directory
mkdir -p "$OUTPUT_DIR" || { echo "Warning: Could not create output directory $OUTPUT_DIR"; exit 1; }

# Login to Hugging Face quietly
huggingface-cli login --token "$HF_TOKEN" > /dev/null 2>&1

# Process each model and precision combination
for model in $MODELS; do
    model_name=$(basename "$model")
    for precision in $PRECISIONS; do
        precision_lower=$(echo "$precision" | tr '[:upper:]' '[:lower:]')
        output_dir="$OUTPUT_DIR/$model_name/$precision/$model_name"
        mkdir -p "$output_dir" || { echo "Warning: Directory $output_dir may already exist or could not be created"; }

        # Check if model is already generated
        if [ -d "$output_dir" ] && [ -f "$output_dir/openvino_model.bin" ]; then
            continue  # Skip silently if output exists
        else
            optimum-cli export openvino \
                --model "$model" \
                --task text-generation-with-past \
                --group-size 64 \
                --ratio 1.0 \
                --weight-format "$precision_lower" \
                --trust-remote-code "$output_dir" || { echo "Error: Failed to generate $model in $precision format"; exit 1; }
        fi
    done
done
