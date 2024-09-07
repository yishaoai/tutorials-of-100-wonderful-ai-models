export MODEL_DIR="black-forest-labs/FLUX.1-dev"
export OUTPUT_DIR="results_dir"
export TRAIN_JSON_FILE="data.jsonl"






#accelerate launch diffusers/examples/controlnet/train_controlnet_flux.py \
accelerate launch --config_file "default_config.yaml" diffusers/examples/controlnet/train_controlnet_flux.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --conditioning_image_column=control_path \
    --image_column=image_path \
    --caption_column=caption \
    --output_dir=$OUTPUT_DIR \
    --jsonl_for_train=$TRAIN_JSON_FILE \
    --mixed_precision="bf16" \
    --resolution=128 \
    --learning_rate=1e-5 \
    --max_train_steps=15000 \
    --validation_steps=10 \
    --checkpointing_steps=10 \
    --validation_image "./images/1_control.png" "./images/2_control.png" \
    --validation_prompt "children's clothing model" "children's clothing model" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --report_to="tensorboard" \
    --num_double_layers=4 \
    --num_single_layers=0 \
    --seed=42 \
    --gradient_checkpointing \
    --flux_controlnet_model_name_or_path InstantX/FLUX.1-dev-Controlnet-Canny 
