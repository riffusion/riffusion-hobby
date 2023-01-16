export MODEL_NAME="riffusion/riffusion-model-v1"
export INSTANCE_DIR="/tmp/sample_clips_tdlcqdfi/images"
export OUTPUT_DIR="/home/ubuntu/lora_dreambooth_waterfalls_2k"

accelerate launch\
  --num_machines 1 \
  --num_processes 8 \
  --dynamo_backend=no \
  --mixed_precision="fp16" \
  riffusion/external/lora/train_lora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="style of sks" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000

# TODO try mixed_precision=fp16
# TODO try num_processes = 8
