#!/bin/bash

declare -A contents_map

erase_types=("instance" "style")
contents_map["instance"]="Snoopy, Mickey, Spongebob, Pikachu"
contents_map["style"]="Van Gogh, Picasso, Monet,Caravaggio"


GPU_IDX=('0')
NUM_GPUS=${#GPU_IDX[@]} 

gpu_idx=0

run_task() {
  local erase_type=$1
  local gpu_id=$2

  echo "Running task for $erase_type on GPU $gpu_id"

  CUDA_VISIBLE_DEVICES=$gpu_id python ./sample_origin.py \
    --erase_type "$erase_type" \
    --target_concept "$erase_type" \
    --contents "${contents_map[$erase_type]}" \
    --num_samples 10 --batch_size 10 \
    --save_root "pretrain" &
}

for erase_type in "${erase_types[@]}"; do
  run_task "$erase_type" ${GPU_IDX[$gpu_idx]}

  gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))
  wait
done


wait