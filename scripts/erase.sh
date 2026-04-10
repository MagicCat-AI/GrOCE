#!/bin/bash

declare -A targets_map
declare -A anchors_map
declare -A contents_map
# ========== Input Params ==========
baselines=(
  "GrOCE"
)
param_groups=(
  "V 10 1e-1"
)
# ==================================

erase_types=("instance" "style")
# ==================================================================
targets_map["instance"]="Snoopy;Snoopy, Mickey;Snoopy, Mickey, Spongebob"
anchors_map["instance"]=" ; ; "
contents_map["instance"]="Snoopy, Mickey, Spongebob, Pikachu" 
# ==================================================================
targets_map["style"]="Van Gogh;Picasso;Monet"
anchors_map["style"]="art;art;art"
contents_map["style"]="Van Gogh, Picasso, Monet, Caravaggio" 
# ==================================================================

GPU_IDX=('0')
GPU_ID=${GPU_IDX[0]}

run_task() {
  local baseline=$1
  local erase_type=$2
  local target=$3
  local anchor=$4
  local contents=$5
  local a=$6
  local b=$7
  local c=$8
  local save_root=$9

  target_concepts=$(echo "$target" | sed 's/[[:space:]]*,[[:space:]]*/,/g') 
  anchor_concepts=$(echo "$anchor" | sed 's/[[:space:]]*,[[:space:]]*/,/g')

  num=$(echo "$target_concepts" | tr -cd ',' | wc -c)
  num=$((num + 1))
  limited_target=$(echo "$target_concepts" | awk -F',' '{for (i=1; i<=NF && i<=5; i++) printf (i<NF && i<5 ? $i "_": $i)}')
  if [ "$num" -gt 5 ]; then
    limited_target="${limited_target}_${num}"
  fi

  echo "[$baseline] Running erase_type=$erase_type, targets=$target_concepts, anchors=$anchor_concepts, contents=$contents"

  top_k_arg=""
  if [ "$erase_type" == "instance" ]; then
    top_k_arg="--top_k 8"
  elif [ "$erase_type" == "style" ]; then
    top_k_arg="--top_k 3"
  fi
  similarity_threshold_arg=""
  if [ "$erase_type" == "instance" ]; then
    similarity_threshold_arg="--similarity_threshold 0.3"
  elif [ "$erase_type" == "style" ]; then
    similarity_threshold_arg="--similarity_threshold 0.7"
  fi

  n_step_arg=""
  if [ "$erase_type" == "instance" ]; then
    n_step_arg="--n_step 2"
  elif [ "$erase_type" == "style" ]; then
    n_step_arg="--n_step 1"
  fi
  CUDA_VISIBLE_DEVICES=$GPU_ID python ./sample_erase.py \
    --erase_type "$erase_type" \
    --target_concepts "$target_concepts" \
    --save_root "$save_root/$limited_target"\
    --contents "$contents" \
    $top_k_arg\
    $n_step_arg\
    $similarity_threshold_arg
}


for baseline in "${baselines[@]}"; do
  for hypers in "${param_groups[@]}"; do
    read a b c <<< "$hypers"
    save_root="./logs/${baseline}" 

    for erase_type in "${erase_types[@]}"; do

      IFS=';' read -ra targets <<< "${targets_map[$erase_type]}"
      IFS=';' read -ra anchors <<< "${anchors_map[$erase_type]}" 
      contents="${contents_map[$erase_type]}"


      for i in "${!targets[@]}"; do
        target="${targets[i]}"
        anchor="${anchors[i]}"

        run_task "$baseline" "$erase_type" "$target" "$anchor" "$contents" "$a" "$b" "$c" "$save_root/$erase_type"
      done


      echo "Evaluating erase_type=$erase_type..."
      CUDA_VISIBLE_DEVICES=$GPU_ID python src/clip_score_cal.py \
        --contents "$contents" \
        --root_path "$save_root/$erase_type" \
        --pretrained_path "pretrain/$erase_type"
    done
  done
done