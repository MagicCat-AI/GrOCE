

## Environment Configuration


```bash
# create conda environment
conda create -n GrOCE -y python=3.11
conda activate GrOCE
cd GrOCE
# install pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# install other dependencies
pip install -r requirements.txt
```


## GrOCE Concept Erasure

### Establishment of the Graph
```bash
python knowledge.py
```

### Generate Erased Images

```bash
python sample_erase.py \
  --save_root "./logs" \
  --sd_ckpt "CompVis/stable-diffusion-v1-4" \
  --mode "edit" \
  --target_concepts "Snoopy" \
  --prompts "A blonde Snoopy with blue eyes sits lazily on the golden, fine sand of the beach." \
  --erase_type "instance" \
  --projection_threshold 5
  --network_path "./concept_network_results/concept_network.json"
```

## GrOCE Metrics Evaluation

### Establishment of the Graph
```bash
python knowledge.py
```

### Generate Original Images
```bash
bash scripts/origin.sh
```
### Generate Erased Images
```bash
bash scripts/erase.sh
```


