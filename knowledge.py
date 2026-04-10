import os
import time
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from diffusers import DiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from collections import defaultdict
import json
import random
import faiss
from torch.utils.data import Dataset, DataLoader



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_valid_tokens(concept_embeddings, tokenizer):
    start_time = time.time()
    pattern = re.compile(r'^[A-Za-z]{2,}$')
    filtered_embeddings = {}
    
    for token_id, embedding in tqdm(concept_embeddings.items(), desc="Filtering valid tokens"):
        token_text = tokenizer.decode([token_id]).strip()
        if pattern.match(token_text):
            filtered_embeddings[token_id] = embedding
    
    elapsed_time = time.time() - start_time
    print(f"Filtering valid tokens completed in {elapsed_time:.2f} seconds")
    return filtered_embeddings


def compute_local_density_correction(embeddings_normalized, index, token_ids, batch_size=1024, k_neighbors=10):
    start_time = time.time()
    num_tokens = len(token_ids)
    local_means = np.zeros(num_tokens)
    local_stds = np.zeros(num_tokens)
    
    for start_idx in tqdm(range(0, num_tokens, batch_size), desc="Calculating local density correction"):
        end_idx = min(start_idx + batch_size, num_tokens)
        batch_embeddings = embeddings_normalized[start_idx:end_idx]
        
        D, I = index.search(batch_embeddings, k=k_neighbors + 1)
        
        for i in range(end_idx - start_idx):
            similarities = D[i][1:]
            local_means[start_idx + i] = np.mean(similarities)
            local_stds[start_idx + i] = np.std(similarities)
    
    elapsed_time = time.time() - start_time
    print(f"Local density correction completed in {elapsed_time:.2f} seconds")
    return local_means, local_stds


def build_concept_network(concept_embeddings, tokenizer, similarity_threshold=0.7, 
                         max_connections=10, batch_size=1024, sigma=0.1, lambda_param=0.1):
    start_time = time.time()
    token_ids = list(concept_embeddings.keys())
    
    if not token_ids:
        print("No valid tokens for network construction")
        return defaultdict(list), []
    
    embeddings_matrix = torch.stack([concept_embeddings[token_id] for token_id in token_ids]).cpu().numpy()
    embeddings_normalized = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    
    index = faiss.IndexFlatIP(embeddings_normalized.shape[1])
    index.add(embeddings_normalized)
    
    local_means, local_stds = compute_local_density_correction(embeddings_normalized, index, token_ids, batch_size)
    
    id_to_index = {token_id: i for i, token_id in enumerate(token_ids)}
    
    concept_network = defaultdict(list)
    num_tokens = len(token_ids)
    
    for start_idx in tqdm(range(0, num_tokens, batch_size), desc="Building network"):
        end_idx = min(start_idx + batch_size, num_tokens)
        batch_ids = token_ids[start_idx:end_idx]
        batch_indices = [id_to_index[tid] for tid in batch_ids]
        
        batch_embeddings = embeddings_normalized[batch_indices]
        D, I = index.search(batch_embeddings, k=min(max_connections + 1, num_tokens))
        
        for i, token_id in enumerate(batch_ids):
            token_idx = id_to_index[token_id]
            tau_i = similarity_threshold + lambda_param * local_stds[token_idx]
            
            valid_candidates = []
            for j in range(len(I[i])):
                idx = I[i][j]
                candidate_id = token_ids[idx]
                if candidate_id != token_id:
                    similarity = D[i][j]
                    if similarity >= tau_i:
                        weight = np.exp(-(tau_i - similarity) / sigma)
                        valid_candidates.append((candidate_id, similarity, weight))
            
            valid_candidates.sort(key=lambda x: -x[1])
            selected_connections = valid_candidates[:max_connections]
            
            concept_network[token_id].extend([(tid, w) for tid, sim, w in selected_connections])
    
    elapsed_time = time.time() - start_time
    print(f"Concept network built in {elapsed_time:.2f} seconds")
    return concept_network, token_ids


class TokenDataset(Dataset):
    def __init__(self, token_ids, tokenizer):
        self.token_ids = token_ids
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.token_ids)
    
    def __getitem__(self, idx):
        return self.token_ids[idx]


def collate_tokens(batch, tokenizer, device):
    input_ids = [
        [tokenizer.bos_token_id, token_id, tokenizer.eos_token_id] 
        for token_id in batch
    ]
    return torch.tensor(input_ids, device=device)


def get_token_embeddings(pipeline, tokenizer, max_tokens=None, batch_size=128):
    start_time = time.time()
    text_encoder = pipeline.text_encoder
    text_encoder.eval()
    device = next(text_encoder.parameters()).device
    
    vocab_size = tokenizer.vocab_size
    max_tokens = vocab_size if max_tokens is None else min(max_tokens, vocab_size)
    token_ids = list(range(max_tokens))
    
    dataset = TokenDataset(token_ids, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=min(4, os.cpu_count())
    )
    
    concept_embeddings = {}
    
    print(f"Starting batch processing of token embeddings (Total: {max_tokens}, Batch size: {batch_size})")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing tokens"):
            input_ids = collate_tokens(batch, tokenizer, device)
            outputs = text_encoder(input_ids)
            embeddings = outputs.last_hidden_state[:, 1, :].cpu()
            
            for i, token_id in enumerate(batch):
                concept_embeddings[token_id] = embeddings[i]
    
    elapsed_time = time.time() - start_time
    print(f"Token embeddings retrieved in {elapsed_time:.2f} seconds")
    return concept_embeddings


def save_network_to_json(concept_network, token_ids, tokenizer, output_path="concept_network.json"):
    start_time = time.time()
    serializable_network = {}
    for token_id, connections in concept_network.items():
        token_text = tokenizer.decode([token_id])
        connected_texts = [(tokenizer.decode([conn_id]), float(weight)) for conn_id, weight in connections]
        serializable_network[token_text] = connected_texts
    
    data = {"concept_network": serializable_network}
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    elapsed_time = time.time() - start_time
    print(f"Concept network saved to {output_path} in {elapsed_time:.2f} seconds")


def main(args):
    overall_start_time = time.time()
    
    device = torch.device("cuda:0" if torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU device: {device}")
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
    
    seed_everything(args.seed)
    
    print(f"Loading model: {args.sd_ckpt}")
    model_load_start = time.time()
    pipe = DiffusionPipeline.from_pretrained(
        args.sd_ckpt, 
        safety_checker=None, 
        torch_dtype=torch.float16,
        load_safety_checker=False,
        text_encoder=None,
        tokenizer=None
    )
    pipe.tokenizer = CLIPTokenizer.from_pretrained(args.sd_ckpt, subfolder="tokenizer")
    pipe.text_encoder = CLIPTextModel.from_pretrained(
        args.sd_ckpt, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)
    
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.2f} seconds")
    
    print("Retrieving token embeddings...")
    concept_embeddings = get_token_embeddings(
        pipe, 
        pipe.tokenizer, 
        max_tokens=args.max_tokens,
        batch_size=args.batch_size
    )
    
    print("Filtering valid tokens...")
    filtered_embeddings = filter_valid_tokens(concept_embeddings, pipe.tokenizer)
    
    if not filtered_embeddings:
        print("No valid tokens filtered, cannot build network")
        return
    
    print("Building concept network...")
    concept_network, token_ids = build_concept_network(
        filtered_embeddings,
        pipe.tokenizer, 
        similarity_threshold=args.similarity_threshold,
        max_connections=args.max_connections,
        batch_size=args.batch_size,
        sigma=args.sigma,
        lambda_param=args.lambda_param
    )
    
    if concept_network:
        total_connections = sum(len(connections) for connections in concept_network.values())
        num_nodes = len(concept_network)
        average_degree = total_connections / num_nodes if num_nodes > 0 else 0
        print(f"Concept network statistics:")
        print(f"  Number of nodes: {num_nodes}")
        print(f"  Total connections: {total_connections}")
        print(f"  Average degree: {average_degree:.2f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "concept_network.json")
    save_network_to_json(concept_network, token_ids, pipe.tokenizer, json_path)
    
    overall_elapsed_time = time.time() - overall_start_time
    print(f"Concept network construction completed!")
    print(f"Total time elapsed: {overall_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct English concept network for Stable Diffusion vocabulary (based on GrOCE method)")
    parser.add_argument('--sd_ckpt', type=str, default="CompVis/stable-diffusion-v1-4", help='Stable Diffusion model path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default="concept_network_results", help='Output directory')
    parser.add_argument('--max_tokens', type=int, default=None, help='Maximum number of tokens to process')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch processing size')
    parser.add_argument('--similarity_threshold', type=float, default=0.3, help='Base similarity threshold τ0')
    parser.add_argument('--max_connections', type=int, default=100, help='Maximum connections per node')
    parser.add_argument('--sigma', type=float, default=0.1, help='Temperature parameter σ (controls weight decay rate)')
    parser.add_argument('--lambda_param', type=float, default=0.1, help='Local density correction parameter λ')
    
    args = parser.parse_args()
    main(args)