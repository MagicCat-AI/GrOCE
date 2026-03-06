import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import re
import copy
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from collections import defaultdict, deque
import faiss
from torch.utils.data import Dataset, DataLoader
import random


from src.template import template_dict
from src.utils import get_textencoding, get_token, seed_everything, process_img

class ConceptNetwork:
    def __init__(self, network_path, decay_factor=0.8, tokenizer=None, text_encoder=None,
                 tau0=0.3, sigma=0.1, lambda_param=0.1, batch_size=32):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.decay_factor = decay_factor
        self.network_path = network_path
        self.tau0 = tau0
        self.sigma = sigma
        self.lambda_param = lambda_param
        self.batch_size = batch_size
        
        if os.path.exists(network_path):
            with open(network_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.network = data['concept_network']
            self.concepts = set(self.network.keys())
            print(f"Concept network loaded successfully, containing {len(self.concepts)} concepts, path decay factor α={decay_factor}")
        else:
            self.network = defaultdict(list)
            self.concepts = set()
            print(f"No existing concept network found, initializing a new network, path decay factor α={decay_factor}")


    def get_concept_embedding(self, concept):
        if not self.tokenizer or not self.text_encoder:
            raise ValueError("Tokenizer and text_encoder are required during initialization to obtain concept embeddings")
            
        with torch.no_grad():
            tokens = get_token(concept, self.tokenizer)
            embedding = self.text_encoder(tokens.to(self.text_encoder.device))[0]
        
        if embedding.ndim > 2:
            embedding = embedding.squeeze(0)
        return torch.mean(embedding, dim=0).cpu().numpy()


    def get_batch_concept_embeddings(self, concepts):
        if not self.tokenizer or not self.text_encoder:
            raise ValueError("Tokenizer and text_encoder are required during initialization to obtain concept embeddings")
            
        embeddings = []
        for i in range(0, len(concepts), self.batch_size):
            batch_concepts = concepts[i:i+self.batch_size]
            
            inputs = self.tokenizer(
                batch_concepts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.text_encoder.device)
            
            with torch.no_grad():
                batch_embeddings = self.text_encoder(** inputs)[0]
            
            batch_embeddings = torch.mean(batch_embeddings, dim=1).cpu().numpy()
            embeddings.extend(batch_embeddings)
        
        return embeddings


    def get_shortest_path(self, start, end):
        if start not in self.concepts or end not in self.concepts:
            return float('inf')
        
        visited = set()
        queue = deque([(start, 0)])
        visited.add(start)
        
        while queue:
            node, distance = queue.popleft()
            if node == end:
                return distance
            if node not in self.network:
                continue
            for neighbor, _ in self.network[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        
        return float('inf')


    def get_neighborhood_stats(self, concept):
        if concept not in self.network or len(self.network[concept]) == 0:
            return 0.0, 0.0, 0
        
        concept_emb = self.get_concept_embedding(concept)
        concept_emb_norm = concept_emb / np.linalg.norm(concept_emb)
        
        neighbors = [n for n, _ in self.network[concept]]
        if not neighbors:
            return 0.0, 0.0, 0
            
        neighbor_embs = self.get_batch_concept_embeddings(neighbors)
        
        neighbor_embs_norm = np.array([e / np.linalg.norm(e) for e in neighbor_embs])
        similarities = np.dot(neighbor_embs_norm, concept_emb_norm)
        
        return np.mean(similarities), np.std(similarities), len(neighbor_embs)


    def compute_dynamic_threshold(self, concept):
        mu_i, std_i, neighbor_count = self.get_neighborhood_stats(concept)
        
        if neighbor_count < 2:
            return self.tau0
            
        return self.tau0 + self.lambda_param * std_i


    def insert_concept(self, concept, top_k=10):
        if concept in self.concepts:
            print(f"Concept '{concept}' is already in the network, no need to insert")
            return False
        
        print(f"Inserting new concept '{concept}' into the network...")
        try:
            new_embedding = self.get_concept_embedding(concept)
            new_embedding_norm = new_embedding.reshape(1, -1) / np.linalg.norm(new_embedding)
        except Exception as e:
            print(f"Failed to obtain embedding for concept '{concept}': {e}")
            return False

        if not hasattr(self, 'faiss_index') or self.faiss_index is None:
            print("Initializing FAISS index...")
            self.concept_embeddings = {}
            
            if self.concepts:
                concepts_list = list(self.concepts)
                embeddings_list = self.get_batch_concept_embeddings(concepts_list)
                
                normalized_embeddings = []
                for c, emb in zip(concepts_list, embeddings_list):
                    self.concept_embeddings[c] = emb
                    emb_normalized = emb.reshape(1, -1) / np.linalg.norm(emb)
                    normalized_embeddings.append(emb_normalized)
                
                if normalized_embeddings:
                    embeddings_array = np.vstack(normalized_embeddings)
                    self.faiss_index = faiss.IndexFlatIP(embeddings_array.shape[1])
                    self.faiss_index.add(embeddings_array)
                    self.concept_list = concepts_list
            else:
                self.network[concept] = []
                self.concepts.add(concept)
                self.concept_embeddings[concept] = new_embedding
                self.faiss_index = None
                print(f"Concept '{concept}' has been inserted into the empty network")
                return True

        search_k = top_k * 3
        D, I = self.faiss_index.search(new_embedding_norm, k=min(search_k, len(self.concept_list)))
        
        similarities = []
        for i, idx in enumerate(I[0]):
            existing_concept = self.concept_list[idx]
            sim = D[0][i]
            
            try:
                dynamic_tau = self.compute_dynamic_threshold(existing_concept)
                if sim > dynamic_tau:
                    weight = float(np.exp(-(dynamic_tau - sim) / self.sigma))
                    similarities.append((existing_concept, weight, float(sim), float(dynamic_tau)))
            except Exception as e:
                print(f"Error processing concept '{existing_concept}': {e}")
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:top_k]
        
        if top_similar:
            avg_tau = np.mean([item[3] for item in top_similar])
            print(f"Average dynamic threshold when inserting new concept: {avg_tau:.4f}")
            print(f"Similarity range: [{min(item[2] for item in top_similar):.4f}, {max(item[2] for item in top_similar):.4f}]")
        
        self.network[concept] = [(c, s) for c, s, _, _ in top_similar]
        
        for c, s, _, _ in top_similar:
            self.network[c].append((concept, s))
            if len(self.network[c]) > 100:
                self.network[c].sort(key=lambda x: x[1], reverse=True)
                self.network[c] = self.network[c][:100]
        
        self.concepts.add(concept)
        self.concept_embeddings[concept] = new_embedding
        self.faiss_index.add(new_embedding_norm)
        self.concept_list.append(concept)
        
        print(f"Concept '{concept}' has been inserted into the network, connected to {len(top_similar)} similar concepts")
        return True


    def _convert_numpy_types(self, data):
        if isinstance(data, dict):
            return {k: self._convert_numpy_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_types(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.generic):
            return data.item()
        else:
            return data


    def save_network(self, output_path=None):
        if output_path is None:
            output_path = self.network_path
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        network_data = self._convert_numpy_types(dict(self.network))
        data = {"concept_network": network_data}
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Concept network has been saved to {output_path}")


    def get_neighbors_with_scores(self, concept, step=2, t=3):
        if concept not in self.concepts:
            return {}
        
        reachable_nodes = {concept}
        queue = deque([(concept, 0)])
        
        while queue:
            current, depth = queue.popleft()
            if depth >= step:
                continue
            if current not in self.network:
                continue
            for neighbor, _ in self.network[current]:
                if neighbor not in reachable_nodes:
                    reachable_nodes.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        subgraph_nodes = list(reachable_nodes)
        n = len(subgraph_nodes)
        node_to_idx = {node: i for i, node in enumerate(subgraph_nodes)}
        target_idx = node_to_idx[concept]
        
        laplacian = self.get_simplified_laplacian(subgraph_nodes)
        if laplacian is None:
            return {}
        
        initial_scores = torch.zeros(n, device=laplacian.device)
        initial_scores[target_idx] = 1.0
        
        diffused_scores = self.spectral_diffusion(
            laplacian, 
            initial_scores,
            diffusion_steps=None,
            t=t
        )
        
        neighbor_scores = {}
        for i, node in enumerate(subgraph_nodes):
            if node != concept:
                neighbor_scores[node] = diffused_scores[i].item()
        
        return neighbor_scores


    def get_simplified_laplacian(self, concepts):
        n = len(concepts)
        if n == 0:
            return None
        concept_to_idx = {c: i for i, c in enumerate(concepts)}
        
        adj_matrix = np.zeros((n, n), dtype=np.float32)
        
        for i, concept in enumerate(concepts):
            if concept in self.network:
                for neighbor, weight in self.network[concept]:
                    if neighbor in concept_to_idx:
                        j = concept_to_idx[neighbor]
                        adj_matrix[i, j] = weight
                        adj_matrix[j, i] = weight
        
        degrees = np.sum(adj_matrix, axis=1)
        degrees = np.maximum(degrees, 1e-6)
        
        inv_sqrt_degrees = 1.0 / np.sqrt(degrees)
        inv_sqrt_degree_matrix = np.diag(inv_sqrt_degrees)
        
        laplacian = np.eye(n) - inv_sqrt_degree_matrix @ adj_matrix @ inv_sqrt_degree_matrix
        
        return torch.tensor(laplacian, dtype=torch.float32)


    def spectral_diffusion(self, laplacian, initial_scores, diffusion_steps, t=3):
        n = laplacian.shape[0]
        if n == 0:
            return initial_scores
        
        v = initial_scores.unsqueeze(1)
        exp_matrix = torch.matrix_exp(-t * laplacian)
        diffused_scores = exp_matrix @ v
        diffused_scores = diffused_scores.squeeze(1)
        diffused_scores = (diffused_scores - diffused_scores.min()) / (diffused_scores.max() - diffused_scores.min() + 1e-8)
        
        return diffused_scores


    def get_cluster(self, target, n_step=2, top_k=5, diffusion_steps=3):
        if target not in self.concepts:
            print(f"Target concept '{target}' is not in the network, attempting to insert...")
            if self.insert_concept(target):
                self.save_network()
            else:
                return [target]
        
        neighbor_scores = self.get_neighbors_with_scores(target, step=n_step)
        if not neighbor_scores:
            return [target]
        
        all_concepts = [target] + list(neighbor_scores.keys())
        concept_to_idx = {c: i for i, c in enumerate(all_concepts)}
        n = len(all_concepts)
        
        initial_scores = torch.zeros(n)
        initial_scores[0] = 1.0
        
        laplacian = self.get_simplified_laplacian(all_concepts)
        if laplacian is not None and n > 1:
            diffused_scores = self.spectral_diffusion(
                laplacian, 
                initial_scores,
                diffusion_steps=diffusion_steps,
                t=3
            )
        else:
            diffused_scores = initial_scores
        
        cluster_scores = [(all_concepts[i], diffused_scores[i].item()) 
                        for i in range(1, n)]
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        
        neighbor_top = [c for c, _ in cluster_scores[:top_k-1] if c != target]
        final_cluster = [target] + neighbor_top
        
        return final_cluster


def expand_concepts(concept_list, concept_net, n_step=2, top_k=5, diffusion_steps=3):
    expanded = []
    for concept in concept_list:
        cluster = concept_net.get_cluster(
            concept, 
            n_step=n_step, 
            top_k=top_k,
            diffusion_steps=diffusion_steps
        )
        expanded.extend(cluster)
    return list(set(expanded))
    
    
def diffusion(unet, scheduler, latents, text_embeddings, total_timesteps, start_timesteps=0, guidance_scale=7.5, desc=None, **kwargs):
    scheduler.set_timesteps(total_timesteps)
    for timestep in tqdm(scheduler.timesteps[start_timesteps: total_timesteps], desc=desc):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

        noise_pred = unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=text_embeddings,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return latents

def compute_similarity(embedding, target_embeddings_list, expanded_targets, concept_net, 
                       tokenizer, input_ids, args, verbose=False):
    if embedding.dim() == 3:
        embedding = embedding.squeeze(0)
    
    special_tokens = set(tokenizer.all_special_tokens + ['</w>'])
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    is_special = [token in special_tokens for token in tokens]
    non_special_mask = ~torch.tensor(is_special, device=embedding.device)
    valid_indices = torch.where(non_special_mask)[0]
    
    if valid_indices.numel() == 0:
        return embedding.unsqueeze(0), tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    valid_embedding = embedding[valid_indices]
    modified_embedding = embedding.clone()
    
    target_embeddings = []
    for emb in target_embeddings_list:
        if emb.dim() == 3:
            emb = emb.squeeze(0)
        avg_emb = emb.mean(dim=0)
        avg_emb = avg_emb / (avg_emb.norm() + 1e-8)
        target_embeddings.append(avg_emb)
    target_embeddings = torch.stack(target_embeddings).to(embedding.device)
    
    processed_tokens = tokens.copy()
    processed_info = []
    
    for i, idx in enumerate(valid_indices):
        p_i = valid_embedding[i]
        token_text = tokens[idx]
        concept = token_text.strip().lower().replace('</w>', '')
        
        sum_term = torch.zeros_like(p_i)
        token_processed = False
        target_contributions = []
        
        for target_idx, target in enumerate(expanded_targets):
            target_emb = target_embeddings[target_idx]
            
            distance = concept_net.get_shortest_path(concept, target)
            if distance == float('inf'):
                continue
            
            alpha = np.exp(-distance * args.decay_factor)
            
            dot_product = torch.dot(p_i, target_emb)
            
            term = alpha * dot_product * target_emb
            term_norm = torch.norm(term).item()
            
            if term_norm > args.projection_threshold:
                sum_term += term
                token_processed = True
                target_contributions.append(f"{target}:{term_norm:.3f}")
        
        if token_processed:
            modified_embedding[idx] = p_i - sum_term
            processed_info.append((token_text, ", ".join(target_contributions)))
            processed_tokens[idx] = f"[CUT:{len(target_contributions)}]"
        
    processed_input_ids = tokenizer.convert_tokens_to_ids(processed_tokens)
    processed_prompt = tokenizer.decode(processed_input_ids, skip_special_tokens=True)
    
    if verbose:
        print(f"Processed words and their contributing targets:")
        for token, targets in processed_info:
            print(f"  {token}: {targets}")
        print(f"Processed prompt: {processed_prompt}")
    
    return modified_embedding.unsqueeze(0), processed_prompt
    


@torch.no_grad()
def main():
    print('Starting sampling')
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', type=str, default='')
    parser.add_argument('--sd_ckpt', type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, default='edit')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--total_timesteps', type=int, default=30)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--prompts', type=str, default=None)
    parser.add_argument('--erase_type', type=str, default='')
    parser.add_argument('--target_concepts', type=str, default='')
    parser.add_argument('--contents', type=str, default='')
    parser.add_argument('--network_path', type=str, default="./concept_network_results/concept_network.json")
    parser.add_argument('--n_step', type=int, default=2)
    parser.add_argument('--top_k', type=int, default=8)
    parser.add_argument('--decay_factor', type=float, default=0.8)
    parser.add_argument('--insert_topk', type=int, default=100)
    parser.add_argument('--similarity_threshold', type=float, default=0.3)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--lambda_param', type=float, default=0.1)
    parser.add_argument('--diffusion_steps', type=int, default=3)
    parser.add_argument('--verbose', action='store_true', default=True)

    parser.add_argument('--projection_threshold', type=float, default=10.0)
    parser.add_argument('--embedding_batch_size', type=int, default=32)
    
    args = parser.parse_args()

    assert args.num_samples >= args.batch_size
   
    bs = args.batch_size
    mode_list = args.mode.replace(' ', '').split(',')


    concept_list, concept_list_tmp = [], [item.strip() for item in args.contents.split(',')]
    if 'edit' in mode_list:
        for concept in concept_list_tmp:
            check_path = os.path.join(args.save_root, args.target_concepts.replace(', ', '_'), concept, 'edit')
            os.makedirs(check_path, exist_ok=True)
            if len(os.listdir(check_path)) != len(template_dict[args.erase_type]) * 10:
                concept_list.append(concept)
    else:
        concept_list = concept_list_tmp
    if len(concept_list) == 0: sys.exit()


    print(f"Loading model: {args.sd_ckpt}")
    pipe = DiffusionPipeline.from_pretrained(
        args.sd_ckpt, 
        safety_checker=None, 
        torch_dtype=torch.float16,
        load_safety_checker=False
    ).to('cuda')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    unet, tokenizer, text_encoder, vae = pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.vae


    concept_net = ConceptNetwork(
        args.network_path, 
        decay_factor=args.decay_factor,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        tau0=args.similarity_threshold,
        sigma=args.sigma,
        lambda_param=args.lambda_param,
        batch_size=args.embedding_batch_size
    )
    
    target_concepts = [c.strip().lower().replace(' ', '') for c in args.target_concepts.split(',') if c.strip()]
    if not target_concepts:
        sys.exit("Please specify at least one target concept via --target_concepts")


    print("Starting to expand target concepts...")
    expanded_targets = expand_concepts(
        target_concepts, 
        concept_net, 
        n_step=args.n_step, 
        top_k=args.top_k,
        diffusion_steps=args.diffusion_steps
    )
    print(f"expanded_targets:{expanded_targets}")

    uncond_embedding = get_textencoding(get_token('', tokenizer), text_encoder)
    
    target_embeddings_list = []
    if expanded_targets:
        target_embeddings_np = concept_net.get_batch_concept_embeddings(expanded_targets)
        for emb_np in target_embeddings_np:
            emb = torch.tensor(emb_np).unsqueeze(0).unsqueeze(0).to(text_encoder.device)
            target_embeddings_list.append(emb)
    
    seed_everything(args.seed, True)
    if args.prompts is None:
        prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]
    else:
        prompt_list = [[x.format(concept) for x in args.prompts.split(';')] for concept in concept_list]
    
    all_concepts = list(set(expanded_targets + concept_list))
    
    for i in range(int(args.num_samples // bs)):
        latent = torch.randn(bs, 4, 64, 64).to(pipe.device, dtype=pipe.dtype)
        for concept, prompts in zip(concept_list, prompt_list):
            for count, prompt in enumerate(prompts):
                print(f"\nProcessing prompt: '{prompt}'")


                save_images = {}
                tokens = get_token(prompt, tokenizer)
                embedding = text_encoder(tokens.to(text_encoder.device))[0]
                
                if 'edit' in mode_list:
                    modified_embedding, processed_prompt = compute_similarity(
                        embedding, 
                        target_embeddings_list,
                        expanded_targets,
                        concept_net,
                        tokenizer,
                        tokens,
                        args,
                        verbose=args.verbose
                    )
                    new_prompt_text = processed_prompt

                    new_tokens = get_token(new_prompt_text, tokenizer)
                    with torch.no_grad():
                        new_original_embedding = text_encoder(new_tokens.to(text_encoder.device))[0]


                    processed_prompt_erase = processed_prompt
                    processed_tokens = get_token(processed_prompt_erase, tokenizer)
                    processed_embedding = text_encoder(processed_tokens.to(text_encoder.device))[0]
                    edit_embedding = new_original_embedding
                    print("Using embedding generated from processed prompt")

                    
                    save_images['edit'] = diffusion(
                        unet=unet, 
                        scheduler=pipe.scheduler,
                        latents=latent, 
                        start_timesteps=0, 
                        text_embeddings=torch.cat([uncond_embedding] * bs + [edit_embedding] * bs, dim=0), 
                        total_timesteps=args.total_timesteps, 
                        guidance_scale=args.guidance_scale, 
                        desc=f"{count} x {prompt} | edit"
                    )
                
                save_path = os.path.join(args.save_root, args.target_concepts.replace(', ', '_'), concept)
                for mode in mode_list: 
                    os.makedirs(os.path.join(save_path, mode), exist_ok=True)
                if len(mode_list) > 1:
                    os.makedirs(os.path.join(save_path, 'combine'), exist_ok=True)


                decoded_imgs = {
                    name: [process_img(vae.decode(img.unsqueeze(0) / vae.config.scaling_factor, return_dict=False)[0]) 
                           for img in img_list]
                    for name, img_list in save_images.items()
                }


                def combine_images_horizontally(images):
                    widths, heights = zip(*(img.size for img in images))
                    new_img = Image.new('RGB', (sum(widths), max(heights)))
                    for i, img in enumerate(images):
                        new_img.paste(img, (sum(widths[:i]), 0))
                    print("Sampling completed")
                    return new_img
                
                for idx in range(len(decoded_imgs[mode_list[0]])):
                    save_filename = re.sub(r'[^\w\s]', '', prompt).replace(' ', '_') + f"_{int(idx + bs * i)}.png"
                    images_to_combine = []
                    for mode in mode_list: 
                        img_path = os.path.join(save_path, mode, save_filename)
                        decoded_imgs[mode][idx].save(img_path)
                        images_to_combine.append(decoded_imgs[mode][idx])
                    if len(mode_list) > 1:
                        combined_img = combine_images_horizontally(images_to_combine)
                        combined_img.save(os.path.join(save_path, 'combine', save_filename))


if __name__ == '__main__':
    main()