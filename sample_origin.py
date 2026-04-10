import warnings
warnings.filterwarnings("ignore")
import os, sys, pdb
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import copy
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from src.template import template_dict
from src.utils import *



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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', type=str, default='', help='Root directory for saving generated images')
    parser.add_argument('--sd_ckpt', type=str, default="CompVis/stable-diffusion-v1-4", help='Path to Stable Diffusion model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale controlling text-image alignment')
    parser.add_argument('--total_timesteps', type=int, default=30, help='Total number of timesteps in sampling process')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate per prompt')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size, must be <= num_samples')
    parser.add_argument('--prompts', type=str, default=None, help='Custom prompt templates, separated by semicolons')
    
    parser.add_argument('--erase_type', type=str, default='', help='Concept type: instance, style, etc.')
    parser.add_argument('--target_concept', type=str, default='', help='Target concept for saving path')
    parser.add_argument('--contents', type=str, default='', help='List of contents, separated by commas')
    args = parser.parse_args()
    
    assert args.num_samples >= args.batch_size, "Batch size must be less than or equal to number of samples"

    pipe = DiffusionPipeline.from_pretrained(
        args.sd_ckpt, 
        safety_checker=None, 
        torch_dtype=torch.float16
    ).to('cuda')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    unet, tokenizer, text_encoder, vae = pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.vae
    
    uncond_embedding = get_textencoding(get_token('', tokenizer), text_encoder)
    
    seed_everything(args.seed, True)
    
    concept_list = [item.strip() for item in args.contents.split(',')]
    
    if args.prompts is None:
        prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]
    else:
        prompt_list = [[x.format(concept) for x in args.prompts.split(';')] for concept in concept_list]
    
    for i in range(int(args.num_samples // args.batch_size)):
        latent = torch.randn(args.batch_size, 4, 64, 64).to(pipe.device, dtype=pipe.dtype)
        
        for concept, prompts in zip(concept_list, prompt_list):
            for count, prompt in enumerate(prompts):
                embedding = get_textencoding(get_token(prompt, tokenizer), text_encoder)
                
                latents = diffusion(
                    unet=unet,
                    scheduler=pipe.scheduler,
                    latents=latent,
                    start_timesteps=0,
                    text_embeddings=torch.cat([uncond_embedding] * args.batch_size + [embedding] * args.batch_size, dim=0),
                    total_timesteps=args.total_timesteps,
                    guidance_scale=args.guidance_scale,
                    desc=f"{count} x {prompt} | original"
                )
                
                decoded_imgs = [
                    process_img(vae.decode(img.unsqueeze(0) / vae.config.scaling_factor, return_dict=False)[0]) 
                    for img in latents
                ]
                
                save_path = os.path.join(args.save_root, args.target_concept, concept, 'original')
                os.makedirs(save_path, exist_ok=True)
                
                for idx in range(len(decoded_imgs)):
                    save_filename = re.sub(r'[^\w\s]', '', prompt).replace(', ', '_') + f"_{int(idx + args.batch_size * i)}.png"
                    decoded_imgs[idx].save(os.path.join(save_path, save_filename))


if __name__ == '__main__':
    main()
