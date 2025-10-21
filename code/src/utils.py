import os, sys, pdb
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import random
import torch
import numpy as np
from PIL import Image


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_token(prompt, tokenizer=None):
    tokens = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    return tokens


def get_textencoding(input_tokens, text_encoder):
    text_encoding = text_encoder(input_tokens.to(text_encoder.device))[0]
    return text_encoding


def process_img(decoded_image):
    decoded_image = decoded_image.squeeze(0)
    decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
    decoded_image = (decoded_image * 255).byte()
    decoded_image = decoded_image.permute(1, 2, 0)

    decoded_image = decoded_image.cpu().numpy()
    return Image.fromarray(decoded_image)