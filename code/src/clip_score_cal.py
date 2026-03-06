import os, sys, re, pdb
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import torch
import torch_fidelity
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from torchvision import transforms
import torch.nn.functional as F
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


class Generate_Dataset(Dataset):
    def __init__(self, path, content, sub_root, include_original=False, original_path=None):
        super().__init__()

        root_path = os.path.join(path, content, sub_root)
        self.content = content

        self.images = [os.path.join(root_path, name) for name in os.listdir(root_path) 
                       if os.path.isfile(os.path.join(root_path, name))]

        self.original_images = []
        self.include_original = include_original
        
        if include_original and original_path:
   
            if os.path.exists(original_path):
         
                for img_path in self.images:
                    img_name = os.path.basename(img_path)
                    original_img_path = os.path.join(original_path, img_name)
                    if os.path.exists(original_img_path):
                        self.original_images.append(original_img_path)
                    else:
                       
                        if not self.original_images:
                            self.original_images = [os.path.join(original_path, 
                                 os.listdir(original_path)[0])] if os.listdir(original_path) else []
                        self.original_images.append(self.original_images[0])
        
        if content == 'coco':
 
            df = pd.read_csv("data/mscoco.csv")
            self.texts = [df.loc[df['image_id'].isin([int(os.path.basename(x).replace('COCO_val2014_', '').split('.')[0])]), 'text'].tolist()[0] for x in self.images]
        else:
         
            self.texts = [('_').join(x.split('/')[-1].split('_')[:-1]) for x in self.images]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        item = {
            'text': self.texts[idx], 
            'image': self.images[idx], 
            'content': self.content
        }
        if self.include_original and self.original_images:
            item['original_image'] = self.original_images[idx]
        return item


class CLIP_Score():
    def __init__(self, version='openai/clip-vit-large-patch14', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = CLIPModel.from_pretrained(version)
        self.processor = CLIPProcessor.from_pretrained(version)
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.device = device
        self.model = self.model.to(self.device)
    
    def __call__(self, dataloader):
        out_score = 0
        for item in dataloader:
            out_score_matrix = self.model_output(images=item['image'], texts=item['text'])
            out_score += out_score_matrix.mean().item() 
        return out_score / len(dataloader)
    
    def model_output(self, images, texts):
        torch.cuda.empty_cache()
   
        images_feats = self.processor(images=[Image.open(img) for img in images], return_tensors="pt").to(self.device)
        images_feats = self.model.get_image_features(** images_feats)

    
        texts_feats = self.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        texts_feats = self.model.get_text_features(**texts_feats)

   
        images_feats = images_feats / images_feats.norm(dim=1, p=2, keepdim=True)
        texts_feats = texts_feats / texts_feats.norm(dim=1, p=2, keepdim=True)
        score = (images_feats * texts_feats).sum(-1)
        return score


class PSNR_Calculator():
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
    
    def __call__(self, dataloader):
        psnr_scores = []
        for item in dataloader:
            if 'original_image' not in item:
                continue
                
            for gen_img_path, orig_img_path in zip(item['image'], item['original_image']):
         
                gen_img = transform(Image.open(gen_img_path).convert('RGB')).to(self.device)
                orig_img = transform(Image.open(orig_img_path).convert('RGB')).to(self.device)
                
    
                mse = F.mse_loss(gen_img, orig_img)
                if mse == 0:
                    psnr = torch.tensor(float('inf'))
                else:
                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                
                psnr_scores.append(psnr.item())
        
        return sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0


def find_root_paths(root_dir, sub_root):
    return sorted(
        list({os.path.abspath(os.path.join(dirpath, '..')) 
                for dirpath, dirnames, _ in os.walk(root_dir) if sub_root in dirnames})
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contents', type=str, required=True, help="如 'Mickey,Pikachu,Snoopy'")
    parser.add_argument('--root_path', type=str, required=True, help="logs 所在根目录，例如 'logs'")
    parser.add_argument('--sub_root', type=str, default='edit', help="生成图像所在子目录，默认 'edit'")
    parser.add_argument('--pretrained_path', type=str, required=True, help="pretrain 所在路径，例如 'pretrain/instance'")
    args = parser.parse_args()

    contents = [item.strip() for item in args.contents.split(',')]
    root_paths = find_root_paths(args.root_path, args.sub_root)

    CS_calculator = CLIP_Score()
    psnr_calculator = PSNR_Calculator()

    for root_path in root_paths:
    
        save_txt = os.path.join(root_path, 'record_metrics.txt')
        if not os.path.exists(save_txt): 
            with open(save_txt, 'a') as f:
                f.writelines('*************************** \n')
                f.writelines(f'Calculating the metrics for {root_path} \n')

        with open(save_txt, 'r') as f:  
            txt_content = f.read()
        for content in tqdm(contents):
            if content + ':' in txt_content: continue
            
        
            dataset_clip = Generate_Dataset(root_path, content, args.sub_root)
            dataloader_clip = DataLoader(dataset_clip, batch_size=10)
            CS = CS_calculator(dataloader_clip)
            
       
            if content != 'coco':
                fid_input2 = os.path.join(args.pretrained_path, content, 'original')
            else:
                fid_input2 = "data/pretrain/coco/coco/original" 
            
            FIDELITY = torch_fidelity.calculate_metrics(
                input1=os.path.join(root_path, content, args.sub_root), 
                input2=fid_input2, 
                cuda=True, 
                fid=True, 
                verbose=False,
            )
            

            original_path = os.path.join(args.pretrained_path, content, 'original') if content != 'coco' else "data/pretrain/coco/coco/original"
            dataset_metrics = Generate_Dataset(
                root_path, content, args.sub_root, 
                include_original=True, 
                original_path=original_path
            )
            dataloader_metrics = DataLoader(dataset_metrics, batch_size=10)
            
            psnr = psnr_calculator(dataloader_metrics)
            
       
            with open(save_txt, 'a') as f:
                f.writelines(
                    f"{content}: CS is {CS * 100:.2f}, "
                    f"FID is {abs(FIDELITY['frechet_inception_distance']):.2f}, "
                    f"PSNR is {psnr:.2f} \n"
                )