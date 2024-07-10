import pdb
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import os
import torch
import numpy as np
from PIL import Image
import cv2

import random
import time
import pdb

from pipelines_ootd.pipeline_gardiff import tryonpipeline
from diffusers import UNet2DConditionModel
from diffusers import UniPCMultistepScheduler
from diffusers import AutoencoderKL

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTokenizer

from einops import rearrange
from safetensors.torch import load_file

from torchvision import transforms
from diffusers.image_processor import VaeImageProcessor


UNET_PATH = "checkpoin/unet/" 
ADAPTER_PATH = 'checkpoin/adapter'
MODEL_PATH = 'pretrained_model_name_or_path'

class Embedding_Adapter(nn.Module):
    def __init__(self, input_nc=38, output_nc=4, norm_layer=nn.InstanceNorm2d, chkpt=None):
        super(Embedding_Adapter, self).__init__()

        self.save_method_name = "adapter"

        self.pool =  nn.MaxPool2d(2)
        self.vae2clip = nn.Linear(1024, 768)


    def forward(self, clip, vae):
        
        vae = self.pool(vae) # 1 4 80 64 --> 1 4 40 32
        vae = rearrange(vae, 'b c h w -> b c (h w)') # 1 4 20 16 --> 1 4 1280

        # clip = self.vae2clip(clip) # 1 4 768

        # Concatenate
        concat = torch.cat((clip, vae), 1)

        return concat

class GardiffHD:

    def __init__(self, gpu_id):
        self.gpu_id = 'cuda:' + str(gpu_id)

        self.vae = AutoencoderKL.from_pretrained(
            'pretrained_model_name_or_path',
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        unet_vton = UNet2DConditionModel.from_pretrained(
            UNET_PATH,
            subfolder="unet",
            torch_dtype=torch.float32,
            use_safetensors=True,
        )

        self.adapter = Embedding_Adapter(input_nc=1280, output_nc=1280)
        ckpt = load_file(ADAPTER_PATH)
        self.adapter.load_state_dict(ckpt, strict=True)

        self.pipe = tryonpipeline.from_pretrained(
            MODEL_PATH,
            unet_vton=unet_vton,
            vae=self.vae,
            torch_dtype=torch.float32,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.gpu_id)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        self.auto_processor = AutoProcessor.from_pretrained('pretrained_model_name_or_path')
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained('pretrained_model_name_or_path').to(self.gpu_id)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_PATH,
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_PATH,
            subfolder="text_encoder",
        ).to(self.gpu_id)

        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])


    def tokenize_captions(self, captions, max_length):
        inputs = self.tokenizer(
            captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


    def __call__(self,
                model_type='hd',
                category='upperbody',
                image_garm=None,
                image_vton=None,
                warp_cloth=None,
                mask=None,
                warp_image=None,
                image_ori=None,
                num_samples=1,
                num_steps=20,
                image_scale=1.0,
                seed=-1,
    ):
        if seed == -1:
            random.seed(time.time())
            seed = random.randint(0, 2147483647)
        print('Initial seed: ' + str(seed))
        generator = torch.manual_seed(seed)

        with torch.no_grad():
            prompt_image = self.auto_processor(images=image_garm, return_tensors="pt").to(self.gpu_id)
            prompt_image = self.image_encoder(prompt_image.data['pixel_values']).image_embeds
            prompt_image = prompt_image.unsqueeze(1)
            
            warp_cloth = self.transform(warp_cloth)
            warp_cloth = self.image_processor.preprocess(warp_cloth.to(self.gpu_id))

            warp_cloth_latent = self.vae.encode(warp_cloth).latent_dist.sample(generator=generator)
            warp_cloth_latent = self.vae.config.scaling_factor * warp_cloth_latent
            prompt_image = self.adapter(prompt_image,warp_cloth_latent)
            
            prompt_embeds = self.text_encoder(self.tokenize_captions([""], 6).to(self.gpu_id))[0]
            prompt_embeds[:, 1:] = prompt_image[:]


            images = self.pipe(prompt_embeds=prompt_embeds,
                        image_vton=image_vton, 
                        warp_cloth=warp_cloth,
                        mask=mask,
                        image_ori=image_ori,
                        warp_image=warp_image,
                        num_inference_steps=num_steps,
                        image_guidance_scale=image_scale,
                        num_images_per_prompt=num_samples,
                        generator=generator,
            ).images

        return images