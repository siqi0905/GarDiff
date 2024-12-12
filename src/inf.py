import argparse
import logging
import os
import numpy as np
import cv2

import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.checkpoint
import torchvision
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import UNet2DConditionModel, DDIMScheduler,AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
import torch.nn as nn
from einops import rearrange
from PIL import Image, ImageOps, ImageFilter
from PIL.ImageOps import exif_transpose

from dataset.vitonhd import VitonHDDataset
from utils.set_seeds import set_seed
from pipelines.tryon_pipe import StableDiffusionTryOnePipeline

def tokenize_captions(pipe, captions, max_length):
        inputs = pipe.tokenizer(
            captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
class Embedding_Adapter(nn.Module):
    def __init__(self, input_nc=38, output_nc=4, norm_layer=nn.InstanceNorm2d, chkpt=None):
        super(Embedding_Adapter, self).__init__()

        self.save_method_name = "adapter"

        self.pool =  nn.MaxPool2d(2)


    def forward(self, clip, vae):
        
        vae = self.pool(vae) # 1 4 80 64 --> 1 4 40 32
        vae = rearrange(vae, 'b c h w -> b c (h w)') # 1 4 20 16 --> 1 4 1280

        # clip = self.vae2clip(clip) # 1 4 768

        # Concatenate
        concat = torch.cat((clip, vae), 1)

        return concat

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"

torch.multiprocessing.set_sharing_strategy('file_system')
def color_correction_func( fg_img, bg_img, mask):
        fg = fg_img.astype(np.float32)
        bg = bg_img.copy().astype(np.float32)
        w = mask[:, :, None].astype(np.float32) / 255.0
        #w = w.round()
        y = fg * w + bg * (1 - w)
        return y.clip(0, 255).astype(np.uint8)

def prepare_fuse_mask(mask, w, h):
        up_mask = up255(resample_image(mask, w, h), t=127)
        open_mask = morphological_open(up_mask)
        return open_mask

def up255(x, t=0):
    y = np.zeros_like(x).astype(np.uint8)
    y[x > t] = 255
    return y

def resample_image(im, width, height):
    # im = Image.fromarray(im)
    im = im.resize((width, height), resample=Image.LANCZOS)
    return np.array(im)

def max33(x):
    x = Image.fromarray(x)
    x = x.filter(ImageFilter.MaxFilter(3))
    return np.array(x)

def morphological_open(x):
    x_int32 = np.zeros_like(x).astype(np.int32)
    x_int32[x > 127] = 256
    for _ in range(32):
        maxed = max33(x_int32) - 8
        x_int32 = np.maximum(maxed, x_int32)
    return x_int32.clip(0, 255).astype(np.uint8)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images
def pt_to_pil(images):
    """
    Convert a torch image to a PIL image.
    """
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().detach().float().permute(0,2,3,1).numpy()
    images = numpy_to_pil(images)
    return images
def parse_args():
    parser = argparse.ArgumentParser(description="VTO training script.")
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
    parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/data/pretrained_model/sd1.5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")

    parser.add_argument(
        "--test_batch_size", type=int, default=1, help="Batch size (per device) for the testing dataloader."
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='fp16',
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--num_workers_test", type=int, default=8,
                        help="Number of workers to use in the test dataloaders.")
    parser.add_argument("--test_order", type=str, default="unpaired", choices=["unpaired", "paired"])
    parser.add_argument("--cloth_input_type", type=str, choices=["warped", "none"], default='warped',
                        help="cloth input type. If 'warped' use the warped cloth, if none do not use the cloth as input of the unet")

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    inversion_adapter = None

    # Check if the dataset dataroot is provided
    if args.dataset == "vitonhd" and args.vitonhd_dataroot is None:
        raise ValueError("VitonHD dataroot must be provided")

    # Setup accelerator.
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        kwargs_handlers=[kwargs],
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenizer and models
    val_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    val_scheduler.set_timesteps(50, device=accelerator.device)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

    vision_encoder = CLIPVisionModelWithProjection.from_pretrained("openai-clip-vit-large-patch14/")
    processor = AutoProcessor.from_pretrained("openai-clip-vit-large-patch14/")

    inversion_adapter = Embedding_Adapter(input_nc=1280, output_nc=1280)
    inversion_adapter.load_state_dict(torch.load('gardiff/inversion_adapter.pth',map_location='cpu'))

    config = UNet2DConditionModel.load_config("gardiff/unet_config.json", subfolder="unet")
    unet = UNet2DConditionModel.from_config(config)

    unet.load_state_dict(torch.load('gardiff/unet.pth',map_location='cpu'))

    vision_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    test_dataset = VitonHDDataset(
        dataroot_path=args.vitonhd_dataroot,
        phase='test',
        order=args.test_order,
        size=(512, 384),
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers_test,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    test_dataloader = accelerator.prepare(test_dataloader)

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vision_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    inversion_adapter.to(accelerator.device, dtype=weight_dtype)

    with torch.no_grad():
                val_pipe = StableDiffusionTryOnePipeline(
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet,
                    tokenizer=tokenizer,
                    scheduler=val_scheduler,
                ).to(accelerator.device)

                # Extract the images
                with torch.cuda.amp.autocast():
                    # Create output directory
                    save_path = os.path.join(args.output_dir)
                    os.makedirs(save_path, exist_ok=True)

                    # Set seed
                    seed =1234
                    generator = torch.Generator("cuda").manual_seed(seed)
                    num_samples = 1

                    # Generate images
                    for idx, batch in enumerate(tqdm(test_dataloader)):
                        model_img = batch.get("image")
                        mask_img = batch.get("inpaint_mask")
                        if mask_img is not None:
                            mask_img = mask_img.type(torch.float32)
        
                        warped_cloth = batch.get('warped_cloth')
                        category = batch.get("category")

                        cloth_latents = val_pipe.vae.encode(batch['warped_cloth'].to(torch.float16)).latent_dist.sample()
                        cloth_latents = cloth_latents * val_pipe.vae.config.scaling_factor
                        
                        input_image = torchvision.transforms.functional.resize((batch["cloth"] + 1) / 2, (224, 224),
                                                                                                antialias=True).clamp(0, 1)
                        prompt_image = processor(images=input_image, return_tensors="pt").to(model_img.device)
                        prompt_image = vision_encoder(prompt_image.data['pixel_values']).image_embeds
                        prompt_image = prompt_image.unsqueeze(1)

                        prompt_image = inversion_adapter(prompt_image,cloth_latents)
                        prompt_embeds = val_pipe.text_encoder(tokenize_captions(val_pipe,batch.get('captions'), 6).to(model_img.device))[0]
                        prompt_embeds[:, 1:] = prompt_image[:]
                        

                        # Generate images
                        generated_images = val_pipe(
                            image=model_img,
                            mask_image=mask_img,
                            warped_cloth=warped_cloth,
                            prompt_embeds=prompt_embeds,
                            height=512,
                            width=384,
                            guidance_scale=0,
                            num_images_per_prompt=num_samples,
                            generator=generator,
                            cloth_input_type="warped",
                            cloth_cond_rate=1,
                            num_inference_steps=50
                        ).images
                        
                        # Save images
                        for gen_image, cat, name,mask_img,gt in zip(generated_images, category, batch["im_name"],batch["inpaint_mask"],batch["image"]):

                            if not os.path.exists(os.path.join(save_path, cat)):
                                os.makedirs(os.path.join(save_path, cat))

                            gen_image.save(
                                os.path.join(save_path, cat, name))
                            
                            pred_img = cv2.imread(os.path.join(save_path, cat, name))
                            mask = Image.open(os.path.join(args.vitonhd_dataroot, 'test', 'agnostic-densepose-v3',name.replace('.png','_mask.jpg'))).resize((384,512))
                            mask = prepare_fuse_mask(mask, w=384, h=512)
                            gt = cv2.imread(os.path.join(args.vitonhd_dataroot, 'test', 'image',name.replace('.png','.jpg')))
                            gt = cv2.resize(gt,(384,512))
                            image = color_correction_func( fg_img = pred_img, bg_img = gt, mask =mask)
                            cv2.imwrite(os.path.join(save_path, cat, name), image)

   
if __name__ == "__main__":
    main()
