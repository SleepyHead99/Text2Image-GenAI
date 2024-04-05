import gradio as gr

from diffusers import DiffusionPipeline, LCMScheduler
import torch

import base64
from io import BytesIO
import os
import gc
import warnings

# Only used when MULTI_GPU set to True
from helper import UNetDataParallel

# SDXL code: https://github.com/huggingface/diffusers/pull/3859

# Process environment variables
# Use `segmind/SSD-1B` (distilled SDXL) for faster generation.
use_ssd = os.getenv("USE_SSD", "false").lower() == "true"
if use_ssd:
    model_key_base = "segmind/SSD-1B"
    model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"
    lcm_lora_id = "latent-consistency/lcm-lora-ssd-1b"
else:
    model_key_base = "stabilityai/stable-diffusion-xl-base-1.0"
    model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"
    lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

# Use LCM LoRA (enabled by default)
if "ENABLE_LCM" not in os.environ:
    warnings.warn("`ENABLE_LCM` environment variable is not set. LCM LoRA will be loaded by default and refiner will be disabled by default. You can set it to `False` to turn off LCM LoRA.")

enable_lcm = os.getenv("ENABLE_LCM", "true").lower() == "true"
# Use refiner (disabled by default if LCM is enabled)
enable_refiner = os.getenv("ENABLE_REFINER", "false" if enable_lcm or use_ssd else "true").lower() == "true"
# Output images before the refiner and after the refiner
output_images_before_refiner = os.getenv("OUTPUT_IMAGES_BEFORE_REFINER", "false").lower() == "true"

offload_base = os.getenv("OFFLOAD_BASE", "false").lower() == "true"
offload_refiner = os.getenv("OFFLOAD_REFINER", "true").lower() == "true"

# Generate how many images by default
default_num_images = int(os.getenv("DEFAULT_NUM_IMAGES", "4"))
if default_num_images < 1:
    default_num_images = 1

print("Loading model", model_key_base)
pipe = DiffusionPipeline.from_pretrained(model_key_base, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

if enable_lcm:
    pipe.load_lora_weights(lcm_lora_id)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

multi_gpu = os.getenv("MULTI_GPU", "false").lower() == "true"

if multi_gpu:
    pipe.unet = UNetDataParallel(pipe.unet)
    pipe.unet.config, pipe.unet.dtype, pipe.unet.add_embedding = pipe.unet.module.config, pipe.unet.module.dtype, pipe.unet.module.add_embedding
    pipe.to("cuda")
else:
    if offload_base:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

if enable_refiner:
    print("Loading model", model_key_refiner)
    pipe_refiner = DiffusionPipeline.from_pretrained(model_key_refiner, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    if multi_gpu:
        pipe_refiner.unet = UNetDataParallel(pipe_refiner.unet)
        pipe_refiner.unet.config, pipe_refiner.unet.dtype, pipe_refiner.unet.add_embedding = pipe_refiner.unet.module.config, pipe_refiner.unet.module.dtype, pipe_refiner.unet.module.add_embedding
        pipe_refiner.to("cuda")
    else:
        if offload_refiner:
            pipe_refiner.enable_model_cpu_offload()
        else:
            pipe_refiner.to("cuda")

is_gpu_busy = False
def infer(prompt, negative, scale, samples=4, steps=50, refiner_strength=0.3, seed=-1):
    prompt, negative = [prompt] * samples, [negative] * samples

    g = torch.Generator(device="cuda")
    if seed != -1:
        g.manual_seed(seed)
    else:
        g.seed()

    images_b64_list = []

    if not enable_refiner or output_images_before_refiner:
        images = pipe(prompt=prompt, negative_prompt=negative, guidance_scale=scale, num_inference_steps=steps, generator=g).images
    else:
        images = pipe(prompt=prompt, negative_prompt=negative, guidance_scale=scale, num_inference_steps=steps, output_type="latent", generator=g).images

    gc.collect()
    torch.cuda.empty_cache()

    if enable_refiner:
        if output_images_before_refiner:
            for image in images:
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                image_b64 = (f"data:image/jpeg;base64,{img_str}")
                images_b64_list.append(image_b64)

        images = pipe_refiner(prompt=prompt, negative_prompt=negative, image=images, num_inference_steps=steps, strength=refiner_strength, generator=g).images

        gc.collect()
        torch.cuda.empty_cache()

    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        image_b64 = (f"data:image/jpeg;base64,{img_str}")
        images_b64_list.append(image_b64)
    
    return images_b64_list

examples = [
    [
        'A high tech solarpunk utopia in the Amazon rainforest',
        'low quality',
        1 if enable_lcm else 9
    ],
    [
        'A pikachu fine dining with a view to the Eiffel Tower',
        'low quality',
        1 if enable_lcm else 9
    ],
    [
        'A mecha robot in a favela in expressionist style',
        'low quality, 3d, photorealistic',
        1 if enable_lcm else 9
    ],
    [
        'an insect robot preparing a delicious meal',
        'low quality, illustration',
        1 if enable_lcm else 9
    ],
    [
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
        'low quality, ugly',
        1 if enable_lcm else 9
    ],
]

gr.Interface(
    fn=infer,
    inputs=["text", "text", "number"],
    outputs="image",
    title="TEXT TO IMAGE GENERATION MODEL",
    description="This Model is using Stable Diffusion XL which is latest text-to-image model from StabilityAI. Source code of this Model is on <a href='https://github.com/aryanchaudh
