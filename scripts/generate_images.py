"""
generate_images.py
Generates images for all occupations and prompt templates.
Supports SD 1.5, SD 2.1, SDXL, and SD 3 Medium.

Usage:
    python scripts/generate_images.py --model sd15
    python scripts/generate_images.py --model sd21
    python scripts/generate_images.py --model sdxl
    python scripts/generate_images.py --model sd3m
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.experiment_config import (
    ALL_OCCUPATIONS, PROMPT_TEMPLATES, IMAGES_PER_PROMPT,
    MODELS, SD3_MODELS, HIGH_RES_MODELS, HIGH_RES_SIZE,
    INFERENCE_STEPS, GUIDANCE_SCALE, IMAGE_SIZE, SEEDS, IMAGES_DIR
)

import torch


def load_pipeline(model_key):
    model_id = MODELS[model_key]
    print(f"\nLoading model: {model_id}")
    print("This may take several minutes on first run (downloading weights)...\n")

    if model_key in SD3_MODELS:
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    elif model_key == "sdxl":
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    elif model_key == "sd21":
        # SD 2.1 base uses EulerDiscreteScheduler and different pipeline
        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    else:
        # SD 1.5
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe


def generate_for_model(model_key):
    pipe = load_pipeline(model_key)
    output_root = Path(IMAGES_DIR) / model_key
    size = HIGH_RES_SIZE if model_key in HIGH_RES_MODELS else IMAGE_SIZE

    total = len(ALL_OCCUPATIONS) * len(PROMPT_TEMPLATES) * IMAGES_PER_PROMPT
    print(f"Generating {total} images")
    print(f"Model      : {model_key} ({MODELS[model_key]})")
    print(f"Image size : {size}x{size}")
    print(f"Output dir : {output_root}")
    print(f"Resumable  : yes\n")

    generated = 0
    skipped = 0
    errors = 0

    for occupation in ALL_OCCUPATIONS:
        occ_slug = occupation.replace(" ", "_")
        for p_idx, template in enumerate(PROMPT_TEMPLATES):
            prompt = template.format(occupation=occupation)
            out_dir = output_root / occ_slug / f"prompt_{p_idx + 1}"
            out_dir.mkdir(parents=True, exist_ok=True)

            for img_idx in range(IMAGES_PER_PROMPT):
                out_path = out_dir / f"img_{img_idx + 1:03d}.png"

                if out_path.exists():
                    skipped += 1
                    continue

                generator = torch.Generator(
                    "cpu" if model_key in SD3_MODELS else "cuda"
                ).manual_seed(SEEDS[img_idx])

                try:
                    result = pipe(
                        prompt=prompt,
                        num_inference_steps=INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        generator=generator,
                        height=size,
                        width=size,
                    )
                    result.images[0].save(out_path)
                    generated += 1

                except torch.cuda.OutOfMemoryError:
                    print(f"\n  OUT OF MEMORY: {occupation} p{p_idx+1} i{img_idx+1}")
                    torch.cuda.empty_cache()
                    errors += 1
                    continue

                except Exception as e:
                    print(f"\n  ERROR: {occupation} p{p_idx+1} i{img_idx+1}: {e}")
                    errors += 1
                    continue

                done = generated + skipped
                if done % 100 == 0 and done > 0:
                    print(f"  {done/total*100:.0f}% | "
                          f"generated: {generated} | "
                          f"skipped: {skipped} | "
                          f"errors: {errors}")

    print(f"\nFinished {model_key}")
    print(f"  Generated : {generated}")
    print(f"  Skipped   : {skipped}")
    print(f"  Errors    : {errors}")
    print(f"  Saved to  : {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        required=True,
        help="sd15, sd21, sdxl, or sd3m"
    )
    args = parser.parse_args()
    generate_for_model(args.model)
