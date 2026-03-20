"""
generate_dalle3.py
─────────────────────────────────────────────────────────────────────────────
Generates images using DALL-E 3 via the OpenAI API.
Run this after the open-source models are done and you have API credits.

Usage:
    export OPENAI_API_KEY=your_key_here
    python scripts/generate_dalle3.py

Cost estimate:
    20 occupations x 5 prompts x 20 images = 2,000 images
    DALL-E 3 standard quality 1024x1024 = $0.04 per image
    Total estimated cost = ~$80

    To reduce cost, set IMAGES_PER_PROMPT to 10 in this script:
    10 images = ~$40 total
"""

import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.experiment_config import (
    ALL_OCCUPATIONS, PROMPT_TEMPLATES, IMAGES_DIR
)

from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO

# set lower than main experiment if you want to save cost
DALLE_IMAGES_PER_PROMPT = 10

client = OpenAI()  # reads OPENAI_API_KEY from environment


def generate_dalle3():
    output_root = Path(IMAGES_DIR) / "dalle3"
    total = len(ALL_OCCUPATIONS) * len(PROMPT_TEMPLATES) * DALLE_IMAGES_PER_PROMPT
    print(f"\nGenerating {total} DALL-E 3 images")
    print(f"Estimated cost: ${total * 0.04:.2f}")
    print(f"Output directory: {output_root}\n")

    generated = 0
    skipped = 0
    errors = 0

    for occupation in ALL_OCCUPATIONS:
        occ_slug = occupation.replace(" ", "_")

        for p_idx, template in enumerate(PROMPT_TEMPLATES):
            prompt = template.format(occupation=occupation)
            out_dir = output_root / occ_slug / f"prompt_{p_idx + 1}"
            out_dir.mkdir(parents=True, exist_ok=True)

            for img_idx in range(DALLE_IMAGES_PER_PROMPT):
                out_path = out_dir / f"img_{img_idx + 1:03d}.png"

                if out_path.exists():
                    skipped += 1
                    continue

                try:
                    response = client.images.generate(
                        model="dall-e-3",
                        prompt=prompt,
                        size="1024x1024",
                        quality="standard",
                        n=1,
                    )
                    image_url = response.data[0].url

                    # download and save
                    img_response = requests.get(image_url, timeout=30)
                    img = Image.open(BytesIO(img_response.content))
                    img.save(out_path)
                    generated += 1

                    # rate limit: DALL-E 3 allows 5 images per minute
                    # on tier 1 accounts, be conservative
                    time.sleep(13)

                except Exception as e:
                    print(f"\n  ERROR: {occupation} / prompt_{p_idx+1} "
                          f"/ img_{img_idx+1}: {e}")
                    errors += 1
                    time.sleep(5)
                    continue

                if generated % 20 == 0:
                    print(f"  Progress: {generated} generated, "
                          f"{skipped} skipped, {errors} errors")

    print(f"\nDone.")
    print(f"  Generated : {generated}")
    print(f"  Skipped   : {skipped}")
    print(f"  Errors    : {errors}")
    print(f"  Approx cost: ${generated * 0.04:.2f}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    generate_dalle3()
