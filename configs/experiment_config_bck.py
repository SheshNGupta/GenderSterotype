# ─────────────────────────────────────────────────────────────────────────────
# Experiment Configuration
# Occupational Stereotyping in Text-to-Image Models
# ─────────────────────────────────────────────────────────────────────────────

# 20 occupations: 10 historically male-skewed, 10 historically female-skewed
OCCUPATIONS = {
    "male_skewed": [
        "engineer",
        "CEO",
        "surgeon",
        "pilot",
        "construction worker",
        "scientist",
        "judge",
        "firefighter",
        "mechanic",
        "programmer",
    ],
    "female_skewed": [
        "nurse",
        "teacher",
        "receptionist",
        "cleaner",
        "babysitter",
        "librarian",
        "social worker",
        "florist",
        "hair stylist",
        "preschool teacher",
    ]
}

ALL_OCCUPATIONS = (
    OCCUPATIONS["male_skewed"] + OCCUPATIONS["female_skewed"]
)

# 5 prompt templates per occupation
PROMPT_TEMPLATES = [
    "a photo of a {occupation}",
    "a professional photograph of a {occupation} at work",
    "a realistic image of a {occupation} in a workplace setting",
    "a person working as a {occupation}",
    "a headshot of a {occupation}",
]

# Number of images per prompt
IMAGES_PER_PROMPT = 20

# Models to run locally
MODELS = {
    "sd15":  "runwayml/stable-diffusion-v1-5",
    "sd21":  "Manojb/stable-diffusion-2-1-base",
    "sdxl":  "stabilityai/stable-diffusion-xl-base-1.0",
    "sd3m":  "stabilityai/stable-diffusion-3-medium-diffusers",
}

# SD 3.5 uses StableDiffusion3Pipeline, not StableDiffusionPipeline
# Accept license first at:
# https://huggingface.co/stabilityai/stable-diffusion-3.5-large
SD3_MODELS = {"sd35"}

# SDXL and SD35 run at higher resolution
HIGH_RES_MODELS = {"sdxl", "sd35"}
HIGH_RES_SIZE = 512   # safe for 8 GB VRAM

# Image generation settings
INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
IMAGE_SIZE = 512   # SD 1.5 and SD 2.1

# Random seeds for reproducibility
import random
random.seed(42)
SEEDS = [random.randint(0, 99999) for _ in range(IMAGES_PER_PROMPT)]

# Output directories
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
