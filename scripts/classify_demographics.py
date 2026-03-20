"""
classify_demographics.py
─────────────────────────────────────────────────────────────────────────────
Runs DeepFace on all generated images and saves results to a CSV file.
Classifies apparent gender and race from each image.

Usage:
    python scripts/classify_demographics.py

Requirements:
    pip install deepface opencv-python pandas tqdm
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.experiment_config import IMAGES_DIR, RESULTS_DIR, MODELS

from deepface import DeepFace


def classify_all_images():
    results = []
    errors = []

    models_to_process = list(MODELS.keys())

    # also handle gpt_image_1 folder if it exists
    gpt_path = Path(IMAGES_DIR) / "gpt_image_1"
    if gpt_path.exists():
        models_to_process.append("gpt_image_1")

    # collect all image paths first for progress bar
    all_images = []
    for model_key in models_to_process:
        model_dir = Path(IMAGES_DIR) / model_key
        if not model_dir.exists():
            print(f"Skipping {model_key} — folder not found")
            continue
        def sort_key(p):
            # numeric sort for plain numbered filenames (1.png, 2.png, 10.png)
            try:
                return (p.parent, int(p.stem))
            except ValueError:
                return (p.parent, p.stem)

        for img_path in sorted(model_dir.rglob("*.png"), key=sort_key):
            all_images.append((model_key, img_path))

    print(f"\nFound {len(all_images)} images to classify\n")

    for model_key, img_path in tqdm(all_images, desc="Classifying"):
        parts = img_path.parts

        try:
            model_idx = parts.index(model_key)

            if model_key == "gpt_image_1":
                # Structure: images/gpt_image_1/occupation/img_NNN.png
                # No prompt subfolder
                occupation_slug = parts[model_idx + 1]
                prompt_idx = 1  # only one prompt used
            else:
                # Structure: images/model/occupation/prompt_N/img_NNN.png
                occupation_slug = parts[model_idx + 1]
                prompt_folder = parts[model_idx + 2]
                prompt_idx = int(prompt_folder.replace("prompt_", ""))

            occupation = occupation_slug.replace("_", " ")

        except Exception:
            print(f"  Could not parse path: {img_path}")
            continue

        try:
            analysis = DeepFace.analyze(
                img_path=str(img_path),
                actions=["gender", "race"],
                enforce_detection=False,
                silent=True
            )

            # DeepFace returns a list; take first face detected
            face = analysis[0]

            results.append({
                "model": model_key,
                "occupation": occupation,
                "prompt_idx": prompt_idx,
                "image_file": img_path.name,
                "image_path": str(img_path),
                "dominant_gender": face["dominant_gender"],
                "gender_man_pct": face["gender"].get("Man", 0),
                "gender_woman_pct": face["gender"].get("Woman", 0),
                "dominant_race": face["dominant_race"],
                "race_white_pct": face["race"].get("white", 0),
                "race_black_pct": face["race"].get("black", 0),
                "race_asian_pct": face["race"].get("asian", 0),
                "race_latino_pct": face["race"].get("latino hispanic", 0),
                "race_indian_pct": face["race"].get("indian", 0),
                "race_middle_eastern_pct": face["race"].get("middle eastern", 0),
                "face_detected": True
            })

        except Exception as e:
            errors.append({
                "model": model_key,
                "occupation": occupation,
                "image_path": str(img_path),
                "error": str(e)
            })
            results.append({
                "model": model_key,
                "occupation": occupation,
                "prompt_idx": prompt_idx,
                "image_file": img_path.name,
                "image_path": str(img_path),
                "dominant_gender": None,
                "gender_man_pct": None,
                "gender_woman_pct": None,
                "dominant_race": None,
                "race_white_pct": None,
                "race_black_pct": None,
                "race_asian_pct": None,
                "race_latino_pct": None,
                "race_indian_pct": None,
                "race_middle_eastern_pct": None,
                "face_detected": False
            })

    # save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = os.path.join(RESULTS_DIR, "demographics.csv")
    results_df.to_csv(results_path, index=False)

    errors_df = pd.DataFrame(errors)
    errors_path = os.path.join(RESULTS_DIR, "errors.csv")
    errors_df.to_csv(errors_path, index=False)

    # summary
    total = len(results)
    detected = results_df["face_detected"].sum()
    no_face = total - detected

    print(f"\nClassification complete.")
    print(f"  Total images processed : {total}")
    print(f"  Faces detected         : {detected} ({detected/total*100:.1f}%)")
    print(f"  No face / errors       : {no_face} ({no_face/total*100:.1f}%)")
    print(f"\nResults saved to: {results_path}")
    print(f"Errors saved to : {errors_path}")

    return results_df


if __name__ == "__main__":
    classify_all_images()
