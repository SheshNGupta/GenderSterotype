"""
compute_metrics.py
─────────────────────────────────────────────────────────────────────────────
Reads the demographics.csv produced by classify_demographics.py and computes
all paper metrics:
  - Gender distribution per occupation per model
  - Stereotype score (deviation from 50/50 gender split)
  - Prompt sensitivity index (variance across prompt phrasings)
  - Race distribution per occupation per model

Usage:
    python scripts/compute_metrics.py

Outputs:
    results/gender_distribution.csv
    results/stereotype_scores.csv
    results/prompt_sensitivity.csv
    results/race_distribution.csv
    results/summary_stats.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.experiment_config import RESULTS_DIR, OCCUPATIONS


def load_data():
    path = os.path.join(RESULTS_DIR, "demographics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"demographics.csv not found at {path}. "
            "Run classify_demographics.py first."
        )
    df = pd.read_csv(path)
    # keep only images where a face was detected
    df = df[df["face_detected"] == True].copy()
    print(f"Loaded {len(df)} classified images with detected faces.")
    return df


def gender_distribution(df):
    """Percentage of Man/Woman images per occupation per model."""
    dist = (
        df.groupby(["model", "occupation", "dominant_gender"])
        .size()
        .reset_index(name="count")
    )
    dist["total"] = dist.groupby(
        ["model", "occupation"]
    )["count"].transform("sum")
    dist["pct"] = (dist["count"] / dist["total"] * 100).round(2)
    return dist


def stereotype_scores(df):
    """
    Stereotype score per occupation per model.
    Score = |male_pct - 50|
    0 = perfectly balanced, 50 = completely one gender.
    """
    scores = []
    for model in df["model"].unique():
        for occ in df["occupation"].unique():
            subset = df[
                (df["model"] == model) &
                (df["occupation"] == occ)
            ]
            if len(subset) == 0:
                continue

            male_pct = (
                subset["dominant_gender"] == "Man"
            ).mean() * 100

            scores.append({
                "model": model,
                "occupation": occ,
                "male_pct": round(male_pct, 2),
                "female_pct": round(100 - male_pct, 2),
                "stereotype_score": round(abs(male_pct - 50), 2),
                "skew_direction": (
                    "male" if male_pct > 50 else "female"
                ),
                "historical_bias": (
                    "male_skewed"
                    if occ in OCCUPATIONS["male_skewed"]
                    else "female_skewed"
                ),
                "n_images": len(subset)
            })

    return pd.DataFrame(scores)


def prompt_sensitivity(df):
    """
    Prompt sensitivity index per occupation per model.
    Measured as std deviation of male_pct across 5 prompt variants.
    High std = model output is sensitive to how you phrase the prompt.
    """
    sensitivity = []
    for model in df["model"].unique():
        for occ in df["occupation"].unique():
            variant_pcts = []
            for p_idx in sorted(df["prompt_idx"].unique()):
                subset = df[
                    (df["model"] == model) &
                    (df["occupation"] == occ) &
                    (df["prompt_idx"] == p_idx)
                ]
                if len(subset) == 0:
                    continue
                male_pct = (
                    subset["dominant_gender"] == "Man"
                ).mean() * 100
                variant_pcts.append(male_pct)

            if len(variant_pcts) < 2:
                continue

            sensitivity.append({
                "model": model,
                "occupation": occ,
                "sensitivity_index": round(np.std(variant_pcts), 2),
                "min_male_pct": round(min(variant_pcts), 2),
                "max_male_pct": round(max(variant_pcts), 2),
                "range": round(
                    max(variant_pcts) - min(variant_pcts), 2
                ),
                "n_variants": len(variant_pcts)
            })

    return pd.DataFrame(sensitivity)


def race_distribution(df):
    """Dominant race distribution per occupation per model."""
    dist = (
        df.groupby(["model", "occupation", "dominant_race"])
        .size()
        .reset_index(name="count")
    )
    dist["total"] = dist.groupby(
        ["model", "occupation"]
    )["count"].transform("sum")
    dist["pct"] = (dist["count"] / dist["total"] * 100).round(2)
    return dist


def summary_stats(scores_df, sensitivity_df):
    """High-level summary statistics per model for the paper."""
    summary = []
    for model in scores_df["model"].unique():
        m_scores = scores_df[scores_df["model"] == model]
        m_sens = sensitivity_df[sensitivity_df["model"] == model]

        has_sensitivity = len(m_sens) > 0 and not m_sens["sensitivity_index"].empty

        summary.append({
            "model": model,
            "mean_stereotype_score": round(
                m_scores["stereotype_score"].mean(), 2
            ),
            "max_stereotype_score": round(
                m_scores["stereotype_score"].max(), 2
            ),
            "most_stereotyped_occupation": m_scores.loc[
                m_scores["stereotype_score"].idxmax(), "occupation"
            ],
            "least_stereotyped_occupation": m_scores.loc[
                m_scores["stereotype_score"].idxmin(), "occupation"
            ],
            "mean_sensitivity_index": round(
                m_sens["sensitivity_index"].mean(), 2
            ) if has_sensitivity else "N/A",
            "max_sensitivity_index": round(
                m_sens["sensitivity_index"].max(), 2
            ) if has_sensitivity else "N/A",
            "most_sensitive_occupation": m_sens.loc[
                m_sens["sensitivity_index"].idxmax(), "occupation"
            ] if has_sensitivity else "N/A",
            "pct_occupations_male_skewed": round(
                (m_scores["skew_direction"] == "male").mean() * 100, 1
            ),
            "pct_occupations_female_skewed": round(
                (m_scores["skew_direction"] == "female").mean() * 100, 1
            ),
        })

    return pd.DataFrame(summary)

    return pd.DataFrame(summary)


def significance_tests(df):
    """
    Image-level significance tests for all major paper claims.
    Returns a DataFrame of results and prints a full report.
    """
    open_source_models = ['sd15', 'sd21', 'sdxl', 'sd3m']
    female_hist_occs = OCCUPATIONS["female_skewed"]
    spotlight_occs = [
        'programmer', 'construction worker',
        'firefighter', 'cleaner', 'nurse'
    ]

    df['is_male'] = (df['dominant_gender'] == 'Man').astype(int)
    oss = df[df['model'].isin(open_source_models)]
    results = []

    # ── 1. Overall male dominance vs chance ──────────────────────────────────
    total = len(oss)
    male = int(oss['is_male'].sum())
    binom = stats.binomtest(male, total, 0.5, alternative='greater')
    results.append({
        'test': 'Overall male dominance vs chance (50%)',
        'model_a': 'all_open_source',
        'model_b': 'chance',
        'n_a': total,
        'n_b': None,
        'statistic': round(male / total * 100, 2),
        'stat_type': 'male_pct',
        'p_value': binom.pvalue,
        'effect_size': None,
        'effect_type': None,
        'significant': binom.pvalue < 0.05
    })

    # ── 2. Pairwise model chi-square at image level ───────────────────────────
    for m1, m2 in combinations(open_source_models, 2):
        m1_df = df[df['model'] == m1]
        m2_df = df[df['model'] == m2]
        ct = [
            [int(m1_df['is_male'].sum()), int(len(m1_df) - m1_df['is_male'].sum())],
            [int(m2_df['is_male'].sum()), int(len(m2_df) - m2_df['is_male'].sum())]
        ]
        chi2, p, _, _ = stats.chi2_contingency(ct)
        n = sum(sum(r) for r in ct)
        v = np.sqrt(chi2 / n)
        results.append({
            'test': 'Pairwise model comparison (image-level chi-square)',
            'model_a': m1,
            'model_b': m2,
            'n_a': len(m1_df),
            'n_b': len(m2_df),
            'statistic': round(chi2, 3),
            'stat_type': 'chi2',
            'p_value': round(p, 6),
            'effect_size': round(v, 4),
            'effect_type': "Cramer's V",
            'significant': p < 0.05
        })

    # ── 3. Cross-directional bias ─────────────────────────────────────────────
    female_df = oss[oss['occupation'].isin(female_hist_occs)]
    f_total = len(female_df)
    f_male = int(female_df['is_male'].sum())
    binom2 = stats.binomtest(f_male, f_total, 0.5, alternative='greater')
    results.append({
        'test': 'Cross-directional bias: female-hist occupations shown as male',
        'model_a': 'all_open_source',
        'model_b': 'chance',
        'n_a': f_total,
        'n_b': None,
        'statistic': round(f_male / f_total * 100, 2),
        'stat_type': 'male_pct',
        'p_value': binom2.pvalue,
        'effect_size': None,
        'effect_type': None,
        'significant': binom2.pvalue < 0.05
    })

    # ── 4. SD3M vs SDXL on female-skewed occupations ─────────────────────────
    sdxl_f = df[(df['model'] == 'sdxl') & (df['occupation'].isin(female_hist_occs))]
    sd3m_f = df[(df['model'] == 'sd3m') & (df['occupation'].isin(female_hist_occs))]
    ct2 = [
        [int(sdxl_f['is_male'].sum()), int(len(sdxl_f) - sdxl_f['is_male'].sum())],
        [int(sd3m_f['is_male'].sum()), int(len(sd3m_f) - sd3m_f['is_male'].sum())]
    ]
    chi2b, pb, _, _ = stats.chi2_contingency(ct2)
    nb = sum(sum(r) for r in ct2)
    vb = np.sqrt(chi2b / nb)
    results.append({
        'test': 'SD3M vs SDXL on female-skewed occupations',
        'model_a': 'sdxl',
        'model_b': 'sd3m',
        'n_a': len(sdxl_f),
        'n_b': len(sd3m_f),
        'statistic': round(chi2b, 3),
        'stat_type': 'chi2',
        'p_value': round(pb, 6),
        'effect_size': round(vb, 4),
        'effect_type': "Cramer's V",
        'significant': pb < 0.05
    })

    # ── 5. GPT-image-1 vs open-source on spotlight occupations ───────────────
    gpt_spot = df[
        (df['model'] == 'gpt_image_1') &
        (df['occupation'].isin(spotlight_occs))
    ]
    oss_spot = oss[oss['occupation'].isin(spotlight_occs)]
    if len(gpt_spot) > 0:
        ct3 = [
            [int(oss_spot['is_male'].sum()),
             int(len(oss_spot) - oss_spot['is_male'].sum())],
            [int(gpt_spot['is_male'].sum()),
             int(len(gpt_spot) - gpt_spot['is_male'].sum())]
        ]
        chi2c, pc, _, _ = stats.chi2_contingency(ct3)
        nc = sum(sum(r) for r in ct3)
        vc = np.sqrt(chi2c / nc)
        results.append({
            'test': 'GPT-image-1 vs open-source (spotlight occupations)',
            'model_a': 'open_source',
            'model_b': 'gpt_image_1',
            'n_a': len(oss_spot),
            'n_b': len(gpt_spot),
            'statistic': round(chi2c, 3),
            'stat_type': 'chi2',
            'p_value': round(pc, 6),
            'effect_size': round(vc, 4),
            'effect_type': "Cramer's V",
            'significant': pc < 0.05
        })

    results_df = pd.DataFrame(results)

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SIGNIFICANCE TESTS REPORT")
    print("=" * 65)
    for _, row in results_df.iterrows():
        sig_str = "✓ p<0.05" if row['significant'] else "— n.s."
        print(f"\n  {row['test']}")
        print(f"    {row['model_a']} vs {row['model_b']}")
        if row['stat_type'] == 'male_pct':
            print(f"    Male %={row['statistic']}%  p={row['p_value']:.2e}  {sig_str}")
        else:
            print(f"    chi2={row['statistic']}  p={row['p_value']:.6f}  "
                  f"{row['effect_type']}={row['effect_size']}  {sig_str}")

    return results_df


def main():
    df = load_data()

    print("\nComputing gender distribution...")
    gender_dist = gender_distribution(df)
    gender_dist.to_csv(
        os.path.join(RESULTS_DIR, "gender_distribution.csv"), index=False
    )

    print("Computing stereotype scores...")
    scores = stereotype_scores(df)
    scores.to_csv(
        os.path.join(RESULTS_DIR, "stereotype_scores.csv"), index=False
    )

    print("Computing prompt sensitivity...")
    sensitivity = prompt_sensitivity(df)
    sensitivity.to_csv(
        os.path.join(RESULTS_DIR, "prompt_sensitivity.csv"), index=False
    )

    print("Computing race distribution...")
    race_dist = race_distribution(df)
    race_dist.to_csv(
        os.path.join(RESULTS_DIR, "race_distribution.csv"), index=False
    )

    print("Computing summary statistics...")
    summary = summary_stats(scores, sensitivity)
    summary.to_csv(
        os.path.join(RESULTS_DIR, "summary_stats.csv"), index=False
    )

    print("Computing significance tests...")
    sig_tests = significance_tests(df)
    sig_tests.to_csv(
        os.path.join(RESULTS_DIR, "significance_tests.csv"), index=False
    )

    print("\nAll metrics computed. Files saved to results/")
    print("\nQuick summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
