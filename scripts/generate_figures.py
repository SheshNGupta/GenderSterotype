"""
generate_figures.py
─────────────────────────────────────────────────────────────────────────────
Reads computed metrics CSVs and generates all publication-ready figures.
Includes SD 1.5, SD 2.1, SDXL, SD 3 Medium, and GPT-image-1.

Usage:
    python scripts/generate_figures.py

Outputs (all 300 DPI PNG):
    figures/fig1_stereotype_heatmap.png
    figures/fig2_gender_by_occupation.png
    figures/fig3_prompt_sensitivity.png
    figures/fig4_race_distribution.png
    figures/fig5_model_comparison.png
    figures/fig6_gpt_spotlight.png
    figures/fig7_cross_directional_bias.png
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.experiment_config import RESULTS_DIR, FIGURES_DIR, OCCUPATIONS

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Palette — all 5 models ────────────────────────────────────────────────────
MODEL_COLORS = {
    "sd15":        "#4C72B0",
    "sd21":        "#DD8452",
    "sdxl":        "#55A868",
    "sd3m":        "#C44E52",
    "gpt_image_1": "#8172B2",
}
MODEL_LABELS = {
    "sd15":        "SD 1.5",
    "sd21":        "SD 2.1",
    "sdxl":        "SDXL",
    "sd3m":        "SD 3M",
    "gpt_image_1": "GPT-image-1",
}
OPEN_SOURCE = ["sd15", "sd21", "sdxl", "sd3m"]
ALL_MODELS  = ["sd15", "sd21", "sdxl", "sd3m", "gpt_image_1"]

sns.set_theme(style="whitegrid", font_scale=1.1)


def load(filename):
    return pd.read_csv(os.path.join(RESULTS_DIR, filename))

def available(df, model_list):
    return [m for m in model_list if m in df["model"].unique()]


# ── Figure 1: Stereotype Score Heatmap (open-source models) ──────────────────
def fig1_stereotype_heatmap():
    scores = load("stereotype_scores.csv")
    models = available(scores, OPEN_SOURCE)

    pivot = scores[scores["model"].isin(models)].pivot_table(
        index="occupation", columns="model", values="stereotype_score"
    )[models]
    pivot.columns = [MODEL_LABELS[m] for m in models]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    # mark historical bias direction
    hist_male = set(OCCUPATIONS["male_skewed"])
    yticklabels = [
        f"{occ} ♂" if occ in hist_male else f"{occ} ♀"
        for occ in pivot.index
    ]

    fig, ax = plt.subplots(figsize=(3 * len(models) + 2, 10))
    sns.heatmap(
        pivot, annot=True, fmt=".1f",
        cmap="RdYlGn_r", vmin=0, vmax=50,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Stereotype Score (0=balanced, 50=fully skewed)"}
    )
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Occupation  (♂=historically male, ♀=historically female)",
                  fontsize=10)
    ax.set_title(
        "Stereotype Score by Occupation and Model\n"
        "(higher = stronger gender skew away from 50/50)",
        fontsize=13, pad=12
    )
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig1_stereotype_heatmap.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 2: Male % by Occupation (grouped bar) ─────────────────────────────
def fig2_gender_by_occupation():
    scores = load("stereotype_scores.csv")
    models = available(scores, OPEN_SOURCE)
    occs = OCCUPATIONS["male_skewed"] + OCCUPATIONS["female_skewed"]
    x = np.arange(len(occs))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(18, 6))
    for i, model in enumerate(models):
        m_scores = scores[scores["model"] == model].set_index("occupation")
        vals = [m_scores.loc[o, "male_pct"] if o in m_scores.index
                else np.nan for o in occs]
        ax.bar(
            x + i * width - (len(models) - 1) * width / 2,
            vals, width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model], alpha=0.85
        )

    ax.axhline(50, color="black", linestyle="--", linewidth=1,
               label="50% (balanced)")
    ax.axvline(9.5, color="gray", linestyle=":", linewidth=1)
    ax.text(4.5, 107, "Historically male-skewed",
            ha="center", fontsize=10, color="gray")
    ax.text(14.5, 107, "Historically female-skewed",
            ha="center", fontsize=10, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(occs, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Male % of generated images")
    ax.set_ylim(0, 115)
    ax.set_title(
        "Percentage of Male-Presenting Images by Occupation and Model",
        fontsize=13, pad=12
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig2_gender_by_occupation.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 3: Prompt Sensitivity (open-source only, GPT has 1 prompt) ────────
def fig3_prompt_sensitivity():
    sensitivity = load("prompt_sensitivity.csv")
    models = available(sensitivity, OPEN_SOURCE)

    fig, axes = plt.subplots(
        1, len(models), figsize=(5 * len(models), 7), sharey=True
    )
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        m_sens = sensitivity[sensitivity["model"] == model].sort_values(
            "sensitivity_index", ascending=True
        )
        ax.barh(
            m_sens["occupation"], m_sens["sensitivity_index"],
            color=MODEL_COLORS[model], alpha=0.85
        )
        mean_val = m_sens["sensitivity_index"].mean()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1,
                   label=f"Mean: {mean_val:.1f}")
        ax.set_title(MODEL_LABELS[model], fontsize=12)
        ax.set_xlabel("Sensitivity Index (std dev of male %)")
        ax.legend(fontsize=9)

    fig.suptitle(
        "Prompt Sensitivity Index by Occupation and Model\n"
        "(higher = output more sensitive to prompt phrasing)\n"
        "Note: GPT-image-1 excluded (single prompt design)",
        fontsize=12, y=1.03
    )
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig3_prompt_sensitivity.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 4: Race Distribution Stacked Bar ───────────────────────────────────
def fig4_race_distribution():
    race = load("race_distribution.csv")
    models = available(race, OPEN_SOURCE)

    race_cats = ["white", "black", "asian",
                 "latino hispanic", "indian", "middle eastern"]
    race_colors = ["#AEC6CF", "#FFB347", "#B39EB5",
                   "#FF6961", "#77DD77", "#CFCFC4"]

    fig, axes = plt.subplots(
        1, len(models), figsize=(5 * len(models), 8), sharey=True
    )
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        m_race = race[race["model"] == model]
        pivot = m_race.pivot_table(
            index="occupation", columns="dominant_race",
            values="pct", aggfunc="sum", fill_value=0
        )
        cols = [c for c in race_cats if c in pivot.columns]
        pivot[cols].plot(
            kind="barh", stacked=True, ax=ax,
            color=[race_colors[race_cats.index(c)] for c in cols],
            legend=(model == models[-1])
        )
        ax.set_title(MODEL_LABELS[model], fontsize=12)
        ax.set_xlabel("% of generated images")
        ax.set_xlim(0, 100)

    fig.suptitle(
        "Racial Composition of Generated Images by Occupation and Model",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig4_race_distribution.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 5: Model Comparison — mean stereotype score evolution ──────────────
def fig5_model_comparison():
    scores = load("stereotype_scores.csv")
    summary = load("summary_stats.csv")
    models = available(scores, OPEN_SOURCE)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: mean stereotype score per model (evolution line)
    ax = axes[0]
    means = [
        summary[summary["model"] == m]["mean_stereotype_score"].values[0]
        for m in models
    ]
    colors = [MODEL_COLORS[m] for m in models]
    bars = ax.bar([MODEL_LABELS[m] for m in models], means,
                  color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
    ax.plot(range(len(models)), means, "k--o", linewidth=1.5,
            markersize=6, zorder=5, label="Trend")
    ax.set_ylabel("Mean Stereotype Score")
    ax.set_title("Mean Stereotype Score by Model\n(lower = more balanced)")
    ax.set_ylim(0, 45)
    ax.legend()
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Right: % occupations showing male majority
    ax = axes[1]
    male_pcts = []
    for m in models:
        m_scores = scores[scores["model"] == m]
        pct = (m_scores["skew_direction"] == "male").mean() * 100
        male_pcts.append(pct)
    bars2 = ax.bar([MODEL_LABELS[m] for m in models], male_pcts,
                   color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
    ax.axhline(50, color="black", linestyle="--", linewidth=1,
               label="50% (chance)")
    ax.set_ylabel("% Occupations with Male Majority")
    ax.set_title("% of Occupations Showing Male Majority\nper Model")
    ax.set_ylim(0, 105)
    ax.legend()
    for bar, val in zip(bars2, male_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.suptitle("Model Evolution: Stereotype Bias Across SD Generations",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig5_model_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 6: GPT-image-1 Spotlight ──────────────────────────────────────────
def fig6_gpt_spotlight():
    scores = load("stereotype_scores.csv")
    spotlight_occs = [
        "programmer", "construction worker",
        "firefighter", "cleaner", "nurse"
    ]
    plot_models = available(scores, ALL_MODELS)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(spotlight_occs))
    width = 0.8 / len(plot_models)

    for i, model in enumerate(plot_models):
        m_scores = scores[scores["model"] == model].set_index("occupation")
        vals = []
        for occ in spotlight_occs:
            if occ in m_scores.index:
                vals.append(m_scores.loc[occ, "male_pct"])
            else:
                vals.append(np.nan)
        offset = i * width - (len(plot_models) - 1) * width / 2

        # GPT bar gets special styling
        edge = "black" if model == "gpt_image_1" else "white"
        lw = 2 if model == "gpt_image_1" else 1
        ax.bar(x + offset, vals, width,
               label=MODEL_LABELS[model],
               color=MODEL_COLORS[model], alpha=0.85,
               edgecolor=edge, linewidth=lw)

    ax.axhline(50, color="black", linestyle="--", linewidth=1,
               label="50% (balanced)")
    ax.set_xticks(x)
    ax.set_xticklabels(spotlight_occs, fontsize=12)
    ax.set_ylabel("Male % of generated images", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title(
        "GPT-image-1 Spotlight: Gender Skew on 5 Most Stereotyped Occupations\n"
        "Compared Against Open-Source Models",
        fontsize=13, pad=12
    )
    ax.legend(loc="upper right", fontsize=10)

    # annotate historical bias direction
    hist_female = set(OCCUPATIONS["female_skewed"])
    for xi, occ in enumerate(spotlight_occs):
        label = "♀ hist." if occ in hist_female else "♂ hist."
        ax.text(xi, -8, label, ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig6_gpt_spotlight.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 7: Cross-Directional Bias Summary ─────────────────────────────────
def fig7_cross_directional_bias():
    scores = load("stereotype_scores.csv")
    models = available(scores, OPEN_SOURCE)
    female_hist_occs = OCCUPATIONS["female_skewed"]

    # for each historically-female occupation, compute male% per model
    data = []
    for occ in female_hist_occs:
        row = {"occupation": occ}
        for m in models:
            m_row = scores[(scores["model"] == m) & (scores["occupation"] == occ)]
            row[MODEL_LABELS[m]] = m_row["male_pct"].values[0] if len(m_row) else np.nan
        data.append(row)
    df = pd.DataFrame(data).set_index("occupation")

    # sort by mean male% descending
    df = df.loc[df.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(df))
    width = 0.8 / len(models)

    for i, m in enumerate(models):
        label = MODEL_LABELS[m]
        if label in df.columns:
            ax.bar(
                x + i * width - (len(models) - 1) * width / 2,
                df[label], width,
                label=label, color=MODEL_COLORS[m], alpha=0.85
            )

    ax.axhline(50, color="black", linestyle="--", linewidth=1.5,
               label="50% (balanced)")
    ax.fill_between([-0.5, len(df) - 0.5], 50, 100,
                    alpha=0.05, color="red",
                    label="Male majority zone (reversed bias)")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Male % of generated images", fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title(
        "Cross-Directional Bias: Historically Female Occupations\n"
        "Values above 50% indicate reversed gender stereotyping",
        fontsize=13, pad=12
    )
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig7_cross_directional_bias.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating figures...\n")
    fig1_stereotype_heatmap()
    fig2_gender_by_occupation()
    fig3_prompt_sensitivity()
    fig4_race_distribution()
    fig5_model_comparison()
    fig6_gpt_spotlight()
    fig7_cross_directional_bias()
    print("\nAll 7 figures saved to figures/")
