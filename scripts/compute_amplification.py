"""
compute_amplification.py
─────────────────────────────────────────────────────────────────────────────
Computes bias amplification: how far each model's gender output deviates
from real-world workforce demographics (BLS 2023 data).

Amplification Gap = model_female_pct - bls_female_pct
  Negative = model shows fewer women than reality (male amplification)
  Positive = model shows more women than reality (female amplification)

Amplification Ratio = model_female_pct / bls_female_pct
  < 1.0 = model underrepresents women
  > 1.0 = model overrepresents women

Usage:
    python scripts/compute_amplification.py

Outputs:
    results/amplification.csv
    figures/fig8_amplification_gap.png
    figures/fig9_amplification_by_occupation.png
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
from configs.experiment_config import RESULTS_DIR, FIGURES_DIR

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── BLS 2023 female workforce percentages ────────────────────────────────────
# Sources: BLS Occupational Employment and Wage Statistics, CPS Table 11
BLS_FEMALE_PCT = {
    # historically female-skewed
    'nurse':             87.0,
    'preschool teacher': 95.0,
    'teacher':           74.0,
    'receptionist':      90.0,
    'cleaner':           46.0,
    'babysitter':        91.0,
    'librarian':         84.0,
    'social worker':     82.0,
    'florist':           65.0,
    'hair stylist':      92.0,
    # historically male-skewed
    'engineer':          16.0,
    'CEO':               29.0,
    'surgeon':           22.0,
    'pilot':              9.0,
    'construction worker': 4.0,
    'scientist':         48.0,
    'judge':             34.0,
    'firefighter':        8.0,
    'mechanic':           4.0,
    'programmer':        25.0,
}

MODEL_LABELS = {
    'sd15':  'SD 1.5',
    'sd21':  'SD 2.1',
    'sdxl':  'SDXL',
    'sd3m':  'SD 3M',
}
MODEL_COLORS = {
    'sd15':  '#4C72B0',
    'sd21':  '#DD8452',
    'sdxl':  '#55A868',
    'sd3m':  '#C44E52',
}
OPEN_SOURCE = ['sd15', 'sd21', 'sdxl', 'sd3m']


def compute_amplification():
    scores = pd.read_csv(os.path.join(RESULTS_DIR, 'stereotype_scores.csv'))
    scores = scores[scores['model'].isin(OPEN_SOURCE)]

    rows = []
    for _, row in scores.iterrows():
        occ = row['occupation']
        if occ not in BLS_FEMALE_PCT:
            continue
        bls_female = BLS_FEMALE_PCT[occ]
        bls_male   = 100 - bls_female
        model_female = row['female_pct']
        model_male   = row['male_pct']

        # gap: negative = model shows more men than reality
        gap = model_female - bls_female

        # ratio: how many times more/less female than reality
        # avoid division by zero for near-zero BLS values
        ratio = model_female / bls_female if bls_female > 0 else None

        rows.append({
            'model':           row['model'],
            'occupation':      occ,
            'bls_female_pct':  bls_female,
            'bls_male_pct':    bls_male,
            'model_female_pct': model_female,
            'model_male_pct':  model_male,
            'amplification_gap': round(gap, 2),       # pp deviation from reality
            'amplification_ratio': round(ratio, 3) if ratio else None,
            'historical_bias': row['historical_bias'],
            'n_images':        row['n_images'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'amplification.csv'), index=False)
    print(f"Saved amplification.csv ({len(df)} rows)")
    return df


def print_summary(df):
    print("\n=== AMPLIFICATION SUMMARY ===")
    print("\nMean amplification gap per model (negative = more male than reality):")
    for m in OPEN_SOURCE:
        m_df = df[df['model'] == m]
        mean_gap = m_df['amplification_gap'].mean()
        fem_gap = m_df[m_df['historical_bias']=='female_skewed']['amplification_gap'].mean()
        mal_gap = m_df[m_df['historical_bias']=='male_skewed']['amplification_gap'].mean()
        print(f"  {MODEL_LABELS[m]:<8} overall={mean_gap:+.1f}pp  "
              f"female-coded={fem_gap:+.1f}pp  male-coded={mal_gap:+.1f}pp")

    print("\nWorst amplification cases (most male vs reality):")
    worst = df.nsmallest(10, 'amplification_gap')[
        ['model','occupation','bls_female_pct','model_female_pct','amplification_gap']
    ]
    for _, r in worst.iterrows():
        print(f"  {MODEL_LABELS[r['model']]:<8} {r['occupation']:<22} "
              f"BLS={r['bls_female_pct']:.0f}%  model={r['model_female_pct']:.0f}%  "
              f"gap={r['amplification_gap']:+.0f}pp")

    print("\nOccupations where model is CLOSEST to reality (best cases):")
    best = df.groupby('occupation')['amplification_gap'].apply(
        lambda x: abs(x).mean()
    ).nsmallest(5)
    for occ, val in best.items():
        print(f"  {occ:<22} mean absolute gap = {val:.1f}pp")


def fig_amplification_gap(df):
    """
    Figure 8: Amplification gap heatmap — deviation from BLS reality per
    occupation and model. Red = model shows more men than reality.
    """
    models = [m for m in OPEN_SOURCE if m in df['model'].unique()]
    pivot = df.pivot_table(
        index='occupation', columns='model',
        values='amplification_gap'
    )[models]
    pivot.columns = [MODEL_LABELS[m] for m in models]

    # sort: most negative (worst male amplification) at top
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    # add BLS column for reference
    bls_col = pd.Series(
        {occ: 0.0 for occ in pivot.index},
        name='BLS Reality'
    )

    fig, ax = plt.subplots(figsize=(3 * len(models) + 3, 10))
    sns.heatmap(
        pivot,
        annot=True, fmt='+.0f',
        cmap='RdYlGn',
        center=0, vmin=-80, vmax=40,
        linewidths=0.5, ax=ax,
        cbar_kws={'label': 'Amplification Gap (pp vs BLS reality)\nNegative = model shows more men than reality'}
    )
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Occupation', fontsize=12)
    ax.set_title(
        'Amplification Gap: How Far Model Outputs Deviate From Real Workforce Demographics\n'
        '(Values in percentage points; negative = model generates more men than BLS data shows)',
        fontsize=12, pad=12
    )
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig8_amplification_gap.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def fig_amplification_by_occupation(df):
    """
    Figure 9: For each occupation, show BLS reality vs model outputs side by side.
    Shows exactly how much each model deviates from ground truth.
    """
    models = [m for m in OPEN_SOURCE if m in df['model'].unique()]
    occs = list(BLS_FEMALE_PCT.keys())

    fig, axes = plt.subplots(4, 5, figsize=(20, 16), sharey=False)
    axes = axes.flatten()

    for idx, occ in enumerate(occs):
        ax = axes[idx]
        occ_df = df[df['occupation'] == occ]
        bls = BLS_FEMALE_PCT[occ]

        # BLS bar
        ax.bar(0, bls, color='#2c7bb6', alpha=0.9, label='BLS Reality', width=0.6)

        # model bars
        for i, m in enumerate(models):
            m_row = occ_df[occ_df['model'] == m]
            if len(m_row) > 0:
                val = m_row['model_female_pct'].values[0]
                ax.bar(i + 1, val, color=MODEL_COLORS[m], alpha=0.85, width=0.6)

        ax.axhline(50, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_title(occ, fontsize=9, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.set_xticks(range(len(models) + 1))
        ax.set_xticklabels(['BLS'] + [MODEL_LABELS[m] for m in models],
                           rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Female %', fontsize=7)

    # legend
    handles = [
        mpatches.Patch(color='#2c7bb6', label='BLS Reality'),
    ] + [
        mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        for m in models
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        'Female Representation: BLS Workforce Reality vs Model Outputs\n'
        'Dashed line = 50% balance. Each panel shows one occupation.',
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig9_amplification_by_occupation.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    print("Computing bias amplification vs BLS workforce data...\n")
    df = compute_amplification()
    print_summary(df)
    print("\nGenerating figures...")
    fig_amplification_gap(df)
    fig_amplification_by_occupation(df)
    print("\nDone. Results in results/amplification.csv and figures/fig8_*, fig9_*")
