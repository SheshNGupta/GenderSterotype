"""
compute_significance.py
─────────────────────────────────────────────────────────────────────────────
Computes all statistical tests reported in the paper:
  - Model-level 95% confidence intervals
  - Occupation-level 95% confidence intervals
  - Binomial tests (overall male dominance, cross-directional bias)
  - Pairwise chi-square tests between models
  - Benjamini-Hochberg multiple comparisons correction across all 10 tests
  - Flags occupation-model cells within ±10pp of 50% (boundary cases)

Usage:
    python scripts/compute_significance.py

Outputs:
    results/significance_tests.csv   — all tests with raw and BH-adjusted p-values
    results/confidence_intervals.csv — CIs for all occupation-model cells
    results/boundary_cases.csv       — cells that straddle the 50% threshold
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest, chi2_contingency
from statsmodels.stats.multitest import multipletests

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

OPEN_SOURCE = ['sd15', 'sd21', 'sdxl', 'sd3m']
MODEL_LABELS = {'sd15': 'SD 1.5', 'sd21': 'SD 2.1', 'sdxl': 'SDXL', 'sd3m': 'SD 3M'}


def load_scores():
    path = os.path.join(RESULTS_DIR, 'stereotype_scores.csv')
    df = pd.read_csv(path)
    return df


# ── 1. Confidence intervals ───────────────────────────────────────────────────

def compute_cis(scores):
    rows = []
    for _, row in scores.iterrows():
        p = row['male_pct'] / 100
        n = int(row['n_images'])
        se = np.sqrt(p * (1 - p) / n) if 0 < p < 1 else 0
        ci = 1.96 * se * 100
        lower = round(row['male_pct'] - ci, 1)
        upper = round(row['male_pct'] + ci, 1)
        boundary = abs(row['male_pct'] - 50) < ci  # CI straddles 50%
        rows.append({
            'model': row['model'],
            'occupation': row['occupation'],
            'n_images': n,
            'male_pct': row['male_pct'],
            'ci_95_pp': round(ci, 1),
            'ci_lower': lower,
            'ci_upper': upper,
            'straddles_50pct': boundary,
            'historical_bias': row['historical_bias'],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'confidence_intervals.csv'), index=False)
    print(f"Saved confidence_intervals.csv ({len(df)} rows)")
    return df


def print_model_level_cis(scores):
    print("\n=== MODEL-LEVEL 95% CIs (n=2000 per model) ===")
    for m in OPEN_SOURCE:
        m_df = scores[scores['model'] == m]
        total = m_df['n_images'].sum()
        male = (m_df['male_pct'] * m_df['n_images'] / 100).sum()
        p = male / total
        se = np.sqrt(p * (1 - p) / total)
        ci = 1.96 * se * 100
        print(f"  {MODEL_LABELS[m]}: {p*100:.1f}% male  "
              f"95% CI [{p*100-ci:.1f}%, {p*100+ci:.1f}%]  ±{ci:.1f}pp")


def print_boundary_cases(ci_df):
    boundary = ci_df[ci_df['straddles_50pct'] == True]
    print(f"\n=== BOUNDARY CASES (CI straddles 50%, n={len(boundary)}) ===")
    print("These should be reported as indicative, not definitive:\n")
    for _, row in boundary.iterrows():
        print(f"  {MODEL_LABELS.get(row['model'],row['model']):<8} "
              f"{row['occupation']:<22} "
              f"{row['male_pct']:.0f}% ±{row['ci_95_pp']:.1f}pp  "
              f"[{row['ci_lower']:.1f}%, {row['ci_upper']:.1f}%]")
    boundary.to_csv(os.path.join(RESULTS_DIR, 'boundary_cases.csv'), index=False)
    print(f"\nSaved boundary_cases.csv")


# ── 2. All statistical tests ─────────────────────────────────────────────────

def run_all_tests(scores):
    tests = []

    # --- Test 1: Overall male dominance (binomial) ---
    oss = scores[scores['model'].isin(OPEN_SOURCE)]
    total = int(oss['n_images'].sum())
    male = int((oss['male_pct'] * oss['n_images'] / 100).sum())
    result = binomtest(male, total, 0.5, alternative='greater')
    tests.append({
        'test_id': 1,
        'description': 'Overall male dominance (binomial vs 50%)',
        'male_count': male,
        'total': total,
        'male_pct': round(male/total*100, 1),
        'statistic': None,
        'raw_p': result.pvalue,
        'cramers_v': None,
        'notes': f'n={total}, {male/total*100:.1f}% male'
    })

    # --- Test 2: Cross-directional bias (binomial) ---
    fem_occs = scores[
        (scores['model'].isin(OPEN_SOURCE)) &
        (scores['historical_bias'] == 'female_skewed')
    ]
    total_f = int(fem_occs['n_images'].sum())
    male_f = int((fem_occs['male_pct'] * fem_occs['n_images'] / 100).sum())
    result_f = binomtest(male_f, total_f, 0.5, alternative='greater')
    tests.append({
        'test_id': 2,
        'description': 'Cross-directional bias: female-coded occs majority male',
        'male_count': male_f,
        'total': total_f,
        'male_pct': round(male_f/total_f*100, 1),
        'statistic': None,
        'raw_p': result_f.pvalue,
        'cramers_v': None,
        'notes': f'n={total_f} images of female-coded occupations'
    })

    # --- Tests 3-8: Pairwise model comparisons (chi-square) ---
    pairs = [
        ('sd15', 'sd21'),
        ('sd15', 'sdxl'),
        ('sd15', 'sd3m'),
        ('sd21', 'sdxl'),
        ('sd21', 'sd3m'),
        ('sdxl', 'sd3m'),
    ]
    for test_id, (m1, m2) in enumerate(pairs, start=3):
        m1_df = scores[scores['model'] == m1]
        m2_df = scores[scores['model'] == m2]
        n1 = int(m1_df['n_images'].sum())
        n2 = int(m2_df['n_images'].sum())
        male1 = int((m1_df['male_pct'] * m1_df['n_images'] / 100).sum())
        male2 = int((m2_df['male_pct'] * m2_df['n_images'] / 100).sum())
        female1, female2 = n1 - male1, n2 - male2
        contingency = [[male1, female1], [male2, female2]]
        chi2, p, _, _ = chi2_contingency(contingency)
        n_total = n1 + n2
        v = np.sqrt(chi2 / n_total)
        tests.append({
            'test_id': test_id,
            'description': f'Pairwise: {MODEL_LABELS[m1]} vs {MODEL_LABELS[m2]}',
            'male_count': f'{male1}/{male2}',
            'total': f'{n1}/{n2}',
            'male_pct': f'{male1/n1*100:.1f}% / {male2/n2*100:.1f}%',
            'statistic': round(chi2, 2),
            'raw_p': p,
            'cramers_v': round(v, 3),
            'notes': f'chi-square test, V={v:.3f}'
        })

    # --- Test 9: SDXL vs SD3M on female-skewed occupations ---
    sdxl_f = scores[
        (scores['model'] == 'sdxl') &
        (scores['historical_bias'] == 'female_skewed')
    ]
    sd3m_f = scores[
        (scores['model'] == 'sd3m') &
        (scores['historical_bias'] == 'female_skewed')
    ]
    n_sdxl = int(sdxl_f['n_images'].sum())
    n_sd3m = int(sd3m_f['n_images'].sum())
    m_sdxl = int((sdxl_f['male_pct'] * sdxl_f['n_images'] / 100).sum())
    m_sd3m = int((sd3m_f['male_pct'] * sd3m_f['n_images'] / 100).sum())
    contingency = [[m_sdxl, n_sdxl-m_sdxl], [m_sd3m, n_sd3m-m_sd3m]]
    chi2, p, _, _ = chi2_contingency(contingency)
    v = np.sqrt(chi2 / (n_sdxl + n_sd3m))
    tests.append({
        'test_id': 9,
        'description': 'SDXL vs SD 3M on female-skewed occupations only',
        'male_count': f'{m_sdxl}/{m_sd3m}',
        'total': f'{n_sdxl}/{n_sd3m}',
        'male_pct': f'{m_sdxl/n_sdxl*100:.1f}% / {m_sd3m/n_sd3m*100:.1f}%',
        'statistic': round(chi2, 2),
        'raw_p': p,
        'cramers_v': round(v, 3),
        'notes': f'female-skewed occupations only, V={v:.3f}'
    })

    # --- Test 10: GPT-image-1 vs open-source (chi-square) ---
    gpt_df = scores[scores['model'] == 'gpt_image_1']
    # Use same 5 occupations for open-source comparison
    spotlight_occs = gpt_df['occupation'].tolist()
    oss_spot = scores[
        (scores['model'].isin(OPEN_SOURCE)) &
        (scores['occupation'].isin(spotlight_occs))
    ]
    n_gpt = int(gpt_df['n_images'].sum())
    n_oss = int(oss_spot['n_images'].sum())
    m_gpt = int((gpt_df['male_pct'] * gpt_df['n_images'] / 100).sum())
    m_oss = int((oss_spot['male_pct'] * oss_spot['n_images'] / 100).sum())
    contingency = [[m_gpt, n_gpt-m_gpt], [m_oss, n_oss-m_oss]]
    chi2, p, _, _ = chi2_contingency(contingency)
    v = np.sqrt(chi2 / (n_gpt + n_oss))
    tests.append({
        'test_id': 10,
        'description': 'GPT-image-1 vs open-source (5 spotlight occupations)',
        'male_count': f'{m_gpt}/{m_oss}',
        'total': f'{n_gpt}/{n_oss}',
        'male_pct': f'{m_gpt/n_gpt*100:.1f}% / {m_oss/n_oss*100:.1f}%',
        'statistic': round(chi2, 2),
        'raw_p': p,
        'cramers_v': round(v, 3),
        'notes': f'EXPLORATORY: n={n_gpt} GPT vs n={n_oss} open-source, V={v:.3f}'
    })

    return tests


# ── 3. Apply BH correction ────────────────────────────────────────────────────

def apply_bh_correction(tests):
    pvals = [t['raw_p'] for t in tests]
    reject, pvals_adj, _, _ = multipletests(pvals, method='fdr_bh')

    for i, t in enumerate(tests):
        t['bh_adjusted_p'] = pvals_adj[i]
        t['survives_bh'] = bool(reject[i])

    df = pd.DataFrame(tests)
    df.to_csv(os.path.join(RESULTS_DIR, 'significance_tests.csv'), index=False)
    return df


def print_significance_table(df):
    print("\n=== SIGNIFICANCE TESTS WITH BH CORRECTION ===\n")
    print(f"{'ID':<4} {'Description':<48} {'Raw p':>10} {'BH p':>12} {'V':>6} {'Survives?'}")
    print("-" * 90)
    for _, row in df.iterrows():
        v_str = f"{row['cramers_v']:.3f}" if row['cramers_v'] is not None and not pd.isna(row['cramers_v']) else "—"
        sig = "✓" if row['survives_bh'] else "✗"
        print(f"  {int(row['test_id']):<3} {row['description']:<47} "
              f"{row['raw_p']:>10.2e} {row['bh_adjusted_p']:>12.2e} "
              f"{v_str:>6} {sig:>9}")
    print()
    surviving = df[df['survives_bh']]['description'].tolist()
    dropped = df[~df['survives_bh']]['description'].tolist()
    print(f"Surviving BH correction: {len(surviving)}/10 tests")
    print(f"Dropped: {dropped}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading stereotype scores...")
    scores = load_scores()

    print("\n--- Confidence Intervals ---")
    ci_df = compute_cis(scores)
    print_model_level_cis(scores)
    print_boundary_cases(ci_df)

    print("\n--- Statistical Tests ---")
    tests = run_all_tests(scores)
    sig_df = apply_bh_correction(tests)
    print_significance_table(sig_df)

    print(f"\nSaved to results/significance_tests.csv and results/confidence_intervals.csv")
