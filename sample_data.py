"""
sample_data.py
Generates a realistic, intentionally biased hiring dataset for demonstration.
Includes gender, caste, and region bias.
"""

import pandas as pd
import numpy as np


# Selection probabilities per group (intentionally biased)
GENDER_PROB   = {"Male": 0.70, "Female": 0.35}
CASTE_PROB    = {"General": 0.72, "OBC": 0.50, "SC": 0.28, "ST": 0.20}
REGION_PROB   = {"Urban": 0.68, "Semi-Urban": 0.48, "Rural": 0.25}

CASTE_COUNTS  = {"General": 80, "OBC": 70, "SC": 60, "ST": 40}   # total = 250
REGION_COUNTS = {"Urban": 100, "Semi-Urban": 90, "Rural": 60}     # total = 250


def get_sample_data() -> pd.DataFrame:
    """
    Returns a synthetic hiring dataset with gender, caste, and region bias.
    Columns: gender, caste, region, experience, education, selected
    """
    rng = np.random.default_rng(42)
    n = 250  # total records

    # ── Gender ────────────────────────────────────────────────────────────────
    genders = (["Male"] * 150) + (["Female"] * 100)
    rng.shuffle(genders)

    # ── Caste ─────────────────────────────────────────────────────────────────
    castes = []
    for caste, count in CASTE_COUNTS.items():
        castes.extend([caste] * count)
    rng.shuffle(castes)

    # ── Region ────────────────────────────────────────────────────────────────
    regions = []
    for region, count in REGION_COUNTS.items():
        regions.extend([region] * count)
    rng.shuffle(regions)

    # ── Features ──────────────────────────────────────────────────────────────
    experience = rng.integers(0, 20, n)
    education  = rng.integers(1, 5, n)

    # ── Selection: blend gender + caste + region bias ─────────────────────────
    selected = []
    for i in range(n):
        p_g = GENDER_PROB[genders[i]]
        p_c = CASTE_PROB[castes[i]]
        p_r = REGION_PROB[regions[i]]
        # Weighted average of the three biases
        p = (p_g * 0.4) + (p_c * 0.35) + (p_r * 0.25)
        selected.append(int(rng.random() < p))

    df = pd.DataFrame({
        "gender":     genders,
        "caste":      castes,
        "region":     regions,
        "experience": experience,
        "education":  education,
        "selected":   selected,
    })

    return df.sample(frac=1, random_state=42).reset_index(drop=True)
