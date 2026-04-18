"""
bias_detector.py
Core bias detection logic: selection rates, fairness score, verdict, explanation.
Supports multi-axis analysis: gender, caste, region.
"""

import pandas as pd

FAIRNESS_THRESHOLD = 80.0

# Human-readable labels for each bias axis
AXIS_LABELS = {
    "gender": "Gender",
    "caste":  "Caste",
    "region": "Region",
}

AXIS_ALERTS = {
    "gender": "🚨 Gender-based bias detected — women are being systematically disadvantaged.",
    "caste":  "🚨 Caste-based bias detected — lower-caste groups face significant discrimination.",
    "region": "🚨 Regional discrimination detected — rural/semi-urban candidates are disadvantaged.",
}


def calculate_selection_rates(
    df: pd.DataFrame, group_col: str = "gender", target_col: str = "selected"
) -> dict:
    """
    Compute selection rate per group.
    selection_rate = count(target==1) / count(total) for each group.
    Returns dict mapping group label -> float rate in [0, 1].
    """
    rates = {}
    for group, subset in df.groupby(group_col):
        total = len(subset)
        if total == 0:
            continue
        rates[str(group)] = float(subset[target_col].sum()) / float(total)
    return rates


def calculate_fairness_score(rates: dict) -> float:
    """
    Compute fairness score as (min_rate / max_rate) * 100.
    Returns 100.0 if rates are equal or only one group exists.
    Returns 0.0 if max_rate is 0.
    """
    if not rates:
        return 100.0
    values = list(rates.values())
    max_rate = max(values)
    min_rate = min(values)
    if max_rate == 0:
        return 0.0
    return max(0.0, min(100.0, (min_rate / max_rate) * 100.0))


def get_bias_verdict(score: float) -> tuple:
    """Return (verdict_text, color). score < 80 → biased, else fair."""
    if score < FAIRNESS_THRESHOLD:
        return "⚠️ Bias Detected", "red"
    return "✅ Fair System", "green"


def get_risk_level(score: float) -> tuple:
    """
    Categorise fairness score into risk tiers.
    Returns (label, color, icon).
      0–40  -> High Risk   🔴
      40–70 -> Medium Risk 🟡
      70–100-> Low Risk    🟢
    """
    if score < 40:
        return "High Risk", "red", "🔴"
    elif score < 70:
        return "Medium Risk", "orange", "🟡"
    return "Low Risk", "green", "🟢"


def generate_explanation(rates: dict, score: float, domain: dict = None, axis: str = "gender") -> str:
    """
    Produce a plain-language explanation of the bias analysis for a given axis.
    """
    if not rates:
        return "No data available for explanation."

    positive = domain["positive"] if domain else "Selected"
    axis_label = AXIS_LABELS.get(axis, axis.title())

    sorted_groups = sorted(rates.items(), key=lambda x: x[1])
    lines = [f"**{positive} rates by {axis_label}:**"]
    for group, rate in sorted_groups:
        lines.append(f"- {group}: {rate * 100:.1f}%")

    if score < FAIRNESS_THRESHOLD:
        lowest_group, lowest_rate = sorted_groups[0]
        highest_group, highest_rate = sorted_groups[-1]
        diff = (highest_rate - lowest_rate) * 100
        lines.append(
            f"\n**{lowest_group} candidates are {positive.lower()} significantly less than "
            f"{highest_group} candidates** ({lowest_rate * 100:.1f}% vs "
            f"{highest_rate * 100:.1f}%, a gap of {diff:.1f} percentage points). "
            f"Fairness score: {score:.1f}% — below the 80% threshold."
        )
    else:
        lines.append(
            f"\n{positive} rates are comparable across all {axis_label.lower()} groups. "
            f"Fairness score: {score:.1f}% — meets the 80% threshold."
        )

    return "\n".join(lines)


def analyse_all_axes(df: pd.DataFrame, axes: list, domain: dict = None) -> dict:
    """
    Run bias analysis across all provided axes (gender, caste, region).
    Returns a dict keyed by axis with keys: rates, score, verdict, risk, explanation, alert.
    """
    results = {}
    for axis in axes:
        if axis not in df.columns:
            continue
        rates   = calculate_selection_rates(df, group_col=axis)
        score   = calculate_fairness_score(rates)
        verdict, color = get_bias_verdict(score)
        risk_label, risk_color, risk_icon = get_risk_level(score)
        explanation = generate_explanation(rates, score, domain=domain, axis=axis)
        alert = AXIS_ALERTS.get(axis, "") if score < FAIRNESS_THRESHOLD else ""
        results[axis] = {
            "rates":       rates,
            "score":       score,
            "verdict":     verdict,
            "color":       color,
            "risk_label":  risk_label,
            "risk_icon":   risk_icon,
            "explanation": explanation,
            "alert":       alert,
        }
    return results


def most_disadvantaged_group(results: dict) -> tuple:
    """
    Find the axis and group with the lowest selection rate across all axes.
    Returns (axis, group, rate).
    """
    worst_axis, worst_group, worst_rate = None, None, 1.0
    for axis, data in results.items():
        for group, rate in data["rates"].items():
            if rate < worst_rate:
                worst_rate  = rate
                worst_group = group
                worst_axis  = axis
    return worst_axis, worst_group, worst_rate
