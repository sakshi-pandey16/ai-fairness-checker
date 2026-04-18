"""
report_generator.py
Generates a plain-text audit report summarising the bias analysis.
"""

from datetime import datetime


def generate_text_report(
    stats: dict,
    rates: dict,
    score: float,
    verdict: str,
    explanation: str,
    risk_label: str = "N/A",
    sim_mode: str = "Hiring System",
    importance: dict = None,
    model_rates: dict = None,
    model_score: float = None,
    after_rates: dict = None,
    after_score: float = None,
    extra_axes: dict = None,
) -> str:
    """
    Build a formatted text report containing:
    - Dataset statistics
    - Fairness score and verdict
    - Selection rates per group
    - Model prediction rates (if available)
    - Before/after comparison (if fix was applied)
    - Bias mitigation suggestions
    Returns a multi-line string.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 60,
        "       AI FAIRNESS CHECKER — AUDIT REPORT",
        f"       Generated: {now}",
        f"       Simulation Mode: {sim_mode}",
        "=" * 60,
        "",
        "DATASET STATISTICS",
        "-" * 40,
        f"Total records      : {stats.get('total_records', 'N/A')}",
    ]

    gender_counts = stats.get("gender_counts", {})
    for group, count in gender_counts.items():
        pct = (count / stats.get("total_records", 1)) * 100
        lines.append(f"  {group:<18}: {count} ({pct:.1f}%)")

    lines += [
        "",
        "BIAS DETECTION RESULTS",
        "-" * 40,
        f"Fairness Score     : {score:.1f}%",
        f"Risk Level         : {risk_label}",
        f"Verdict            : {verdict}",
        "",
        "Selection Rates by Group:",
    ]
    for group, rate in sorted(rates.items()):
        lines.append(f"  {group:<18}: {rate * 100:.1f}%")

    lines += ["", "Explanation:", explanation, ""]

    # ── Extra bias axes (caste, region) ───────────────────────────────────────
    if extra_axes:
        for axis, data in extra_axes.items():
            lines += [
                f"{axis.upper()} BIAS ANALYSIS",
                "-" * 40,
                f"Fairness Score : {data.get('score', 0):.1f}%",
                f"Risk Level     : {data.get('risk_label', 'N/A')}",
                f"Verdict        : {data.get('verdict', '')}",
                "",
                f"Selection Rates by {axis.title()}:",
            ]
            for group, rate in sorted(data.get("rates", {}).items()):
                lines.append(f"  {group:<18}: {rate * 100:.1f}%")
            if data.get("alert"):
                lines.append(f"  ALERT: {data['alert']}")
            lines.append("")

    if importance:
        lines += [
            "MODEL EXPLAINABILITY (XAI)",
            "-" * 40,
            "Feature importance (absolute coefficients):",
        ]
        for feat, val in sorted(importance.items(), key=lambda x: -x[1]):
            lines.append(f"  {feat:<18}: {val:.4f}")
        lines.append("")

    if model_rates is not None:
        lines += [
            "MODEL PREDICTION RATES",
            "-" * 40,
            f"Model Fairness Score: {model_score:.1f}%" if model_score is not None else "",
        ]
        for group, rate in sorted(model_rates.items()):
            lines.append(f"  {group:<18}: {rate * 100:.1f}%")
        lines.append("")

    if after_rates is not None and after_score is not None:
        lines += [
            "BIAS FIX SIMULATION",
            "-" * 40,
            f"Before Fix Score   : {score:.1f}%",
            f"After Fix Score    : {after_score:.1f}%",
            "",
            "After-Fix Selection Rates:",
        ]
        for group, rate in sorted(after_rates.items()):
            lines.append(f"  {group:<18}: {rate * 100:.1f}%")
        lines.append("")

    lines += [
        "BIAS MITIGATION SUGGESTIONS",
        "-" * 40,
        "1. Balance the dataset — ensure equal representation of all groups.",
        "2. Remove sensitive attributes — exclude gender from model features.",
        "3. Use fairness-aware training — apply re-weighting or adversarial debiasing.",
        "4. Audit regularly — re-run this analysis after each model update.",
        "5. Diverse review panels — combine automated checks with human oversight.",
        "",
        "=" * 60,
        "End of Report",
        "=" * 60,
    ]

    return "\n".join(lines)
