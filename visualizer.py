"""
visualizer.py
All Plotly chart builders for the AI Fairness Checker dashboard.
Supports gender, caste, and region bias axes.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

FAIR_COLOR    = "#2ecc71"
BIASED_COLOR  = "#e74c3c"
NEUTRAL_COLOR = "#3498db"
WARN_COLOR    = "#f39c12"
FAIRNESS_THRESHOLD = 80.0

AXIS_PALETTES = {
    "gender": [NEUTRAL_COLOR, "#9b59b6"],
    "caste":  ["#e67e22", "#e74c3c", "#8e44ad", "#2c3e50"],
    "region": ["#27ae60", "#f39c12", "#c0392b"],
}


def _rate_color(score: float) -> str:
    return FAIR_COLOR if score >= FAIRNESS_THRESHOLD else BIASED_COLOR


def group_distribution_bar(df: pd.DataFrame, col: str, title: str = None) -> go.Figure:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    palette = AXIS_PALETTES.get(col, px.colors.qualitative.Set2)
    fig = px.bar(
        counts, x=col, y="count", color=col,
        color_discrete_sequence=palette,
        title=title or f"{col.title()} Distribution",
        labels={"count": "Candidates", col: col.title()},
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
    return fig


def gender_distribution_bar(df: pd.DataFrame) -> go.Figure:
    return group_distribution_bar(df, "gender", "Gender Distribution")


def gender_distribution_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["gender"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.4,
        marker_colors=AXIS_PALETTES["gender"],
    ))
    fig.update_layout(title="Gender Proportion")
    return fig


def selection_rate_bar(rates: dict, score: float, domain: dict = None, axis: str = "gender") -> go.Figure:
    groups = list(rates.keys())
    values = [rates[g] * 100 for g in groups]
    bar_color = _rate_color(score)
    label = domain["positive"] if domain else "Selected"
    axis_label = axis.title()
    fig = go.Figure(go.Bar(
        x=groups, y=values,
        marker_color=bar_color,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{label} Rate by {axis_label}",
        yaxis_title=f"{label} Rate (%)",
        xaxis_title=axis_label,
        yaxis=dict(range=[0, 115]),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_hline(y=FAIRNESS_THRESHOLD, line_dash="dash", line_color="orange",
                  annotation_text="80% fairness threshold")
    return fig


def fairness_gauge(score: float, title: str = "Fairness Score (%)") -> go.Figure:
    color = FAIR_COLOR if score >= FAIRNESS_THRESHOLD else BIASED_COLOR
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": FAIRNESS_THRESHOLD, "valueformat": ".1f"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 80],  "color": "#fadbd8"},
                {"range": [80, 100],"color": "#d5f5e3"},
            ],
            "threshold": {
                "line": {"color": "orange", "width": 4},
                "thickness": 0.75,
                "value": FAIRNESS_THRESHOLD,
            },
        },
    ))
    fig.update_layout(height=280)
    return fig


def risk_meter(score: float) -> go.Figure:
    bar_color = "#e74c3c" if score < 40 else ("#f39c12" if score < 70 else "#2ecc71")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Bias Risk Meter"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, 40],  "color": "#fadbd8"},
                {"range": [40, 70], "color": "#fef9e7"},
                {"range": [70, 100],"color": "#d5f5e3"},
            ],
        },
    ))
    fig.update_layout(height=280)
    return fig


def multi_axis_score_bar(axis_results: dict) -> go.Figure:
    axes   = list(axis_results.keys())
    scores = [axis_results[a]["score"] for a in axes]
    colors = [FAIR_COLOR if s >= FAIRNESS_THRESHOLD else (WARN_COLOR if s >= 40 else BIASED_COLOR)
              for s in scores]
    labels = [a.title() for a in axes]
    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        title="Fairness Score by Bias Axis",
        xaxis=dict(range=[0, 115], title="Fairness Score (%)"),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_vline(x=FAIRNESS_THRESHOLD, line_dash="dash", line_color="orange",
                  annotation_text="80% threshold")
    return fig


def before_after_comparison(
    before_rates: dict, after_rates: dict,
    before_score: float, after_score: float,
    axis: str = "gender",
) -> go.Figure:
    groups = sorted(set(list(before_rates.keys()) + list(after_rates.keys())))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Before Fix", x=groups,
        y=[before_rates.get(g, 0) * 100 for g in groups],
        marker_color=BIASED_COLOR,
        text=[f"{before_rates.get(g, 0)*100:.1f}%" for g in groups],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="After Fix", x=groups,
        y=[after_rates.get(g, 0) * 100 for g in groups],
        marker_color=FAIR_COLOR,
        text=[f"{after_rates.get(g, 0)*100:.1f}%" for g in groups],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{axis.title()} — Before vs After  |  {before_score:.1f}% → {after_score:.1f}%",
        barmode="group",
        yaxis_title="Selection Rate (%)",
        xaxis_title=axis.title(),
        yaxis=dict(range=[0, 120]),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def feature_importance_chart(importance: dict) -> go.Figure:
    sensitive = {"gender", "caste", "region"}
    features = list(importance.keys())
    values   = [importance[f] for f in features]
    colors   = [BIASED_COLOR if f in sensitive else NEUTRAL_COLOR for f in features]
    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Importance (XAI)",
        xaxis_title="Absolute Coefficient",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def decision_comparison_chart(
    df: pd.DataFrame,
    original_preds: np.ndarray,
    fair_preds: np.ndarray,
    domain: dict = None,
    group_col: str = "gender",
) -> go.Figure:
    positive = domain["positive"] if domain else "Selected"
    groups = sorted(df[group_col].unique())
    orig_rates, fair_rates = {}, {}
    for g in groups:
        mask = df[group_col] == g
        n = mask.sum()
        orig_rates[g] = float(original_preds[mask].sum()) / n if n else 0
        fair_rates[g] = float(fair_preds[mask].sum()) / n if n else 0
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Original AI Decision", x=groups,
        y=[orig_rates[g] * 100 for g in groups],
        marker_color=BIASED_COLOR,
        text=[f"{orig_rates[g]*100:.1f}%" for g in groups],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Fair-Adjusted Decision", x=groups,
        y=[fair_rates[g] * 100 for g in groups],
        marker_color=FAIR_COLOR,
        text=[f"{fair_rates[g]*100:.1f}%" for g in groups],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Original vs Fair-Adjusted {positive} Rates ({group_col.title()})",
        barmode="group",
        yaxis_title=f"{positive} Rate (%)",
        xaxis_title=group_col.title(),
        yaxis=dict(range=[0, 120]),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
