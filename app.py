"""
app.py
AI Fairness Checker — Advanced Edition with Gender, Caste & Region Bias Detection
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd

from data_loader import load_csv, load_sample_data, validate_columns, get_available_bias_axes
from bias_detector import (
    calculate_selection_rates,
    calculate_fairness_score,
    get_bias_verdict,
    get_risk_level,
    generate_explanation,
    analyse_all_axes,
    most_disadvantaged_group,
    AXIS_LABELS,
)
from model_trainer import (
    train_model,
    balance_dataset,
    get_model_prediction_rates,
    get_feature_importance,
    remove_sensitive_and_retrain,
    FEATURES,
)
from visualizer import (
    gender_distribution_bar,
    gender_distribution_pie,
    group_distribution_bar,
    selection_rate_bar,
    fairness_gauge,
    risk_meter,
    multi_axis_score_bar,
    before_after_comparison,
    feature_importance_chart,
    decision_comparison_chart,
)
from report_generator import generate_text_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fairness Checker",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    sim_mode = st.selectbox(
        "🌐 Real-World Simulation Mode",
        ["Hiring System", "Loan Approval", "Healthcare"],
    )

    st.divider()
    st.markdown("**Fix Bias Options**")
    fix_balance       = st.checkbox("Balance dataset (resampling)", value=True)
    fix_remove_gender = st.checkbox("Remove gender from model", value=False)
    fix_remove_caste  = st.checkbox("Remove caste from model", value=False)
    fix_remove_region = st.checkbox("Remove region from model", value=False)

    st.divider()
    st.caption("⚖️ AI Fairness Checker · Advanced Edition")

# ── Domain labels ─────────────────────────────────────────────────────────────
DOMAIN = {
    "Hiring System": {"target": "selected", "positive": "Hired",    "negative": "Rejected", "icon": "💼"},
    "Loan Approval": {"target": "selected", "positive": "Approved", "negative": "Denied",   "icon": "🏦"},
    "Healthcare":    {"target": "selected", "positive": "Treated",  "negative": "Untreated","icon": "🏥"},
}
domain = DOMAIN[sim_mode]

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"⚖️ AI Fairness Checker  {domain['icon']} {sim_mode}")
st.markdown(
    "Detect, explain, and reduce bias across **gender, caste, and region** in automated decision systems."
)
st.info(
    "🌍 **This system ensures fair decisions across social and regional groups, "
    "reducing real-world discrimination risks.**"
)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📂 1. Upload Dataset")

col_up, col_info = st.columns([2, 1])
with col_up:
    uploaded_file = st.file_uploader(
        "Upload CSV (columns: gender, caste, region, experience, education, selected)",
        type=["csv"],
    )
with col_info:
    st.info(
        "**Required:** gender, experience, education, selected\n\n"
        "**Optional (for extended analysis):** caste, region"
    )

if uploaded_file is not None:
    df_raw = load_csv(uploaded_file)
    data_source = "Uploaded file"
else:
    df_raw = load_sample_data()
    data_source = "Built-in sample dataset"
    st.success("Using built-in sample dataset — includes gender, caste & region bias.")

valid, err_msg = validate_columns(df_raw)
if not valid:
    st.error(f"❌ {err_msg}")
    st.stop()

df = df_raw.copy()
df.columns = df.columns.str.lower()

if df.empty:
    st.warning("Dataset is empty.")
    st.stop()

bias_axes = get_available_bias_axes(df)
st.caption(
    f"Data source: {data_source} — {len(df):,} records | "
    f"Bias axes available: {', '.join(a.title() for a in bias_axes)}"
)

with st.expander("Preview raw data (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📊 2. Data Analysis")

total = len(df)
gender_counts = df["gender"].value_counts().to_dict()

# Top metrics
mcols = st.columns(len(gender_counts) + 1)
mcols[0].metric("Total Records", f"{total:,}")
for i, (grp, cnt) in enumerate(gender_counts.items(), 1):
    mcols[i].metric(grp, f"{cnt:,}", f"{cnt/total*100:.1f}%")

# Distribution charts — one tab per axis
dist_tabs = st.tabs([f"{a.title()} Distribution" for a in bias_axes])
for tab, axis in zip(dist_tabs, bias_axes):
    with tab:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(group_distribution_bar(df, axis), use_container_width=True)
        with c2:
            if axis == "gender":
                st.plotly_chart(gender_distribution_pie(df), use_container_width=True)
            else:
                counts = df[axis].value_counts()
                st.dataframe(
                    counts.rename_axis(axis.title()).reset_index(name="Count"),
                    use_container_width=True,
                )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BIAS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("🔍 3. Bias Analysis Results")

analyze_btn = st.button("🔍 Analyze for Bias", type="primary")

if analyze_btn or st.session_state.get("analyzed"):
    st.session_state["analyzed"] = True

    # Run analysis across all available axes
    axis_results = analyse_all_axes(df, bias_axes, domain=domain)
    st.session_state["axis_results"] = axis_results

    # ── Overall summary bar ───────────────────────────────────────────────────
    st.plotly_chart(multi_axis_score_bar(axis_results), use_container_width=True)

    # ── Most disadvantaged group ──────────────────────────────────────────────
    worst_axis, worst_group, worst_rate = most_disadvantaged_group(axis_results)
    if worst_axis:
        st.error(
            f"🚨 **Most Disadvantaged Group:** **{worst_group}** ({worst_axis.title()}) "
            f"— selection rate only **{worst_rate*100:.1f}%**"
        )

    # ── Per-axis tabs ─────────────────────────────────────────────────────────
    axis_tabs = st.tabs([f"{AXIS_LABELS.get(a, a.title())} Bias" for a in bias_axes])

    for tab, axis in zip(axis_tabs, bias_axes):
        res = axis_results[axis]
        with tab:
            # Alert
            if res["alert"]:
                st.error(res["alert"])

            # Verdict + metrics
            fn = st.success if res["color"] == "green" else st.error
            fn(f"{res['verdict']}  |  Score: {res['score']:.1f}%  |  Risk: {res['risk_icon']} {res['risk_label']}")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(
                    selection_rate_bar(res["rates"], res["score"], domain, axis),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    fairness_gauge(res["score"], title=f"{axis.title()} Fairness Score (%)"),
                    use_container_width=True,
                )
            with c3:
                st.plotly_chart(risk_meter(res["score"]), use_container_width=True)

            st.info(res["explanation"])

    # ── AI Model + XAI ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("🤖 AI Model Analysis + Explainability")
    st.caption("Logistic Regression trained on experience + education. Feature importance shows what drove decisions.")

    try:
        model, preds = train_model(df)
        importance   = get_feature_importance(model, FEATURES)
        st.session_state.update({"model": model, "preds": preds, "importance": importance})

        xai_c1, xai_c2 = st.columns(2)
        with xai_c1:
            st.plotly_chart(feature_importance_chart(importance), use_container_width=True)
            top_feat = max(importance, key=importance.get)
            st.info(
                f"**Why did the model decide this way?**\n\n"
                f"The most influential feature was **{top_feat}** "
                f"(coefficient: {importance[top_feat]:.3f}). "
                f"Even without gender/caste/region as direct inputs, "
                f"correlated features can still carry hidden bias."
            )
        with xai_c2:
            # Show model prediction rates for each axis
            for axis in bias_axes:
                model_rates = get_model_prediction_rates(df, preds, group_col=axis)
                model_score = calculate_fairness_score(model_rates)
                st.plotly_chart(
                    selection_rate_bar(model_rates, model_score, domain, axis),
                    use_container_width=True,
                )

        # Decision Comparison Panel
        st.divider()
        st.subheader("⚖️ Decision Comparison Panel")
        _, fair_preds = remove_sensitive_and_retrain(df)
        st.session_state["fair_preds"] = fair_preds

        comp_tabs = st.tabs([a.title() for a in bias_axes])
        for tab, axis in zip(comp_tabs, bias_axes):
            with tab:
                st.plotly_chart(
                    decision_comparison_chart(df, preds, fair_preds, domain, group_col=axis),
                    use_container_width=True,
                )

    except Exception as e:
        st.error(f"Model training failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — IMPROVEMENTS & FIX BIAS
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("🛠️ 4. Improvements & Fix Bias")

suggestions = [
    ("⚖️ Balance the dataset",        "Resample minority groups (gender, caste, region) to equal representation."),
    ("🚫 Remove sensitive attributes", "Exclude gender, caste, region from model features."),
    ("🎯 Fairness-aware training",     "Apply re-weighting, adversarial debiasing, or post-processing calibration."),
    ("🔄 Audit regularly",             "Re-run this analysis after every model update or dataset refresh."),
    ("👥 Diverse review panels",       "Combine automated checks with human oversight from diverse stakeholders."),
]

scols = st.columns(len(suggestions))
for col, (title, desc) in zip(scols, suggestions):
    with col:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(desc)

st.markdown("")

if st.session_state.get("analyzed"):
    fix_btn = st.button("🔧 Fix Bias (Simulate)", type="secondary")

    if fix_btn or st.session_state.get("fixed"):
        st.session_state["fixed"] = True

        with st.spinner("Applying fixes and retraining…"):
            df_fixed = df.copy()

            # Balance across all selected axes
            cols_to_balance = [c for c, flag in
                               [("gender", fix_balance), ("caste", fix_balance), ("region", fix_balance)]
                               if flag and c in df_fixed.columns]
            if cols_to_balance:
                df_fixed = balance_dataset(df_fixed, group_cols=cols_to_balance)

            # Remove selected sensitive attributes
            remove_cols = [c for c, flag in
                           [("gender", fix_remove_gender), ("caste", fix_remove_caste), ("region", fix_remove_region)]
                           if flag]
            if remove_cols:
                _, preds_fixed = remove_sensitive_and_retrain(df_fixed, sensitive_cols=remove_cols)
            else:
                _, preds_fixed = train_model(df_fixed)

            # Compute after-fix results for all axes
            after_results = analyse_all_axes(df_fixed, bias_axes, domain=domain)
            st.session_state["after_results"] = after_results

        # Summary
        st.success("✅ Bias fix applied!")
        st.plotly_chart(multi_axis_score_bar(after_results), use_container_width=True)

        # Before vs After per axis
        st.subheader("📊 Before vs After Comparison Dashboard")
        prev = st.session_state.get("axis_results", {})

        ba_tabs = st.tabs([a.title() for a in bias_axes])
        for tab, axis in zip(ba_tabs, bias_axes):
            with tab:
                if axis in prev and axis in after_results:
                    b_score = prev[axis]["score"]
                    a_score = after_results[axis]["score"]
                    delta   = a_score - b_score
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Before Score", f"{b_score:.1f}%")
                    c2.metric("After Score",  f"{a_score:.1f}%", delta=f"+{delta:.1f}%")
                    c3.metric("Risk After",   f"{after_results[axis]['risk_icon']} {after_results[axis]['risk_label']}")
                    st.plotly_chart(
                        before_after_comparison(
                            prev[axis]["rates"], after_results[axis]["rates"],
                            b_score, a_score, axis=axis,
                        ),
                        use_container_width=True,
                    )

    # ── Download Report ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("📄 Auto Report Generator")

    if st.session_state.get("axis_results"):
        ar = st.session_state["axis_results"]
        gender_res = ar.get("gender", {})
        report_text = generate_text_report(
            stats={"total_records": total, "gender_counts": gender_counts},
            rates=gender_res.get("rates", {}),
            score=gender_res.get("score", 0),
            verdict=gender_res.get("verdict", ""),
            explanation=gender_res.get("explanation", ""),
            risk_label=gender_res.get("risk_label", "N/A"),
            sim_mode=sim_mode,
            importance=st.session_state.get("importance"),
            model_rates=None,
            model_score=None,
            after_rates=st.session_state.get("after_results", {}).get("gender", {}).get("rates"),
            after_score=st.session_state.get("after_results", {}).get("gender", {}).get("score"),
            extra_axes={k: v for k, v in ar.items() if k != "gender"},
        )
        st.download_button(
            label="⬇️ Download Full Audit Report (.txt)",
            data=report_text,
            file_name="fairness_audit_report.txt",
            mime="text/plain",
        )
        with st.expander("Preview report"):
            st.text(report_text[:2500] + "\n…[truncated]")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — WHY THIS MATTERS
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("💡 5. Why This Matters")

st.markdown(
    "Algorithmic bias is a **documented, real-world problem** affecting millions of people. "
    "Bias can stem from gender, caste, region, and many other factors."
)

wc1, wc2, wc3 = st.columns(3)
with wc1:
    with st.container(border=True):
        st.markdown("### 💼 Hiring Discrimination")
        st.write(
            "Automated resume screening tools systematically downrank candidates from "
            "underrepresented genders, lower castes, or rural regions — "
            "silently filtering out qualified people before a human ever sees their application."
        )
        st.error("Real impact: Qualified candidates never get interviews.")

with wc2:
    with st.container(border=True):
        st.markdown("### 🏦 Loan Approval Bias")
        st.write(
            "Credit-scoring algorithms encode past discriminatory lending practices. "
            "Applicants from lower-caste or rural backgrounds may be denied loans "
            "not because of their creditworthiness, but because of inherited bias."
        )
        st.error("Real impact: Families denied financial opportunities.")

with wc3:
    with st.container(border=True):
        st.markdown("### 🏥 Healthcare Inequality")
        st.write(
            "Clinical AI tools show lower accuracy for minority groups due to "
            "under-representation in training data — leading to delayed diagnoses "
            "and worse outcomes for already-vulnerable communities."
        )
        st.error("Real impact: People receive worse medical care.")

st.info(
    "**Fairness is not just a technical metric — it is an ethical obligation.** "
    "Every percentage point of fairness score improvement represents real people "
    "being treated more equitably by automated systems."
)

st.divider()
st.caption("⚖️ AI Fairness Checker · Advanced Edition · Gender + Caste + Region · Built with Streamlit, scikit-learn & Plotly")
