"""Emotion Detection in Text - Interactive Streamlit Web Application.

Provides a unified UI for analyzing emotion in text using a classical baseline
(TF-IDF + Logistic Regression) and a fine-tuned DistilBERT transformer, with
LIME explainability, side-by-side model comparison, batch inference, and experimental results.
"""

from typing import Any, Optional
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.config import (
    ACTIVE_DISTILBERT,
    ACTIVE_DISTILBERT_DIR,
    ACTIVE_DISTILBERT_PREDS,
    FIGURES_DIR,
    LABELS,
    METRICS_DIR,
    MODEL_DIRS,
    MODELS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    available_models,
)
from src.explain import explain_text
from src.predict import EmotionPredictor
from src.presets import build_pools, next_sample

# Emotion emojis and theme accents
EMOTION_EMOJIS: dict[str, str] = {
    "sadness": "😢",
    "joy": "😊",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

EMOTION_COLORS: dict[str, str] = {
    "sadness": "#4A90E2",
    "joy": "#F5A623",
    "love": "#E91E63",
    "anger": "#D0021B",
    "fear": "#7B1FA2",
    "surprise": "#00BFA5",
}

PRESET_EMOTIONS: list[tuple[str, str]] = [
    ("joy", "😊 Joy"),
    ("sadness", "😢 Sadness"),
    ("love", "❤️ Love"),
    ("anger", "😠 Anger"),
    ("fear", "😨 Fear"),
    ("surprise", "😲 Surprise"),
]


@st.cache_data
def get_cached_pools() -> dict[str, list[str]]:
    """Build and cache the prediction sentence pools for quick preset buttons with fallback."""
    try:
        return build_pools()
    except Exception:
        from src.presets import FALLBACK_POOLS
        return {label: list(FALLBACK_POOLS[label]) for label in LABELS}


# ==============================================================================
# Model Loading & Caching (Single source of truth: src.predict & src.config)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_cached_predictor(model_type: str) -> EmotionPredictor:
    """Load and cache an EmotionPredictor instance.

    Uses st.cache_resource to ensure model weights and tokenizers are loaded once.

    Args:
        model_type: 'baseline' or 'distilbert'.

    Returns:
        EmotionPredictor instance.
    """
    predictor = EmotionPredictor(model_type=model_type)
    predictor._ensure_loaded()
    return predictor


def safe_get_predictor(model_type: str) -> Optional[EmotionPredictor]:
    """Retrieve cached predictor with user-friendly error handling.

    Args:
        model_type: 'baseline' or 'distilbert'.

    Returns:
        EmotionPredictor or None if model files are missing.
    """
    try:
        return get_cached_predictor(model_type)
    except FileNotFoundError as err:
        st.error(f"⚠️ Model files for **{model_type}** are missing:\n\n`{err}`")
        return None
    except Exception as err:
        st.error(f"⚠️ Failed to initialize **{model_type}** predictor: `{err}`")
        return None


def load_json_metrics(file_path: Path) -> Optional[dict[str, Any]]:
    """Read a JSON metrics file safely without raising errors."""
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ==============================================================================
# UI Styling
# ==============================================================================
def apply_custom_styles() -> None:
    """Inject custom CSS for modern aesthetics."""
    st.markdown(
        """
        <style>
            /* Header Accent */
            .main-title {
                font-size: 2.2rem;
                font-weight: 700;
                background: linear-gradient(120deg, #4A90E2 0%, #9013FE 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.2rem;
            }
            .sub-title {
                color: #888888;
                font-size: 1.05rem;
                margin-bottom: 1.2rem;
            }
            /* Metric Card */
            .emotion-card {
                padding: 1.2rem;
                border-radius: 10px;
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.12);
                margin-bottom: 1rem;
            }
            .emotion-badge {
                font-size: 1.6rem;
                font-weight: 700;
                display: inline-block;
                padding: 0.3rem 0.8rem;
                border-radius: 8px;
            }
            /* Tab container polish */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                padding: 10px 16px;
                border-radius: 6px 6px 0 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# Sidebar Component
# ==============================================================================
def render_sidebar() -> str:
    """Render the sidebar with model selector, project overview, and metrics.

    Returns:
        Selected model key ('baseline' or 'distilbert').
    """
    st.sidebar.markdown("## 🎭 Emotion AI Studio")
    st.sidebar.markdown(
        "Compare a classical NLP baseline with fine-tuned DistilBERT on the "
        "HuggingFace `dair-ai/emotion` benchmark."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Model Selector")

    avail = available_models()
    if not avail:
        st.sidebar.error(
            "❌ **No models found in `models/`!**\n\n"
            "Please download pre-trained models:\n"
            "```bash\npython -m src.download_models\n```\n"
            "or train a baseline model:\n"
            "```bash\npython -m src.data\npython -m src.baseline\n```"
        )
        st.error(
            "❌ **No models found in `models/`!**\n\n"
            "Please run `python -m src.download_models` to install model artifacts, "
            "or train from scratch with `python -m src.baseline`."
        )
        st.stop()

    if ACTIVE_DISTILBERT not in avail and "baseline" in avail:
        st.sidebar.info(
            f"ℹ️ Active model **{ACTIVE_DISTILBERT}** is not installed. Defaulting to **Baseline**.\n\n"
            "To download transformer weights, run:\n```bash\npython -m src.download_models\n```"
        )

    all_display_names = {
        "distilbert_v2": "DistilBERT v2 (improved)",
        "distilbert": "DistilBERT v1",
        "baseline": "Baseline (TF-IDF + Logistic Regression)",
    }
    # Only include models whose weights are present
    model_display_names = {
        all_display_names[m]: m for m in avail if m in all_display_names
    }

    if ACTIVE_DISTILBERT in model_display_names.values():
        default_idx = list(model_display_names.values()).index(ACTIVE_DISTILBERT)
    elif "baseline" in model_display_names.values():
        default_idx = list(model_display_names.values()).index("baseline")
    else:
        default_idx = 0

    selected_display = st.sidebar.selectbox(
        "Active Classifier",
        list(model_display_names.keys()),
        index=default_idx,
        help="Select the active model used across Analyze, Explain, and Batch tabs.",
    )
    selected_model_type = model_display_names[selected_display]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Metrics Summary")

    baseline_metrics = load_json_metrics(METRICS_DIR / "baseline.json")
    v1_metrics = load_json_metrics(METRICS_DIR / "distilbert.json")
    if v1_metrics is None:
        v1_metrics = load_json_metrics(METRICS_DIR / "distilbert_smoke.json")
    v2_metrics = load_json_metrics(METRICS_DIR / "distilbert_v2.json")

    col_b, col_v1 = st.sidebar.columns(2)
    with col_b:
        st.markdown("**Baseline**")
        if baseline_metrics:
            acc = baseline_metrics.get("accuracy", 0.0)
            f1 = baseline_metrics.get("macro_f1", 0.0)
            st.metric("Accuracy", f"{acc * 100:.1f}%")
            st.metric("Macro-F1", f"{f1 * 100:.1f}%")
        else:
            st.caption("No metrics found")

    with col_v1:
        st.markdown("**DistilBERT v1**")
        if v1_metrics:
            acc = v1_metrics.get("accuracy", 0.0)
            f1 = v1_metrics.get("macro_f1", 0.0)
            st.metric("Accuracy", f"{acc * 100:.1f}%")
            st.metric("Macro-F1", f"{f1 * 100:.1f}%")
        else:
            st.caption("No metrics found")

    if v2_metrics:
        st.sidebar.markdown("**DistilBERT v2 (improved)**")
        col_v2_1, col_v2_2 = st.sidebar.columns(2)
        with col_v2_1:
            st.metric("Accuracy", f"{v2_metrics.get('accuracy', 0.0) * 100:.1f}%")
        with col_v2_2:
            st.metric("Macro-F1", f"{v2_metrics.get('macro_f1', 0.0) * 100:.1f}%")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Dataset Overview")
    st.sidebar.markdown(
        "- **Corpus**: `dair-ai/emotion`\n"
        "- **Splits**: 16,000 train / 2,000 val / 2,000 test\n"
        "- **Classes**: 6 canonical emotion labels\n"
        "  - 😢 `sadness`\n"
        "  - 😊 `joy`\n"
        "  - ❤️ `love`\n"
        "  - 😠 `anger`\n"
        "  - 😨 `fear`\n"
        "  - 😲 `surprise`"
    )

    return selected_model_type


# ==============================================================================
# Tab 1: Analyze Single Text
# ==============================================================================
def render_analyze_tab(selected_model_type: str) -> None:
    """Render Tab 1: Interactive single text analysis with probabilities bar chart."""
    st.markdown("### 🔍 Real-Time Emotion Classification")
    st.markdown("Type any English text below or click one of the preset benchmark samples.")

    pools = get_cached_pools()

    # Initial default text if not already in session state
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = (
            "I just found out that I got accepted into my dream university! I am absolutely thrilled and grateful!"
        )

    def set_text(emotion_label: str) -> None:
        """Callback to cycle through emotion pool sentences into input_text."""
        st.session_state["input_text"] = next_sample(emotion_label, st.session_state, pools=pools)

    # Preset button row
    st.markdown("**Quick Preset Examples:**")
    button_cols = st.columns(len(PRESET_EMOTIONS))
    for idx, (label, display_label) in enumerate(PRESET_EMOTIONS):
        button_cols[idx].button(
            display_label,
            key=f"btn_preset_{label}",
            on_click=set_text,
            args=(label,),
            use_container_width=True,
        )

    user_text = st.text_area(
        "Text to Analyze:",
        height=110,
        placeholder="Enter a sentence or paragraph describing an emotional experience...",
        key="input_text",
    )

    col_btn, _ = st.columns([1, 4])
    classify_clicked = col_btn.button("⚡ Classify Emotion", type="primary", use_container_width=True)

    if classify_clicked or user_text:
        if not user_text or not user_text.strip():
            st.warning("⚠️ Input is empty. Please enter or select non-empty text to analyze.")
            return

        predictor = safe_get_predictor(selected_model_type)
        if predictor is None:
            return

        try:
            with st.spinner(f"Analyzing with {selected_model_type.capitalize()}..."):
                result = predictor.predict(user_text)
        except ValueError as val_err:
            st.warning(f"⚠️ Invalid input: {val_err}")
            return
        except Exception as err:
            st.error(f"⚠️ Inference error: {err}")
            return

        pred_label = result["label"]
        confidence = result["confidence"]
        all_probs = result["all_probs"]
        is_confident = result.get("is_confident", True)
        emoji = EMOTION_EMOJIS.get(pred_label, "✨")

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if not is_confident:
                st.markdown("#### Primary Prediction")
                sorted_emotions = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                top1_lbl, top1_prob = sorted_emotions[0]
                top2_lbl, top2_prob = sorted_emotions[1]

                st.warning("⚠️ **Not sure: low confidence**")
                st.markdown(
                    f"""
                    <div class="emotion-card" style="border: 1px dashed #FFA500;">
                        <div style="font-size: 1.15rem; font-weight: bold; color: #FFA500; margin-bottom: 6px;">
                            🤔 Not sure: low confidence
                        </div>
                        <div style="font-size: 0.9rem; color: #AAA; margin-bottom: 12px;">
                            Prediction confidence ({confidence * 100:.1f}%) is below the selective abstain threshold.
                        </div>
                        <div style="font-size: 0.95rem;">
                            <strong>Top 2 candidate emotions:</strong>
                            <div style="margin-top: 6px;">
                                1. {EMOTION_EMOJIS.get(top1_lbl, '')} <strong>{top1_lbl.capitalize()}</strong>: {top1_prob * 100:.1f}%
                            </div>
                            <div style="margin-top: 4px;">
                                2. {EMOTION_EMOJIS.get(top2_lbl, '')} <strong>{top2_lbl.capitalize()}</strong>: {top2_prob * 100:.1f}%
                            </div>
                        </div>
                        <div style="font-size: 0.85rem; color: #888; margin-top: 10px;">
                            Model: {selected_model_type} (calibrated)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("#### Primary Prediction")
                st.markdown(
                    f"""
                    <div class="emotion-card">
                        <div style="font-size: 0.95rem; color: #888;">Detected Emotion:</div>
                        <div class="emotion-badge" style="color: {EMOTION_COLORS.get(pred_label, '#FFF')};">
                            {emoji} {pred_label.upper()}
                        </div>
                        <div style="margin-top: 10px; font-size: 1.1rem;">
                            <strong>Confidence:</strong> {confidence * 100:.2f}%
                        </div>
                        <div style="font-size: 0.85rem; color: #888; margin-top: 4px;">
                            Model: {selected_model_type.capitalize()}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with res_col2:
            st.markdown("#### Class Probabilities Breakdown")
            chart_df = pd.DataFrame([
                {
                    "Emotion": f"{EMOTION_EMOJIS.get(l, '')} {l.capitalize()}",
                    "Probability": round(prob, 4),
                }
                for l, prob in all_probs.items()
            ])
            st.bar_chart(chart_df.set_index("Emotion"), height=240, use_container_width=True)


# ==============================================================================
# Tab 2: LIME Explainability
# ==============================================================================
def render_explain_tab(selected_model_type: str) -> None:
    """Render Tab 2: LIME interactive explanation for DistilBERT."""
    st.markdown("### 🔬 LIME Model Explainability")
    st.markdown(
        "Local Interpretable Model-agnostic Explanations (LIME) perturb tokens to identify "
        "which specific words positively or negatively drove the model's classification."
    )

    if selected_model_type == "baseline":
        st.info(
            "ℹ️ **DistilBERT Feature**: LIME local surrogate explanations are configured for the "
            "deep contextual DistilBERT model. The Baseline (TF-IDF + Logistic Regression) relies on "
            "fixed linear coefficients. You can switch to **DistilBERT** in the sidebar, or run LIME "
            "using DistilBERT directly below."
        )

    # Use the current text from Tab 1 if available
    default_text = st.session_state.get(
        "input_text",
        "I just found out that I got accepted into my dream university! I am absolutely thrilled and grateful!",
    )
    explain_input = st.text_area(
        "Text to Explain with LIME:",
        value=default_text,
        height=100,
        placeholder="Enter text to generate token contribution weights...",
        key="lime_explain_text",
    )

    col_samples, col_feat, col_btn = st.columns([1, 1, 1.2])
    num_samples = col_samples.select_slider(
        "Perturbation Samples",
        options=[100, 250, 300, 500],
        value=300,
        help="Higher samples produce more stable explanations but take slightly longer on CPU.",
    )
    num_features = col_feat.slider("Top Features", min_value=4, max_value=12, value=8)

    st.caption("⏱️ Note: Generating explanations on CPU takes approximately 2-4 seconds.")
    run_explain = col_btn.button("✨ Generate LIME Explanation", type="primary", use_container_width=True)

    if run_explain:
        if not explain_input or not explain_input.strip():
            st.warning("⚠️ Input is empty. Please enter non-empty text to generate an explanation.")
            return

        with st.spinner("Generating LIME perturbation explanation on CPU..."):
            try:
                if selected_model_type != "baseline" and selected_model_type in MODEL_DIRS:
                    target_model_dir = MODEL_DIRS[selected_model_type]
                elif ACTIVE_DISTILBERT_DIR.exists():
                    target_model_dir = ACTIVE_DISTILBERT_DIR
                elif MODEL_DIRS["distilbert"].exists():
                    target_model_dir = MODEL_DIRS["distilbert"]
                else:
                    target_model_dir = ACTIVE_DISTILBERT_DIR

                res = explain_text(
                    text=explain_input,
                    num_samples=num_samples,
                    num_features=num_features,
                    model_dir=target_model_dir,
                )
            except FileNotFoundError as fnf:
                st.error(f"⚠️ DistilBERT model not found for explainability: {fnf}")
                return
            except Exception as err:
                st.error(f"⚠️ Explanation generation failed: {err}")
                return

        if res.get("error"):
            st.warning(f"⚠️ {res['error']}")
            return

        pred_label = res["predicted_label"]
        weights = res["weights"]
        exp = res["explanation"]

        st.markdown("---")
        st.markdown(
            f"#### Explanation for Predicted Emotion: "
            f"**{EMOTION_EMOJIS.get(pred_label, '')} {pred_label.upper()}**"
        )

        col_html, col_table = st.columns([2, 1])

        with col_html:
            st.markdown("**Interactive LIME Visualization:**")
            if exp is not None:
                html_data = exp.as_html()
                components.html(html_data, height=420, scrolling=True)

        with col_table:
            st.markdown("**Word Attributions:**")
            if weights:
                w_df = pd.DataFrame(weights, columns=["Token", "Weight"])
                w_df["Impact"] = w_df["Weight"].apply(
                    lambda w: "🟢 Positive" if w > 0 else "🔴 Negative"
                )
                w_df["Weight"] = w_df["Weight"].apply(lambda w: round(w, 4))
                st.dataframe(w_df, use_container_width=True, height=360)


# ==============================================================================
# Tab 3: Compare Models Side-by-Side
# ==============================================================================
def render_compare_tab() -> None:
    """Render Tab 3: Side-by-side comparison of models."""
    st.markdown("### ⚖️ Side-by-Side Model Comparison")
    st.markdown(
        "Evaluate classifiers on identical input text to detect agreement, divergence, "
        "and nuance handling (e.g. negations, sarcasm, or subtle emotional cues)."
    )

    avail = available_models()
    all_compare_options = {
        "baseline": "Baseline (TF-IDF + LR)",
        "distilbert": "DistilBERT v1",
        "distilbert_v2": "DistilBERT v2 (improved)",
    }
    model_options = {
        all_compare_options[m]: m for m in avail if m in all_compare_options
    }

    if len(model_options) < 2:
        st.info(
            "ℹ️ Side-by-side comparison requires at least 2 available models.\n\n"
            f"Currently available: {list(model_options.keys()) or 'None'}.\n\n"
            "Run `python -m src.download_models` to download pre-trained transformer models."
        )
        return

    col_m1, col_m2 = st.columns(2)
    keys_list = list(model_options.keys())
    with col_m1:
        m1_display = st.selectbox("Model A", keys_list, index=0)
    with col_m2:
        m2_display = st.selectbox(
            "Model B", keys_list, index=min(1, len(keys_list) - 1)
        )

    m1_key = model_options[m1_display]
    m2_key = model_options[m2_display]

    default_text = st.session_state.get(
        "input_text",
        "I was really looking forward to the holiday, but the constant rain ruined all our plans.",
    )
    compare_input = st.text_area(
        "Input Text for Head-to-Head Comparison:",
        value=default_text,
        height=100,
        key="compare_input_text",
    )

    if st.button("🚀 Compare Selected Models", type="primary", use_container_width=False):
        if not compare_input or not compare_input.strip():
            st.warning("⚠️ Input is empty. Please enter non-empty text to compare.")
            return

        pred_a = safe_get_predictor(m1_key)
        pred_b = safe_get_predictor(m2_key)

        if pred_a is None or pred_b is None:
            return

        with st.spinner(f"Executing inference across {m1_display} and {m2_display}..."):
            try:
                res_a = pred_a.predict(compare_input)
                res_b = pred_b.predict(compare_input)
            except Exception as err:
                st.error(f"⚠️ Comparison inference error: {err}")
                return

        # Agreement / Disagreement Banner
        agrees = res_a["label"] == res_b["label"]
        st.markdown("---")
        if agrees:
            st.success(
                f"🤝 **Models Agree!** Both classifiers predicted **{res_a['label'].upper()}** "
                f"{EMOTION_EMOJIS.get(res_a['label'], '')} "
                f"({m1_display}: {res_a['confidence'] * 100:.1f}%, "
                f"{m2_display}: {res_b['confidence'] * 100:.1f}%)."
            )
        else:
            st.warning(
                f"⚡ **Model Disagreement!** "
                f"{m1_display} predicted **{res_a['label'].upper()}** {EMOTION_EMOJIS.get(res_a['label'], '')} "
                f"({res_a['confidence'] * 100:.1f}%), whereas {m2_display} predicted "
                f"**{res_b['label'].upper()}** {EMOTION_EMOJIS.get(res_b['label'], '')} "
                f"({res_b['confidence'] * 100:.1f}%)."
            )

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(f"#### 🏛️ {m1_display}")
            st.markdown(
                f"""
                <div class="emotion-card">
                    <div style="font-size: 1.4rem; font-weight: bold; color: {EMOTION_COLORS.get(res_a['label'], '#FFF')};">
                        {EMOTION_EMOJIS.get(res_a['label'], '')} {res_a['label'].upper()}
                    </div>
                    <div>Confidence: <strong>{res_a['confidence'] * 100:.2f}%</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            df_a = pd.DataFrame([
                {"Emotion": f"{EMOTION_EMOJIS[l]} {l.capitalize()}", "Probability": res_a["all_probs"][l]}
                for l in LABELS
            ])
            st.bar_chart(df_a.set_index("Emotion"), height=220, use_container_width=True)

        with col_right:
            st.markdown(f"#### 🤖 {m2_display}")
            st.markdown(
                f"""
                <div class="emotion-card">
                    <div style="font-size: 1.4rem; font-weight: bold; color: {EMOTION_COLORS.get(res_b['label'], '#FFF')};">
                        {EMOTION_EMOJIS.get(res_b['label'], '')} {res_b['label'].upper()}
                    </div>
                    <div>Confidence: <strong>{res_b['confidence'] * 100:.2f}%</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            df_b = pd.DataFrame([
                {"Emotion": f"{EMOTION_EMOJIS[l]} {l.capitalize()}", "Probability": res_b["all_probs"][l]}
                for l in LABELS
            ])
            st.bar_chart(df_b.set_index("Emotion"), height=220, use_container_width=True)

        # Delta Table
        st.markdown(f"#### Probability Differential ({m2_display} − {m1_display})")
        diff_records = []
        for l in LABELS:
            p_a = res_a["all_probs"][l]
            p_b = res_b["all_probs"][l]
            diff_records.append({
                "Emotion": f"{EMOTION_EMOJIS[l]} {l.capitalize()}",
                f"{m1_display} Prob": round(p_a, 4),
                f"{m2_display} Prob": round(p_b, 4),
                f"Delta ({m2_display} − {m1_display})": round(p_b - p_a, 4),
            })
        st.dataframe(pd.DataFrame(diff_records), use_container_width=True)


# ==============================================================================
# Tab 4: Batch CSV Inference
# ==============================================================================
def render_batch_tab(selected_model_type: str) -> None:
    """Render Tab 4: Batch CSV upload, validation, progress bar, and export."""
    st.markdown("### 📁 Batch Prediction via CSV")
    st.markdown(
        "Upload a `.csv` file containing a **`text`** column. The app will validate "
        "the structure, run batched inference with a progress indicator, and provide a downloadable results file."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file:",
        type=["csv"],
        help="The CSV file must contain a 'text' column header.",
    )

    # Sample CSV download helper
    sample_csv_text = "text\n" \
                      "I am so excited for the project release tomorrow!\n" \
                      "I feel very sad and dejected by the bad news.\n" \
                      "That driver cut me off and I got so angry!\n" \
                      "I am scared of walking in the woods at night.\n" \
                      "I never expected to win the lottery ticket!\n"

    st.download_button(
        label="📄 Download Sample Batch CSV Template",
        data=sample_csv_text,
        file_name="sample_batch_input.csv",
        mime="text/csv",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as read_err:
            st.error(f"❌ Failed to parse uploaded CSV file: {read_err}")
            return

        # Column validation (case-insensitive)
        text_col: Optional[str] = None
        for col in df.columns:
            if str(col).strip().lower() == "text":
                text_col = str(col)
                break

        if text_col is None:
            st.error(
                f"❌ Invalid CSV schema! The file must contain a **'text'** column. "
                f"Detected columns: `{list(df.columns)}`"
            )
            return

        if text_col != "text":
            df = df.rename(columns={text_col: "text"})

        st.success(f"✅ CSV successfully validated! Loaded **{len(df):,}** total rows.")
        st.dataframe(df.head(5), use_container_width=True)

        # Check for empty entries
        raw_texts = df["text"].tolist()
        valid_indices: list[int] = []
        valid_texts: list[str] = []
        for i, val in enumerate(raw_texts):
            if val is not None and str(val).strip():
                valid_indices.append(i)
                valid_texts.append(str(val).strip())

        empty_count = len(df) - len(valid_texts)
        if empty_count > 0:
            st.warning(
                f"⚠️ Note: {empty_count} row(s) contain empty or blank text and will be skipped."
            )

        if not valid_texts:
            st.warning("⚠️ No valid non-empty text rows found to classify.")
            return

        if st.button("⚡ Run Batch Predictions", type="primary"):
            predictor = safe_get_predictor(selected_model_type)
            if predictor is None:
                return

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            chunk_size = 32
            total_items = len(valid_texts)
            predictions_list: list[dict[str, Any]] = []

            try:
                for start_idx in range(0, total_items, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_items)
                    chunk = valid_texts[start_idx:end_idx]
                    status_text.text(f"Processing items {start_idx + 1} to {end_idx} of {total_items}...")
                    chunk_preds = predictor.predict_batch(chunk)
                    predictions_list.extend(chunk_preds)
                    progress_bar.progress(end_idx / total_items)
                status_text.text(f"Completed inference for all {total_items} samples!")
            except Exception as batch_err:
                st.error(f"⚠️ Error during batch inference: {batch_err}")
                return

            # Assemble result dataframe
            results_df = df.copy()
            results_df["predicted_emotion"] = "N/A"
            results_df["confidence"] = 0.0

            for l in LABELS:
                results_df[f"prob_{l}"] = 0.0

            for idx, p_res in zip(valid_indices, predictions_list):
                results_df.at[idx, "predicted_emotion"] = p_res["label"]
                results_df.at[idx, "confidence"] = p_res["confidence"]
                for l in LABELS:
                    results_df.at[idx, f"prob_{l}"] = p_res["all_probs"][l]

            st.markdown("---")
            st.markdown("#### Batch Prediction Results")
            st.dataframe(results_df.head(20), use_container_width=True)

            csv_export = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Batch Predictions CSV",
                data=csv_export,
                file_name=f"{selected_model_type}_batch_predictions.csv",
                mime="text/csv",
                type="primary",
            )


# ==============================================================================
# Tab 5: Experimental Results & Reports
# ==============================================================================
def render_results_tab() -> None:
    """Render Tab 5: Model comparison table, confusion matrices, and training curves."""
    st.markdown("### 📈 Experimental Results & Evaluation Reports")
    st.markdown(
        "Review benchmark comparison tables, test set confusion matrices, per-class F1 scores, "
        "and training loss curves generated across project evaluation runs."
    )

    # 1. Comparison Table
    st.markdown("#### 1. Model Performance Benchmark")
    comp_csv = METRICS_DIR / "comparison_table.csv"
    comp_md = METRICS_DIR / "comparison_table.md"

    if comp_csv.exists():
        try:
            df_comp = pd.read_csv(comp_csv)
            st.dataframe(df_comp, use_container_width=True)
        except Exception:
            if comp_md.exists():
                st.markdown(comp_md.read_text(encoding="utf-8"))
    else:
        # Construct summary from baseline.json and distilbert.json/distilbert_smoke.json
        base_m = load_json_metrics(METRICS_DIR / "baseline.json")
        dist_m = load_json_metrics(METRICS_DIR / "distilbert.json")
        if dist_m is None:
            dist_m = load_json_metrics(METRICS_DIR / "distilbert_smoke.json")
        v2_m = load_json_metrics(METRICS_DIR / "distilbert_v2.json")

        if base_m or dist_m or v2_m:
            rows = []
            if base_m:
                rows.append({
                    "Model": "Baseline (TF-IDF + LR)",
                    "Accuracy": f"{base_m.get('accuracy', 0):.4f}",
                    "Macro-F1": f"{base_m.get('macro_f1', 0):.4f}",
                    "Weighted-F1": f"{base_m.get('weighted_f1', 0):.4f}",
                    "Inference (ms/1k)": f"{base_m.get('inference_time_per_1000_samples_ms', 0):.2f}",
                })
            if dist_m:
                is_smoke = "subset" in dist_m
                lbl = "DistilBERT (Smoke)" if is_smoke else "DistilBERT v1"
                rows.append({
                    "Model": lbl,
                    "Accuracy": f"{dist_m.get('accuracy', 0):.4f}",
                    "Macro-F1": f"{dist_m.get('macro_f1', 0):.4f}",
                    "Weighted-F1": f"{dist_m.get('weighted_f1', 0):.4f}",
                    "Inference (ms/1k)": f"{dist_m.get('inference_time_per_1000_samples_ms', 0):.2f}",
                })
            if v2_m:
                rows.append({
                    "Model": "DistilBERT v2 (Improved)",
                    "Accuracy": f"{v2_m.get('accuracy', 0):.4f}",
                    "Macro-F1": f"{v2_m.get('macro_f1', 0):.4f}",
                    "Weighted-F1": f"{v2_m.get('weighted_f1', 0):.4f}",
                    "Inference (ms/1k)": f"{v2_m.get('inference_time_per_1000_samples_ms', 0):.2f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption("ℹ️ Run `.venv/bin/python -m src.compare` to compile the full comparison table.")
        else:
            st.info("ℹ️ Benchmark comparison table not yet generated. Run training and evaluation first.")

    st.markdown("---")

    # 2. Confusion Matrices
    st.markdown("#### 2. Test Set Confusion Matrices")
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("**Baseline Confusion Matrix**")
        base_conf = FIGURES_DIR / "baseline_confusion.png"
        if base_conf.exists():
            st.image(str(base_conf), caption="Baseline (TF-IDF + Logistic Regression)", use_container_width=True)
        else:
            st.info(f"ℹ️ {base_conf.name} not found. Run `python -m src.baseline` to generate.")

    with c_col2:
        st.markdown("**DistilBERT Confusion Matrix**")
        dist_conf = FIGURES_DIR / f"{ACTIVE_DISTILBERT}_confusion.png"
        if not dist_conf.exists():
            dist_conf = FIGURES_DIR / "distilbert_confusion.png"

        if dist_conf.exists():
            st.image(str(dist_conf), caption=f"DistilBERT ({dist_conf.stem})", use_container_width=True)
        else:
            st.info(f"ℹ️ {dist_conf.name} not found. Run `python -m src.transformer` to generate.")

    st.markdown("---")

    # 3. Per-Class F1 & Training Curves
    st.markdown("#### 3. Per-Class F1 Scores & Training Curves")
    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.markdown("**Per-Class F1 Scores**")
        compare_f1 = FIGURES_DIR / "compare_per_class_f1.png"
        base_f1 = FIGURES_DIR / "baseline_per_class_f1.png"
        target_f1 = compare_f1 if compare_f1.exists() else base_f1

        if target_f1.exists():
            st.image(str(target_f1), caption=f"Per-Class F1 ({target_f1.name})", use_container_width=True)
        else:
            st.info("ℹ️ Per-class F1 plot not found.")

    with f_col2:
        st.markdown("**DistilBERT Training & Validation Loss Curves**")
        curves = FIGURES_DIR / f"{ACTIVE_DISTILBERT}_training_curves.png"
        if not curves.exists():
            curves = FIGURES_DIR / "distilbert_training_curves.png"

        if curves.exists():
            st.image(str(curves), caption="Loss & Macro-F1 Progression", use_container_width=True)
        else:
            st.info("ℹ️ Training curves figure not found.")

    # 4. Error Analysis Details
    error_md = METRICS_DIR / "error_analysis.md"
    if error_md.exists():
        st.markdown("---")
        with st.expander("🔍 Detailed Model Error Analysis"):
            st.markdown(error_md.read_text(encoding="utf-8"))


# ==============================================================================
# Main Application Entrypoint
# ==============================================================================
def main() -> None:
    """Initialize page configuration, layout tabs, and application flow."""
    st.set_page_config(
        page_title="Emotion Detection in Text",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_styles()

    st.markdown('<div class="main-title">🎭 Emotion Detection in Text</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Classical Machine Learning vs. Transformer Architecture with Explainable AI</div>',
        unsafe_allow_html=True,
    )

    # Render Sidebar
    selected_model_type = render_sidebar()

    # Layout Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Analyze",
        "🔬 Explain",
        "⚖️ Compare Models",
        "📁 Batch",
        "📈 Results",
    ])

    with tab1:
        render_analyze_tab(selected_model_type)

    with tab2:
        render_explain_tab(selected_model_type)

    with tab3:
        render_compare_tab()

    with tab4:
        render_batch_tab(selected_model_type)

    with tab5:
        render_results_tab()


if __name__ == "__main__":
    main()
