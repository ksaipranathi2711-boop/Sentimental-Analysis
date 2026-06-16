"""
app.py — Streamlit Sentiment Analysis Tool
Run locally: streamlit run app.py
Deploy on Streamlit Cloud: point it at this repo / app.py.
"""

import os
import pickle
import streamlit as st

from train_model import clean_text, main as train_main

st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-header {
            background: linear-gradient(90deg,#4f46e5 0%, #9333ea 50%, #ec4899 100%);
            padding: 2rem 1.5rem; border-radius: 14px; color: white;
            text-align: center; margin-bottom: 1.5rem;
            box-shadow: 0 8px 24px rgba(79,70,229,.25);
        }
        .main-header h1 { margin:0; font-size: 2.3rem; }
        .main-header p  { margin:.4rem 0 0; opacity:.95; }
        .result-card {
            padding: 1.4rem 1.6rem; border-radius: 14px; color:white;
            margin-top: 1rem; box-shadow: 0 6px 18px rgba(0,0,0,.15);
        }
        .result-card h2 { margin: 0 0 .3rem 0; }
        .pos { background: linear-gradient(135deg,#16a34a,#22c55e); }
        .neg { background: linear-gradient(135deg,#dc2626,#ef4444); }
        .neu { background: linear-gradient(135deg,#2563eb,#3b82f6); }
        .footer {
            text-align:center; padding: 1rem; margin-top: 2rem;
            color:#6b7280; font-size:.9rem; border-top:1px solid #e5e7eb;
        }
        .stTextArea textarea { font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>💬 Sentiment Analysis Tool</h1>
        <p>NLP • TF-IDF • Machine Learning — Positive / Negative / Neutral</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This tool classifies text as **Positive**, **Negative**, or **Neutral** "
        "using a TF-IDF vectorizer and a Logistic Regression model trained on "
        "labeled review-style data."
    )
    st.markdown("---")
    st.subheader("🧠 Pipeline")
    st.markdown(
        "- Load dataset\n"
        "- Clean & lowercase text\n"
        "- Remove stopwords\n"
        "- TF-IDF vectorization\n"
        "- Logistic Regression classifier\n"
        "- Save `model.pkl` & `vectorizer.pkl`"
    )
    st.markdown("---")
    st.caption("AI/ML Internship Submission")

# ---------------------------------------------------------------------------
# Load (or train) model
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")
VEC_PATH = os.path.join(HERE, "vectorizer.pkl")


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH)):
        # Auto-train on first run (e.g. on Streamlit Cloud).
        train_main()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


model, vectorizer = load_artifacts()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.subheader("✍️ Enter text to analyze")
text = st.text_area(
    "Your text",
    height=180,
    placeholder="Type or paste a review, tweet, comment, etc...",
    label_visibility="collapsed",
)

col_btn, _ = st.columns([1, 4])
with col_btn:
    analyze = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

if analyze:
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing sentiment..."):
            cleaned = clean_text(text)
            vec = vectorizer.transform([cleaned if cleaned else text.lower()])
            pred = model.predict(vec)[0]
            probs = model.predict_proba(vec)[0]
            classes = list(model.classes_)
            confidence = float(probs[classes.index(pred)]) * 100

        css_class = {"positive": "pos", "negative": "neg", "neutral": "neu"}[pred]
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}[pred]

        st.markdown(
            f"""
            <div class="result-card {css_class}">
                <h2>{emoji} {pred.capitalize()}</h2>
                <p style="margin:.2rem 0 0;font-size:1.05rem;">
                    Confidence: <strong>{confidence:.2f}%</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📊 Class probabilities")
        prob_cols = st.columns(len(classes))
        for c, p, col in zip(classes, probs, prob_cols):
            with col:
                st.metric(label=c.capitalize(), value=f"{p*100:.1f}%")
                st.progress(float(p))

st.markdown(
    '<div class="footer">Built with ❤️ using Streamlit, scikit-learn & TF-IDF · '
    "Sentiment Analysis Tool © 2026</div>",
    unsafe_allow_html=True,
)
