"""
app.py
Streamlit web UI for the Fake News Detector.

Run with:
    streamlit run app.py

Make sure you've already trained a model first:
    python train.py
"""

import os
import joblib
import streamlit as st
from utils import clean_text

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)

MODEL_PATH = "model/model.joblib"
VECTORIZER_PATH = "model/vectorizer.joblib"


@st.cache_resource
def load_pipeline():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    confidence = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        confidence = max(proba)

    elif hasattr(model, "decision_function"):
        score = model.decision_function(vec)[0]
        confidence = 1 / (1 + pow(2.718281828, -abs(score)))

    return pred, confidence


st.title("📰 Fake News Detector")
st.write(
    "Paste a headline or article snippet below and check whether it looks REAL or FAKE."
)

if not (
    os.path.exists(MODEL_PATH)
    and os.path.exists(VECTORIZER_PATH)
):
    st.error(
        "No trained model found. Please run `python train.py` first, then restart this app."
    )
    st.stop()

model, vectorizer = load_pipeline()

text_input = st.text_area(
    "News text",
    height=150,
    placeholder="e.g. Scientists confirm new vaccine reduces flu cases by 40% in clinical trial...",
)

col1, col2 = st.columns([1, 4])

with col1:
    analyze_clicked = st.button("Analyze", type="primary")

if analyze_clicked:

    if not text_input.strip():
        st.warning("Please enter some text first.")

    else:
        pred, conf = predict(text_input, model, vectorizer)

        if pred == "FAKE":
            st.error("### 🚩 Prediction: FAKE")
        else:
            st.success("### ✅ Prediction: REAL")

        if conf is not None:
            st.metric("Confidence", f"{conf:.1%}")
            st.progress(min(max(conf, 0.0), 1.0))

        st.caption(
            "Note: This model is trained on the Fake and Real News dataset."
        )

st.divider()

st.markdown("""
<div style="text-align: center; padding: 20px;">

<h3>📊 Dataset Information</h3>

<p>
<b>Fake.csv</b> → 23,481 articles<br>
<b>True.csv</b> → 21,417 articles<br>
<b>Total Dataset</b> → 44,898 articles
</p>

<hr style="width:60%; margin:auto;">

<p>
<b>Developed by</b><br>
<b>Priya Kale</b><br>
📧 pkale3922@gmail.com
</p>

</div>
""", unsafe_allow_html=True)