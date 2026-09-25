"""
predict.py
Load the trained model and classify new headlines/articles as REAL or FAKE.

Usage:
    python predict.py "Some headline or article text here"
    python predict.py               # launches interactive mode
"""

import sys
import joblib
from utils import clean_text


def load_pipeline():
    model = joblib.load("model/model.joblib")
    vectorizer = joblib.load("model/vectorizer.joblib")
    return model, vectorizer


def predict(text: str, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        confidence = max(proba)
    elif hasattr(model, "decision_function"):
        score = model.decision_function(vec)[0]
        confidence = 1 / (1 + pow(2.718281828, -abs(score)))  # sigmoid approx

    return pred, confidence


def main():
    model, vectorizer = load_pipeline()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        pred, conf = predict(text, model, vectorizer)
        conf_str = f" (confidence: {conf:.2%})" if conf is not None else ""
        print(f"\nText: {text}\nPrediction: {pred}{conf_str}\n")
    else:
        print("Interactive mode. Type a headline/article and press Enter (or 'quit' to exit).\n")
        while True:
            text = input(">> ")
            if text.strip().lower() in ("quit", "exit"):
                break
            pred, conf = predict(text, model, vectorizer)
            conf_str = f" (confidence: {conf:.2%})" if conf is not None else ""
            print(f"Prediction: {pred}{conf_str}\n")


if __name__ == "__main__":
    main()
