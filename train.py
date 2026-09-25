"""
train.py
Trains a Fake News Detector using TF-IDF + Logistic Regression
(and a PassiveAggressiveClassifier for comparison).

Usage:
    python train.py                          # uses data/news_sample.csv
    python train.py --data path/to/your.csv  # use your own dataset
                                               # (must have 'text' and 'label' columns,
                                               #  label values: REAL / FAKE)

To get much higher real-world accuracy, download the popular Kaggle
"Fake and Real News Dataset" (Fake.csv + True.csv), combine them into
one CSV with columns text,label and pass it via --data.
"""

import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import clean_text


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].str.upper().str.strip()
    df = df[df["label"].isin(["REAL", "FAKE"])]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/news_sample.csv",
                         help="Path to CSV with 'text' and 'label' columns")
    parser.add_argument("--test_size", type=float, default=0.25)
    args = parser.parse_args()

    print(f"Loading data from {args.data} ...")
    df = load_data(args.data)
    print(f"Loaded {len(df)} rows | REAL={sum(df.label=='REAL')} FAKE={sum(df.label=='FAKE')}")

    print("Cleaning text ...")
    df["clean_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=args.test_size, random_state=42, stratify=df["label"]
    )

    print("Vectorizing (TF-IDF) ...")
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "PassiveAggressive": SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1",
                                            eta0=1.0, max_iter=1000, random_state=42),
    }

    best_model, best_name, best_acc = None, None, -1
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test, preds, zero_division=0))
        print("Confusion matrix [rows=true, cols=pred] order=FAKE,REAL:")
        print(confusion_matrix(y_test, preds, labels=["FAKE", "REAL"]))

        if acc > best_acc:
            best_model, best_name, best_acc = model, name, acc

    print(f"\nBest model: {best_name} (accuracy={best_acc:.3f})")

    joblib.dump(best_model, "model/model.joblib")
    joblib.dump(vectorizer, "model/vectorizer.joblib")
    print("Saved model/model.joblib and model/vectorizer.joblib")


if __name__ == "__main__":
    main()
