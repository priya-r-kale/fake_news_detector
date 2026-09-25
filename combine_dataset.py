"""
combine_dataset.py
Combines Kaggle's Fake.csv and True.csv into the text,label CSV format
that train.py expects.
"""

import argparse
import pandas as pd


def combine_col(df: pd.DataFrame, source_name: str) -> pd.Series:
    if "text" not in df.columns:
        raise ValueError(f"Expected a 'text' column in {source_name}, found: {list(df.columns)}")
    if "title" in df.columns:
        return (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    return df["text"].fillna("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", default="data/Fake.csv", help="Path to Kaggle Fake.csv")
    parser.add_argument("--real", default="data/True.csv", help="Path to Kaggle True.csv")
    parser.add_argument("--out", default="data/full_news.csv", help="Output combined CSV path")
    args = parser.parse_args()

    print(f"Reading {args.fake} ...")
    fake = pd.read_csv(args.fake)
    print(f"Reading {args.real} ...")
    real = pd.read_csv(args.real)

    fake_df = pd.DataFrame({"text": combine_col(fake, args.fake), "label": "FAKE"})
    real_df = pd.DataFrame({"text": combine_col(real, args.real), "label": "REAL"})

    combined = pd.concat([fake_df, real_df], ignore_index=True)
    combined = combined[combined["text"].str.strip() != ""]
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    combined.to_csv(args.out, index=False)
    print(f"\nSaved {len(combined)} rows to {args.out}")
    print(f"  FAKE: {len(fake_df)} | REAL: {len(real_df)}")
    print(f"\nNow train with:\n  python train.py --data {args.out}")


if __name__ == "__main__":
    main()