import os
import pandas as pd

IN_CSV = "../model/dataset/merge_csv_samples_20240809_with_categories.csv"
OUT_CSV = "../model/dataset/merge_csv_samples_20240809_filtered_categories.csv"

RANDOM_STATE = 42
BENIGN_LABEL = "benign"
MAX_BENIGN_SAMPLES = 5000  # change if needed


def main():
    df = pd.read_csv(IN_CSV)

    if "categories" not in df.columns:
        raise ValueError("Column 'categories' not found in input CSV")

    # Safety strip
    df["categories"] = df["categories"].astype(str).str.strip()

    # Remove duplicates
    before_dup = len(df)
    df = df.drop_duplicates().copy()
    after_dup = len(df)
    print(f"Removed duplicate rows: {before_dup - after_dup}")

    # 1) Drop other_malware
    df = df[df["categories"] != "other_malware"].copy()

    # 2) Drop categories with count < 600
    cat_counts = df["categories"].value_counts()
    keep_cats = cat_counts[cat_counts >= 600].index
    df = df[df["categories"].isin(keep_cats)].copy()

    # 3) Randomly reduce benign rows if too dominant
    if BENIGN_LABEL in df["categories"].unique():
        benign_df = df[df["categories"] == BENIGN_LABEL].copy()
        other_df = df[df["categories"] != BENIGN_LABEL].copy()

        if len(benign_df) > MAX_BENIGN_SAMPLES:
            benign_df = benign_df.sample(
                n=MAX_BENIGN_SAMPLES,
                random_state=RANDOM_STATE
            )
            print(f"Downsampled '{BENIGN_LABEL}' to {MAX_BENIGN_SAMPLES} rows")

        df = pd.concat([benign_df, other_df], axis=0)
        df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print("\nFinal counts per category (kept):")
    print(df["categories"].value_counts().to_string())

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print("\nSaved:", OUT_CSV)


if __name__ == "__main__":
    main()