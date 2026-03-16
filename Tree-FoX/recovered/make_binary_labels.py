import pandas as pd


def to_binary_label(cat: str) -> int:
    cat = str(cat).strip().lower()
    return 0 if cat == "benign" else 1


def add_binary_label_column(df: pd.DataFrame, cat_col="categories", out_col="binary_label"):
    if cat_col not in df.columns:
        raise ValueError(f"Missing column: {cat_col}")
    df = df.copy()
    df[out_col] = df[cat_col].apply(to_binary_label)
    return df