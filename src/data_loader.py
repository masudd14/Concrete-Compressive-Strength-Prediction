"""Data ingestion and schema mapping for UCI Concrete Compressive Strength dataset."""

from pathlib import Path
from typing import Tuple, List
import pandas as pd

RAW_COLUMNS: List[str] = [
    "Cement (component 1)(kg in a m^3 mixture)",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
    "Fly Ash (component 3)(kg in a m^3 mixture)",
    "Water  (component 4)(kg in a m^3 mixture)",
    "Superplasticizer (component 5)(kg in a m^3 mixture)",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)",
    "Age (day)",
    "Concrete compressive strength(MPa, megapascals) ",
]

CLEAN_FEATURE_NAMES: List[str] = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]

TARGET_NAME: str = "compressive_strength"


def load_concrete_data(data_path: str) -> pd.DataFrame:
    """Load dataset from Excel or CSV file.

    Args:
        data_path: Path to dataset file (.csv or .xls/.xlsx).

    Returns:
        pd.DataFrame containing 1,030 mix records.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    if path.suffix in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate input features and compressive strength target."""
    target_col = [c for c in df.columns if "strength" in c.lower() or c == TARGET_NAME][0]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
