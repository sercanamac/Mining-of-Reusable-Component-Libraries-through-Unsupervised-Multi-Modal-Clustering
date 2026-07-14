"""Feature preprocessing for BIM clustering."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import RobustScaler

FEATURE_COLS = ['Length', 'CrossSectionArea', 'Volume', 'NumVertices']


def load_features(path: str | Path) -> pd.DataFrame:
    """Load features from CSV or parquet file.

    Parameters
    ----------
    path : str or Path
        Path to feature file (.csv or .parquet)

    Returns
    -------
    pd.DataFrame
        Features dataframe with columns: Length, CrossSectionArea, Volume,
        NumVertices, GlobalId, IfcType, Split
    """
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None
) -> Tuple[np.ndarray, list[str], RobustScaler]:
    """Extract and scale raw features for clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Features dataframe
    feature_cols : list[str], optional
        Columns to use as features. Defaults to FEATURE_COLS.

    Returns
    -------
    X_scaled : np.ndarray
        Scaled feature matrix (n_samples, n_features)
    feature_names : list[str]
        Names of features used
    scaler : RobustScaler
        Fitted scaler for inverse transform
    """
    cols = feature_cols or FEATURE_COLS
    X = df[cols].values
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, cols, scaler
