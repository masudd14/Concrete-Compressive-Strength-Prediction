"""Production Machine Learning Pipeline definition."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
    VotingRegressor
)
from src.features import ConcreteFeatureEngineering


def build_ensemble_pipeline() -> Pipeline:
    """Instantiate production VotingRegressor pipeline.

    Combines top performing tree-based regressors with domain feature engineering.
    """
    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    hgbr = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.05, random_state=42)
    et = ExtraTreesRegressor(n_estimators=200, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42)

    voting = VotingRegressor(
        estimators=[
            ("gbr", gbr),
            ("hgbr", hgbr),
            ("et", et),
            ("rf", rf),
        ],
        weights=[3, 3, 2, 1],
    )

    full_pipeline = Pipeline([
        ("feature_engineering", ConcreteFeatureEngineering()),
        ("scaler", StandardScaler()),
        ("ensemble", voting),
    ])

    return full_pipeline
