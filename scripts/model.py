import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from typing import Dict, Any

# --- 1. Claim Severity Regression Models ---

def build_severity_models(preprocessor: Pipeline) -> Dict[str, Pipeline]:
    """
    Builds and trains Claim Severity regression models (TotalClaims).
    
    Args:
        preprocessor: The ColumnTransformer Pipeline for feature preparation.

    Returns:
        A dictionary of trained model pipelines.
    """
    
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10),
        'XGBoostRegressor': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1, max_depth=5)
    }

    pipelines = {}
    for name, model in models.items():
        # Create a complete pipeline: Preprocessing -> Model
        pipelines[name] = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
    return pipelines


def train_models(pipelines: Dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
    """
    Trains a dictionary of models and returns the trained pipelines.
    """
    print(f"\n--- Starting Training on {len(X_train):,} samples ---")
    trained_pipelines = {}
    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
    return trained_pipelines

# Example Usage:
# pipelines_sev = build_severity_models(preprocessor_sev)
# trained_sev_models = train_models(pipelines_sev, X_train_sev, y_train_sev)