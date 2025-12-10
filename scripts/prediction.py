import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import shap
from typing import Dict, Any, Tuple

# --- 1. Evaluation Functions ---

def evaluate_regression_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluates a regression model using RMSE and R-squared.
    """
    y_pred = model.predict(X_test)
    
    # Ensure no negative predictions (claims cannot be negative)
    y_pred[y_pred < 0] = 0

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {'RMSE': rmse, 'R-squared': r2}

def evaluate_all_models(models: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluates all trained models and returns a performance DataFrame.
    """
    results = {}
    for name, model in models.items():
        results[name] = evaluate_regression_model(model, X_test, y_test)
        
    return pd.DataFrame(results).T


# --- 2. Feature Importance / Interpretation (SHAP) ---

def explain_model_predictions(model_pipeline: Pipeline, X_test: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
    """
    Uses SHAP to analyze feature importance for the best performing model.
    """
    # Isolate the regressor (the last step in the pipeline)
    model = model_pipeline.named_steps['regressor']
    
    # Get preprocessed data for SHAP Explainer
    X_test_preprocessed = model_pipeline.named_steps['preprocessor'].transform(X_test)
    
    # Get feature names after one-hot encoding
    feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    # Use TreeExplainer for tree-based models (assuming XGBoost/RF will be best)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_preprocessed)
    
    return shap_values, feature_names


# --- 3. Full Risk-Based Premium Prediction Framework ---

def calculate_risk_premium(prob_model: Pipeline, sev_model: Pipeline, X_data: pd.DataFrame, expense_load: float = 0.1, profit_margin: float = 0.05) -> pd.Series:
    """
    Calculates the Risk-Based Premium using the advanced formula.
    Premium = (Predicted Probability of Claim * Predicted Claim Severity) + Expense Loading + Profit Margin
    """
    # 1. Predict Probability of Claim (0 to 1)
    # We use predict_proba and take the probability of the positive class (index 1)
    prob_claim = prob_model.predict_proba(X_data)[:, 1]
    
    # 2. Predict Claim Severity (Amount)
    pred_severity = sev_model.predict(X_data)
    pred_severity[pred_severity < 0] = 0 # Ensure no negative severity

    # 3. Calculate Pure Premium
    pure_premium = prob_claim * pred_severity

    # 4. Calculate final Risk-Based Premium
    # Note: Expense and Profit are added as a percentage of Pure Premium here for simplicity
    loading_factor = 1 + expense_load + profit_margin
    risk_premium = pure_premium * loading_factor

    return pd.Series(risk_premium, index=X_data.index)