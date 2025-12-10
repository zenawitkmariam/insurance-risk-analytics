import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple
from sklearn.impute import SimpleImputer

def preprocess_data(df: pd.DataFrame, target_col: str, claims_only: bool = False) -> Tuple[pd.DataFrame, pd.Series, Pipeline]:
    """
    Final robust preprocessing function for model training.

    Includes fixes for: Whitespace/string errors, missing column type assignment,
    and MemoryError due to dense One-Hot Encoding.
    """
    df_working = df.copy()

    # 1. Claims-Only Filtering for Severity Model
    if claims_only:
        df_working = df_working[df_working['HasClaim'] == 1].copy()
        
    # --- CRITICAL FIX 1: HANDLE WHITESPACE STRINGS ---
    # Replaces any string consisting only of whitespace with a true NaN value.
    df_working = df_working.replace(r'^\s*$', np.nan, regex=True)

    # 2. Select Features (Excluding IDs, targets, dates, and problematic columns)
    features_to_exclude = [
        # IDs/Keys (High Cardinality, no predictive value)
        'PolicyNumber', 'ClientNumber', 'ClaimNumber', 'UnderwrittenCoverID', 'PolicyID',
        # Targets/Engineered Metrics
        'TotalPremium', 'Margin', 'HasClaim', 'CalculatedPremiumPerTerm', target_col,
        # Date/Time Columns (Excluded to prevent string conversion errors)
        'RegistrationDate', 'QuoteDate', 'ClaimDate', 'TransactionDate', 'DateOfBirth',
        # Columns 100% Missing in Training Set (Fixes UserWarning)
        'CrossBorder' 
    ]
    
    # Identify final feature set (X)
    X = df_working.drop(columns=[col for col in features_to_exclude if col in df_working.columns], errors='ignore')
    y = df_working[target_col]

    # 3. Define Feature Types (Crucial for correct Imputer usage)
    
    # Explicitly list categorical columns to ensure object columns like 'Citizenship' 
    # are routed away from the numerical imputer.
    explicit_categorical_features = [
        'Province', 'PostalCode', 'Gender', 'CoverType', 'Colour', 
        'VehicleType', 'Citizenship', 'LegalType', 'Title', 'Language', 
        'BankAccountType', 'MaritalStatus', 'CoverCategory', 'CoverGroup', 
        'Section', 'Product', 'StatutoryClass', 'StatutoryRiskType'
    ]
    
    # List of all columns in X that are numerical types
    potential_numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    
    # Final list of numerical features (excluding any categorical columns that might have slipped)
    numerical_features = [
        col for col in potential_numerical_features 
        if col not in explicit_categorical_features
    ]
    
    # Final list of categorical features is the intersection of our list and columns present in X
    categorical_features = [col for col in explicit_categorical_features if col in X.columns]
    
    # --- CRITICAL FIX 2: COERCE NUMERICAL COLUMNS ---
    # Forces columns that should be numerical to the correct type, converting errors to NaN.
    for col in numerical_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # 4. Define Preprocessing Pipeline Steps
    
    # Numerical Pipeline: Impute with median, then scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Pipeline: Impute with mode, then One-Hot Encode (sparse output for memory)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)) # <--- MEMORY FIX
    ])

    # 5. Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' 
    )
    
    # 6. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if target_col == 'HasClaim' else None
    )

    return X_train, X_test, y_train, y_test, preprocessor

# Example Usage:
# from data_preparation import preprocess_data
# X_train_sev, X_test_sev, y_train_sev, y_test_sev, preprocessor_sev = preprocess_data(df_clean, 'TotalClaims', claims_only=True)
# X_train_prob, X_test_prob, y_train_prob, y_test_prob, preprocessor_prob = preprocess_data(df_clean, 'HasClaim', claims_only=False)