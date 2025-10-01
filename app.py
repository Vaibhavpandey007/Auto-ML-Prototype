import sklearn
import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
import shap
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple, Dict, Any
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# --------------------------- Streamlit Utility ---------------------------

def load_data(uploaded_file) -> pd.DataFrame:
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
        else:
            return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()


def infer_task(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        return 'regression'
    else:
        return 'classification'


import sklearn
from sklearn.preprocessing import OneHotEncoder

def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Version check for compatibility
    if sklearn.__version__ >= "1.2":
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', onehot)
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ])

    return preprocessor, numeric_features, categorical_features

# --------------------------- Optuna Tuning ---------------------------

def optuna_tuning(model, param_space, X, y, task, n_trials=20):
    def objective(trial):
        params = {key: trial.suggest_categorical(key, vals) for key, vals in param_space.items()}
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model.set_params(**params))
        ])
        if task == 'classification':
            score = cross_val_score(pipeline, X, y, cv=3, scoring='f1_macro').mean()
        else:
            score = cross_val_score(pipeline, X, y, cv=3, scoring='r2').mean()
        return score
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# --------------------------- SHAP Explainability ---------------------------

import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def explain_model(best_pipeline, X_sample, y_sample=None):
    try:
        # Preprocess the sample data
        X_processed = best_pipeline['preprocessor'].transform(X_sample)
        feature_names = best_pipeline['preprocessor'].get_feature_names_out()
        X_processed = pd.DataFrame(X_processed, columns=feature_names)

        st.subheader("Feature Importance (SHAP)")

        # Create matplotlib figure
        fig, ax = plt.subplots()

        # Use general SHAP Explainer
        explainer = shap.Explainer(best_pipeline['model'], X_processed)
        shap_values = explainer(X_processed)

        shap.summary_plot(shap_values, X_processed, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"SHAP explainability failed: {e}")
        st.info("Falling back to permutation feature importance...")

        # Fallback: permutation importance
        if y_sample is not None:
            result = permutation_importance(best_pipeline['model'], X_processed, y_sample,
                                            n_repeats=5, random_state=42, n_jobs=-1)
            importances = pd.Series(result.importances_mean, index=feature_names)
            importances = importances.sort_values(ascending=False)

            fig, ax = plt.subplots()
            importances.plot(kind='bar', ax=ax)
            ax.set_title("Feature Importance (Permutation)")
            st.pyplot(fig)
# --------------------------- Streamlit App ---------------------------

st.set_page_config(page_title='SmartML Pipeline Builder', layout='wide')
st.title('🧠 SmartML Pipeline Builder — AutoML Prototype (Enhanced)')

uploaded_file = st.file_uploader('Upload CSV/Excel file', type=['csv', 'xls', 'xlsx'])

if uploaded_file:
    df = load_data(uploaded_file)
    target_col = st.selectbox('Select target column', df.columns)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    task = infer_task(y)
    st.info(f"Task inferred: {task}")

    preprocessor, num_features, cat_features = build_preprocessor(X)
    if task == 'classification' and not pd.api.types.is_numeric_dtype(y):
        y = LabelEncoder().fit_transform(y.astype(str))

    # Candidate model (example: RandomForest)
    if task == 'classification':
        base_model = RandomForestClassifier()
        param_space = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]}
    else:
        base_model = RandomForestRegressor()
        param_space = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]}

    # Optuna tuning
    best_params = optuna_tuning(base_model, param_space, X, y, task)
    st.write("Best hyperparameters:", best_params)

    best_model = base_model.set_params(**best_params)
    best_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', best_model)
    ])
    best_pipeline.fit(X, y)

    st.success("Best model fitted.")
    explain_model(best_pipeline, X.sample(min(100, len(X))))

# --------------------------- FastAPI Server ---------------------------

app = FastAPI(title="SmartML AutoML API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/automl")
async def automl_endpoint(file: UploadFile = File(...), target: str = None):
    try:
        df = pd.read_csv(file.file)
        if target not in df.columns:
            return JSONResponse(content={"error": "Invalid target column"}, status_code=400)

        y = df[target]
        X = df.drop(columns=[target])
        task = infer_task(y)
        preprocessor, _, _ = build_preprocessor(X)

        if task == 'classification' and not pd.api.types.is_numeric_dtype(y):
            y = LabelEncoder().fit_transform(y.astype(str))

        if task == 'classification':
            model = RandomForestClassifier()
        else:
            model = RandomForestRegressor()

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipeline.fit(X, y)

        return {"status": "success", "task": task, "model": str(type(model))}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
import sklearn
from packaging import version

if version.parse(sklearn.__version__) >= version.parse("1.2"):
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)