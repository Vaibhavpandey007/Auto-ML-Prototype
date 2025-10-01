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
from matplotlib.backends.backend_pdf import PdfPages

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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False



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
    """
    Displays feature importance in Streamlit.
    Tries SHAP first; if it fails (including additivity errors), falls back to permutation importance.
    """
    try:
        # Preprocess sample data
        X_processed = best_pipeline['preprocessor'].transform(X_sample)
        feature_names = best_pipeline['preprocessor'].get_feature_names_out()
        X_processed = pd.DataFrame(X_processed, columns=feature_names)

        st.subheader("Feature Importance (SHAP)")
        
        fig, ax = plt.subplots()

        # Use general SHAP Explainer (works for trees, linear, and others)
        explainer = shap.Explainer(best_pipeline['model'], X_processed)
        shap_values = explainer(X_processed)

        shap.summary_plot(shap_values, X_processed, show=False)
        # Render the active SHAP figure rather than the blank one we created
        st.pyplot(plt.gcf())

    except Exception:
        # Catch any SHAP errors silently and fallback
        st.info("SHAP explainability failed or produced warnings. Falling back to permutation importance...")
        if y_sample is not None:
            result = permutation_importance(best_pipeline['model'], X_processed, y_sample,
                                            n_repeats=5, random_state=42, n_jobs=-1)
            importances = pd.Series(result.importances_mean, index=feature_names)
            importances = importances.sort_values(ascending=False)

            fig, ax = plt.subplots()
            importances.plot(kind='bar', ax=ax)
            ax.set_title("Feature Importance (Permutation)")
            st.pyplot(fig)
            
def generate_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), ax=ax, cmap='coolwarm', annot=False)
    ax.set_title('Correlation Heatmap')
    return fig

def generate_target_distribution(y: pd.Series, task: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    if task == 'classification':
        pd.Series(y).value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Target Class Distribution')
    else:
        sns.histplot(y, bins=30, kde=True, ax=ax)
        ax.set_title('Target Distribution')
    return fig

def generate_feature_importance_permutation(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series):
    try:
        X_processed = pipeline['preprocessor'].transform(X)
        feature_names = pipeline['preprocessor'].get_feature_names_out()
        result = permutation_importance(pipeline['model'], X_processed, y,
                                        n_repeats=5, random_state=42, n_jobs=-1)
        importances = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)[:30]
        fig, ax = plt.subplots(figsize=(8, 6))
        importances.plot(kind='barh', ax=ax)
        ax.invert_yaxis()
        ax.set_title('Feature Importance (Permutation)')
        return fig
    except Exception:
        return None

def generate_pred_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.6)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, 'r--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Predicted vs Actual')
    return fig

def generate_residuals_plot(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(residuals, bins=30, kde=True, ax=ax)
    ax.set_title('Residuals Distribution')
    return fig

def generate_confusion_matrix_plot(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix')
    return fig

def generate_roc_curve_plot(model, X_processed_test, y_true):
    # Only for binary classification with predict_proba
    try:
        if len(np.unique(y_true)) != 2:
            return None
        proba = model.predict_proba(X_processed_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, proba)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, label='ROC curve')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve (Test)')
        ax.legend()
        return fig
    except Exception:
        return None
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
    # Train/Test split for evaluation and plots
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if task=='classification' else None)
    best_pipeline.fit(X_train, y_train)

    st.success("Best model fitted.")

    # Metrics
    y_pred = best_pipeline.predict(X_test)
    metrics_dict = {}
    if task == 'classification':
        metrics_dict['accuracy'] = accuracy_score(y_test, y_pred)
        metrics_dict['f1_macro'] = f1_score(y_test, y_pred, average='macro')
        # ROC AUC if binary and proba available
        roc_auc_val = None
        try:
            if len(np.unique(y_test)) == 2 and hasattr(best_pipeline['model'], 'predict_proba'):
                proba = best_pipeline['model'].predict_proba(best_pipeline['preprocessor'].transform(X_test))[:, 1]
                roc_auc_val = roc_auc_score(y_test, proba)
        except Exception:
            roc_auc_val = None
        if roc_auc_val is not None:
            metrics_dict['roc_auc'] = roc_auc_val
    else:
        metrics_dict['r2'] = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        metrics_dict['rmse'] = rmse

    metrics_df = pd.DataFrame([metrics_dict])
    st.subheader("Evaluation Metrics (Test)")
    st.dataframe(metrics_df)

    # Chart selection UI
    st.subheader("Visualizations")
    chart_options = [
        'Correlation Heatmap',
        'Target Distribution',
        'Feature Importance (Permutation)',
        'Predicted vs Actual' if task=='regression' else 'Confusion Matrix',
        'Residuals' if task=='regression' else 'ROC Curve',
    ]
    selected_charts = st.multiselect('Select charts to display and include in report', chart_options, default=chart_options[:2])

    figures = []
    # Generate and show figures as per selection
    if 'Correlation Heatmap' in selected_charts:
        fig = generate_correlation_heatmap(df)
        if fig is not None:
            st.pyplot(fig)
            figures.append(fig)
        else:
            st.info('Correlation Heatmap not available (no numeric columns).')

    if 'Target Distribution' in selected_charts:
        fig = generate_target_distribution(y, task)
        st.pyplot(fig)
        figures.append(fig)

    if 'Feature Importance (Permutation)' in selected_charts:
        fig = generate_feature_importance_permutation(best_pipeline, X_test, y_test)
        if fig is not None:
            st.pyplot(fig)
            figures.append(fig)
        else:
            st.info('Feature importance could not be computed.')

    if task == 'regression':
        if 'Predicted vs Actual' in selected_charts:
            fig = generate_pred_vs_actual(y_test, y_pred)
            st.pyplot(fig)
            figures.append(fig)
        if 'Residuals' in selected_charts:
            fig = generate_residuals_plot(y_test, y_pred)
            st.pyplot(fig)
            figures.append(fig)
    else:
        if 'Confusion Matrix' in selected_charts:
            fig = generate_confusion_matrix_plot(y_test, y_pred)
            st.pyplot(fig)
            figures.append(fig)
        if 'ROC Curve' in selected_charts:
            X_test_processed = best_pipeline['preprocessor'].transform(X_test)
            fig = generate_roc_curve_plot(best_pipeline['model'], X_test_processed, y_test)
            if fig is not None:
                st.pyplot(fig)
                figures.append(fig)
            else:
                st.info('ROC curve not available (needs binary classification and predict_proba).')

    # Optional: SHAP on a small sample
    sample_size = min(100, len(X_train))
    X_sample = X_train.sample(sample_size, random_state=42)
    y_sample = pd.Series(y_train).sample(sample_size, random_state=42) if hasattr(y_train, '__len__') else None
    explain_model(best_pipeline, X_sample, y_sample)

    # Downloads: metrics CSV and figures PDF
    st.subheader("Download Reports")
    csv_bytes = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Download Metrics (CSV)',
        data=csv_bytes,
        file_name='metrics_report.csv',
        mime='text/csv'
    )

    if len(figures) > 0:
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            for fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
        pdf_buffer.seek(0)
        st.download_button(
            label='Download Selected Charts (PDF)',
            data=pdf_buffer,
            file_name='visualizations_report.pdf',
            mime='application/pdf'
        )
    else:
        st.info('Select at least one chart to enable PDF download.')

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