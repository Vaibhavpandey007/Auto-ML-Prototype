🧠 AutoML Interactive Pipeline (Prototype)

My AutoML app provides a complete interactive pipeline for end-to-end machine learning, allowing users to process data, train models, and visualize results—all in one place.

📌 Features
1️⃣ Upload & Validate Data

Upload CSV files.

Validate and load datasets into Pandas DataFrames.

2️⃣ Data Cleaning

Handle missing values automatically.

Detect and remove outliers.

3️⃣ Exploratory Data Analysis (EDA)

Generate summary statistics.

Visualize distributions, histograms, and correlation heatmaps.

4️⃣ Data Preprocessing

Convert categorical features to numerical values.

Prepare data for model training.

5️⃣ Model Training (AutoML)

Detects the problem type: classification or regression.

Trains multiple candidate models.

Compares performance metrics and selects the best model.

Performs hyperparameter tuning on the best model.

Allows users to download the tuned model (.pkl) for future use.

6️⃣ Advanced Visualization

Users can select columns and chart types (scatter, bar, heatmap, correlation).

Generate customized visual insights.

7️⃣ Report Generation

Export a comprehensive PDF report including:

Dataset summary

Data cleaning steps

EDA visualizations

Feature engineering

Model results

⚙️ Tech Stack

Frontend: Streamlit

Backend: FastAPI (Python)

Python Libraries: Numpy, Pandas, Scikit-learn, Matplotlib, Seaborn

Deployment: Streamlit

⚠️ Notes

Streamlit provides 512MB RAM; backend may take a few seconds to spin up if idle.

Please be patient when testing. 😊

🔮 Future Scope

Add Unsupervised Learning: K-Means, DBSCAN, PCA.

Improve UI with Vite + React.

Extend models with LightGBM, CatBoost, Neural Networks, Ensemble Stacking.

Implement regularization & cross-validation to reduce overfitting.

Enable database integration (Postgres/MySQL).

Add user authentication & session management.

🌐 Live Preview :  https://lnkd.in/gJdmFTpp

