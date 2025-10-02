📌 Project Overview: (Prototype).
My AutoML app provides a complete interactive pipeline where users can:
1️⃣ Upload & Validate Data – Upload CSVs, validate, and load into Pandas DataFrames.
2️⃣ Data Cleaning – Handle missing values, Removing Outliers.
3️⃣ Exploratory Data Analysis (EDA) – Generate summary stats & visualizations (histograms, heatmaps, distributions).
4️⃣ Data Preprocessing – Convert categorical features into numerical for training.
5️⃣ Model Training (AutoML):
 ▸Detects problem type (classification/regression).
 ▸Trains multiple candidate models.
 ▸Compares performance & selects the best model.
 ▸Runs hyper-parameter tuning on the best model.
 ▸Let the user download the tuned model (.pkl).
6️⃣ Advanced Visualization – Users can select columns + chart types (scatter, bar, heatmap, correlation) based on his/her requirement.
7️⃣ Report Generation – Export a PDF report with dataset summary, cleaning, EDA, feature engineering, model results, and visualizations.

⚙️ Tech Stack
Frontend: Streamlit
Backend: FastAPI (Python)
Python Lib: Numpy,Pandas, Scikit-learn, Matplotlib/Seaborn
Deployment: Streamlit


⚠️ Note: Since we’re using Streamlit(512MB RAM), the backend can take up some time to spin up when idle. Please be patient when testing. 😊

🔮 Future Scope
 🚀 Add Unsupervised Learning (K-Means, DBSCAN, PCA, etc.)
🚀 Will add better UI with Vite + React.
 🚀 Extend models with LightGBM, CatBoost, Neural Nets, Ensemble Stacking
 🚀 Implement regularization & cross-validation to avoid overfitting
 🚀 Enable database integration (Postgres/MySQL)
 🚀 Add user authentication & session management

 🌐 View Live Preview : https://lnkd.in/gJdmFTpp
 
This project was an amazing opportunity to combine our knowledge of Data Science..
