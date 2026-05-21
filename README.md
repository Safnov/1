Stroke-Associated Pneumonia (SAP) Prediction in Older Patients

A machine learning pipeline to develop, compare, and interpret eight classifiers for predicting stroke‑associated pneumonia in acute ischemic stroke patients aged ≥65 years. The project includes robust pre‑processing (LASSO feature selection, SMOTE, standardization), hyperparameter tuning with 5‑fold cross‑validation, comprehensive model evaluation (AUC, Brier score, calibration, decision curve analysis), and SHAP‑based interpretability. A ready‑to‑use best_model.pkl is exported for clinical deployment.
Key Features

    Data Pre‑processing – Assumes a cleaned, complete dataset (no imputation).

    Feature Selection – LASSO with 10‑fold CV selects the 10–12 most predictive features.

    Class Imbalance Handling – SMOTE integrated inside the cross‑validation loop to prevent data leakage.

    Model Training – Eight classifiers (Logistic Regression, SVM, MLP, LightGBM, XGBoost, Random Forest, Gradient Boosting, CatBoost) optimized via 5‑fold stratified grid search.

    Comprehensive Metrics – AUC (Bootstrap 95% CI), Brier score (Bootstrap 95% CI), sensitivity, specificity, precision, F1, optimal threshold, Hosmer‑Lemeshow test, calibration slope/intercept.

    Model Comparison – Bootstrap DeLong test for pairwise AUC differences.

    Visualizations – ROC, Precision‑Recall, calibration, and decision curves; SHAP summary and importance plots.

    Reproducibility – Fixed random seed throughout; all outputs saved to a dedicated directory.

Repository Structure
text

├── main_analysis.py               # Main V15 script
├── datafinalversion.csv           # Input data (1011 patients × 33 features)
├── medical_analysis_output/
│   └── optimized_v15_fixed/
│       ├── figures/               # All generated plots (PNG)
│       ├── performance_matrix_*.csv
│       ├── SHAP_values_*.csv
│       ├── SHAP_Importance_*.csv
│       ├── best_model.pkl         # Final model, features, threshold
│       └── analysis.log           # Full run log
└── README.md

Installation & Dependencies
1. Clone the repository
bash

git clone https://github.com/yourusername/SAP_Prediction.git
cd SAP_Prediction

2. Create a virtual environment (recommended)
bash

conda create -n sap python=3.10
conda activate sap

3. Install required packages

All dependencies can be installed via pip:
bash

pip install -r requirements.txt

Core dependencies (also listed in requirements.txt):

    Python ≥3.9

    pandas, numpy, scipy

    scikit-learn ≥1.2

    imbalanced-learn

    xgboost, lightgbm, catboost

    shap

    matplotlib, seaborn

    joblib, psutil

How to Run

    Prepare the data
    Ensure the cleaned dataset datafinalversion.csv is placed at the root directory or update DATA_PATH in the script to point to your file.

    Run the analysis
    bash

    python main_analysis.py

    Outputs
    All results will be saved under medical_analysis_output/optimized_v15_fixed/:

        Figures – ROC, PR, calibration, DCA, SHAP plots.

        Performance matrices – CSV files with AUC, Brier, F1, etc.

        Log file – analysis.log contains all metrics, best parameters, DeLong comparisons, and Hosmer‑Lemeshow test results.

        Trained model – best_model.pkl can be directly loaded for new predictions.

    Example to load and use the saved model:
    python

    import joblib
    model_pack = joblib.load('best_model.pkl')
    pipe = model_pack['pipe']       # Full pipeline (scaler + SMOTE + classifier)
    features = model_pack['features']
    threshold = model_pack['threshold']

    X_new = ...  # new patient data with the same features
    proba = pipe.predict_proba(X_new)[:, 1]
    pred = (proba >= threshold).astype(int)

Data

The model was developed on a single‑centre retrospective cohort of 1,011 acute ischemic stroke patients aged ≥65 years. The dataset includes demographic, clinical, and laboratory variables; the outcome is stroke‑associated pneumonia (18.8% incidence).
Note: The data file is not included in this repository due to institutional privacy restrictions. A synthetic sample or description of the feature schema can be provided upon request.
Citation

If you use this code or results in your research, please cite our manuscript:

    [Manuscript reference – to be added]

License

This project is licensed under the MIT License – see the LICENSE file for details.
