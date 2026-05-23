# -*- coding: utf-8 -*-
"""
完整可运行版本（V15）—— 基于 V14，移除缺失值填补功能
- 假设输入数据已经过严格筛选，无缺失值
- 保留 LASSO 特征选择、SMOTE 内嵌交叉验证、多模型比较
- 保留 Bootstrap DeLong 检验和 Hosmer‑Lemeshow 校准检验
- 保留 SHAP 分析及数据表导出
- 保留所有原始绘图功能
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV, Lasso
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
                             precision_score, f1_score, roc_curve,
                             precision_recall_curve, confusion_matrix,
                             brier_score_loss, auc)
from sklearn.calibration import calibration_curve
from sklearn.base import clone
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings
import logging
import joblib
import psutil

# ========================== 全局配置 ==========================
DATA_PATH = r"D:\desktop\datafinalversion.csv"
OUTPUT_DIR = r"D:\desktop\代码合集250524\可运行\medical_analysis_output\optimized_v15_fixed"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

os.environ['LOKY_MAX_CPU_COUNT'] = '4'
os.environ['JOBLIB_START_METHOD'] = 'forkserver'

log_file = os.path.join(OUTPUT_DIR, 'analysis.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
logger = logging.getLogger()
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
np.random.seed(42)

MODEL_COLORS = {
    'XGBoost': '#1f77b4', 'SVM': '#ff7f0e', 'LightGBM': '#2ca02c',
    'LR': '#d62728', 'RF': '#9467bd', 'MLP': '#8c564b',
    'GBDT': '#e377c2', 'CatBoost': '#7f7f7f'
}

FEATURE_NAME_MAP = {
    'mRS': 'Modified Rankin Scale', 'VTE_score': 'VTE Risk Score',
    'Urea': 'Blood Urea Nitrogen', 'NIHSS': 'NIH Stroke Scale',
    'Total Cholesterol': 'Total Cholesterol', 'Albumin': 'Serum Albumin',
    'LDL': 'LDL Cholesterol', 'NLR': 'Neutrophil-Lymphocyte Ratio',
    'CRP': 'C-Reactive Protein', 'RDW': 'Red Cell Distribution Width',
    'age': 'Age', 'kwst': 'Wada drinking test',
    'ppi': 'Proton Pump Inhibitor',
    'COPD': 'Chronic Obstructive Pulmonary Disease',
    'HeartFailure': 'Heart Failure',
    'IntSS': 'Internal Carotid Artery Stenosis Score',
    'pqi': 'Patient Quality Index', 'Atrial Fibrillation': 'Atrial Fibrillation',
    'DM': 'Diabetes Mellitus', 'Smoking': 'Smoking Status',
    'Hyperlipidemia': 'Hyperlipidemia'
}

# ========================== 工具函数 ==========================
def load_and_preprocess_data(file_path):
    """加载数据并处理基本数据类型"""
    logger.info(f"Loading data: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Original dataset shape: {df.shape}")
    if 'Outcome' not in df.columns:
        raise ValueError("Dataset is missing 'Outcome' column")
    df['Outcome'] = df['Outcome'].astype(int)
    for col in df.columns:
        if col == 'Outcome':
            continue
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                df[col] = pd.Categorical(df[col]).codes
    # 移除可能的无穷值
    df = df.replace([np.inf, -np.inf], np.nan)
    # 假设数据已清理，仅移除任何残留的 NaN 行（安全处理）
    if df.isnull().any().any():
        logger.warning("Data contains NaN values. Dropping rows with NaN (assuming minimal).")
        df = df.dropna()
    return df

def feature_selection_with_lasso(X_train, y_train, min_features=10, max_features=12):
    logger.info("Performing LASSO feature selection (with internal scaling)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    alphas = np.logspace(-6, 6, 100)
    lasso_cv = LassoCV(alphas=alphas, cv=10, max_iter=10000, random_state=42, n_jobs=1)
    lasso_cv.fit(X_scaled, y_train)
    coefs = np.abs(lasso_cv.coef_)
    sorted_idx = np.argsort(coefs)[::-1]
    n_features = max(min_features, min(max_features, np.sum(coefs > 1e-5)))
    selected_features = X_train.columns[sorted_idx[:n_features]].tolist()
    logger.info(f"Selected {len(selected_features)} features: {selected_features}")
    lasso_cv.X_train_ = X_scaled
    lasso_cv.y_train_ = y_train.values
    return selected_features, lasso_cv

def calculate_brier_ci(y_true, y_proba, brier_score, n_bootstraps=2000):
    rng = np.random.default_rng(42)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        scores.append(brier_score_loss(
            y_true.iloc[idx] if hasattr(y_true, 'iloc') else y_true[idx],
            y_proba[idx]))
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

def auc_ci_bootstrap(y_true, y_proba, n_bootstraps=2000):
    rng = np.random.default_rng(42)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        scores.append(roc_auc_score(
            y_true.iloc[idx] if hasattr(y_true, 'iloc') else y_true[idx],
            y_proba[idx]))
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

def delong_roc_test(y_true, proba1, proba2, n_bootstraps=2000):
    rng = np.random.default_rng(42)
    n = len(y_true)
    diff_obs = roc_auc_score(y_true, proba1) - roc_auc_score(y_true, proba2)
    boot_diffs = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        auc1 = roc_auc_score(y_true.iloc[idx] if hasattr(y_true, 'iloc') else y_true[idx], proba1[idx])
        auc2 = roc_auc_score(y_true.iloc[idx] if hasattr(y_true, 'iloc') else y_true[idx], proba2[idx])
        boot_diffs.append(auc1 - auc2)
    boot_diffs = np.array(boot_diffs)
    p_value = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))
    return diff_obs, p_value

def hosmer_lemeshow_test(y_true, y_proba, n_bins=10):
    """
    Hosmer-Lemeshow 校准检验 (10 分箱)
    返回: chi2_statistic, p_value
    """
    bins_idx = np.percentile(y_proba, np.linspace(0, 100, n_bins + 1))
    bins_idx[-1] += 1e-8
    bins = np.digitize(y_proba, bins_idx) - 1
    bins = np.clip(bins, 0, n_bins - 1)

    observed_pos = np.array([y_true[bins == i].sum() for i in range(n_bins)])
    expected_pos = np.array([y_proba[bins == i].sum() for i in range(n_bins)])
    bin_counts = np.array([(bins == i).sum() for i in range(n_bins)])

    mask = bin_counts > 0
    observed_pos = observed_pos[mask]
    expected_pos = expected_pos[mask]
    bin_counts = bin_counts[mask]

    if len(bin_counts) < 2:
        return np.nan, np.nan

    hl_stat = np.sum((observed_pos - expected_pos) ** 2 /
                     (expected_pos * (1 - expected_pos / bin_counts)))
    df = len(bin_counts) - 2
    if df < 1:
        return hl_stat, np.nan
    p_value = chi2.sf(hl_stat, df)
    return hl_stat, p_value

# ========================== Lasso 可视化 (保留原功能) ==========================
def plot_lasso_cv_results(lasso_cv, X_train, y_train, output_dir, dataset_name):
    logger.info(f"Generating LassoCV visualizations for {dataset_name}...")
    try:
        plt.figure(figsize=(12, 6), dpi=600)
        MSEs = lasso_cv.mse_path_
        MSEs_mean = np.mean(MSEs, axis=1)
        MSEs_std = np.std(MSEs, axis=1)
        plt.errorbar(lasso_cv.alphas_, MSEs_mean, yerr=MSEs_std, fmt="o", ms=3,
                     mfc="r", mec="r", ecolor="lightblue", elinewidth=2, capsize=4,
                     label='Cross-Validation MSE', capthick=1)
        plt.semilogx()
        plt.axvline(lasso_cv.alpha_, color="black", ls="--", label=f'Best alpha {round(lasso_cv.alpha_, 3)}')
        plt.xlabel("Alpha (log scale)"); plt.ylabel("Mean Squared Error")
        plt.title("LassoCV: MSE vs Alpha (10-fold CV)"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        save_path1 = os.path.join(output_dir, f'lasso_cv_mse_{dataset_name}.jpg')
        plt.savefig(save_path1, dpi=600, bbox_inches='tight'); plt.close()
        logger.info(f"Saved LassoCV MSE plot: {save_path1}")

        plt.figure(figsize=(12, 6), dpi=600)
        alphas = lasso_cv.alphas_
        coefs = []
        for alpha in alphas:
            model = Lasso(alpha=alpha, max_iter=10000)
            model.fit(X_train, y_train)
            coefs.append(model.coef_)
        coefs = np.array(coefs).T
        for i in range(coefs.shape[0]):
            plt.semilogx(alphas, coefs[i, :], "-", linewidth=1.5, alpha=0.7)
        plt.axvline(lasso_cv.alpha_, color="black", ls="--", label=f'Best alpha {round(lasso_cv.alpha_, 3)}')
        plt.xlabel("Alpha (log scale)"); plt.ylabel("Coefficients")
        plt.title("LassoCV: Coefficient Paths"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        save_path2 = os.path.join(output_dir, f'lasso_cv_coefs_{dataset_name}.jpg')
        plt.savefig(save_path2, dpi=600, bbox_inches='tight'); plt.close()
        logger.info(f"Saved LassoCV coefficient paths plot: {save_path2}")

        return {'mse_plot': save_path1, 'coef_plot': save_path2,
                'best_alpha': lasso_cv.alpha_, 'selected_features': np.sum(lasso_cv.coef_ != 0)}
    except Exception as e:
        logger.error(f"Failed to generate LassoCV visualizations: {str(e)}")
        return None

def plot_lasso_path(lasso_cv, output_dir, dataset_name, feature_names):
    logger.info("Generating LASSO path plot...")
    try:
        alphas = lasso_cv.alphas_
        coefs = []
        for alpha in alphas:
            model = Lasso(alpha=alpha, max_iter=10000)
            model.fit(lasso_cv.X_train_, lasso_cv.y_train_)
            coefs.append(model.coef_)
        coefs = np.array(coefs).T
        log_alphas = np.log10(alphas)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=600)
        fig.suptitle("LASSO Regression Analysis", fontsize=16, fontweight='bold', y=0.98)
        for i in range(coefs.shape[0]):
            ax1.plot(log_alphas, coefs[i, :], linewidth=1.5, alpha=0.7)
        ax1.axvline(np.log10(lasso_cv.alpha_), color='r', linestyle='--', linewidth=2, label=f'Optimal Lambda ({lasso_cv.alpha_:.4f})')
        ax1.set_xlabel('Log(Lambda)'); ax1.set_ylabel('Coefficients'); ax1.set_title('(A) LASSO Coefficient Paths'); ax1.grid(True, alpha=0.5); ax1.legend()
        mean_abs = np.mean(np.abs(coefs), axis=1)
        top_idx = np.argsort(mean_abs)[::-1][:10]
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_idx)))
        for idx, color in zip(top_idx, colors):
            fname = feature_names[idx] if idx < len(feature_names) else f"Feature {idx+1}"
            ax2.plot(log_alphas, coefs[idx, :], color=color, linewidth=2.5, label=FEATURE_NAME_MAP.get(fname, fname))
        ax2.axvline(np.log10(lasso_cv.alpha_), color='r', linestyle='--', linewidth=2, label='Optimal Lambda')
        ax2.set_xlabel('Log(Lambda)'); ax2.set_ylabel('Coefficient Value'); ax2.set_title('(B) Top Feature Trajectories')
        ax2.grid(True, alpha=0.5); ax2.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.35,1))
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'LASSO_Path_{dataset_name}.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight'); plt.close()
        logger.info(f"Saved LASSO path plot: {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate LASSO path plot: {str(e)}")

def detect_multicollinearity(X, output_dir, dataset_name, prefix="", threshold=0.8):
    logger.info("Detecting multicollinearity...")
    try:
        corr_matrix = X.corr()
        plt.figure(figsize=(14, 12), dpi=600)
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5, annot_kws={"size":8})
        plt.title(f'Correlation Matrix ({prefix}{dataset_name})', fontsize=16)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'Correlation_Matrix_{prefix}{dataset_name}.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight'); plt.close()
        logger.info(f"Saved correlation matrix: {save_path}")
        high_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        if high_pairs:
            logger.warning(f"Found {len(high_pairs)} highly correlated feature pairs (|r| > {threshold})")
        return corr_matrix, high_pairs
    except Exception as e:
        logger.error(f"Multicollinearity detection failed: {str(e)}")
        return None, []

# ========================== 模型训练（内嵌SMOTE） ==========================
def train_models_with_cv(X_train_original, y_train_original, X_test, y_test, n_folds=5):
    models = {
        'LR': LogisticRegression(max_iter=1000, class_weight='balanced', penalty='l2', random_state=42),
        'SVM': SVC(probability=True, kernel='linear', class_weight='balanced', random_state=42),
        'MLP': MLPClassifier(max_iter=1000, early_stopping=True, hidden_layer_sizes=(30, 15),
                             alpha=0.5, learning_rate_init=0.01, random_state=42),
        'LightGBM': LGBMClassifier(verbose=-1, reg_alpha=3.0, reg_lambda=3.0, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', reg_alpha=3.0, reg_lambda=3.0,
                                 use_label_encoder=False, random_state=42),
        'RF': RandomForestClassifier(class_weight='balanced', max_depth=5,
                                     min_samples_leaf=20, random_state=42),
        'GBDT': GradientBoostingClassifier(max_depth=3, min_samples_leaf=20, random_state=42),
        'CatBoost': CatBoostClassifier(verbose=False, depth=4, l2_leaf_reg=3.0, random_seed=42)
    }
    param_grids = {
        'LR': {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear']
        },
        'SVM': {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10],
            'classifier__gamma': ['scale', 'auto'],
            'classifier__kernel': ['linear']
        },
        'MLP': {
            'classifier__hidden_layer_sizes': [(30,), (50,), (30,15), (50,25)],
            'classifier__alpha': [0.001, 0.01, 0.1],
            'classifier__learning_rate_init': [0.001, 0.01],
            'classifier__early_stopping': [True]
        },
        'LightGBM': {
            'classifier__learning_rate': [0.01, 0.05],
            'classifier__n_estimators': [200, 300],
            'classifier__num_leaves': [7, 15, 31],
            'classifier__max_depth': [3, 5, 7],
            'classifier__min_child_samples': [50, 100],
            'classifier__reg_alpha': [1.0, 3.0, 5.0],
            'classifier__reg_lambda': [1.0, 3.0, 5.0],
            'classifier__subsample': [0.6, 0.8]
        },
        'XGBoost': {
            'classifier__learning_rate': [0.01, 0.05],
            'classifier__max_depth': [3, 5, 7],
            'classifier__n_estimators': [100, 200],
            'classifier__subsample': [0.6, 0.8],
            'classifier__colsample_bytree': [0.6, 0.8],
            'classifier__reg_alpha': [1.0, 3.0],
            'classifier__reg_lambda': [1.0, 3.0],
            'classifier__gamma': [0.1, 0.5]
        },
        'RF': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [5, 10, 15],
            'classifier__min_samples_split': [10, 20],
            'classifier__min_samples_leaf': [4, 8],
            'classifier__max_features': ['sqrt', 'log2']
        },
        'GBDT': {
            'classifier__learning_rate': [0.01, 0.05],
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5],
            'classifier__min_samples_leaf': [20, 50],
            'classifier__subsample': [0.6, 0.8]
        },
        'CatBoost': {
            'classifier__iterations': [200, 500],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__depth': [4, 6, 8],
            'classifier__l2_leaf_reg': [3.0, 5.0, 7.0],
            'classifier__border_count': [64],
            'classifier__subsample': [0.6, 0.8]
        }
    }
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {}

    for name, base_model in models.items():
        logger.info(f"\n=== Training {name} ===")
        pipe = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42, k_neighbors=min(5, len(y_train_original)-2))),
            ('classifier', base_model)
        ])
        grid = GridSearchCV(pipe, param_grids[name], cv=cv, scoring='roc_auc', n_jobs=1, verbose=1)
        grid.fit(X_train_original, y_train_original)
        best_pipe = grid.best_estimator_
        logger.info(f"Best parameters: {grid.best_params_}")

        y_train_proba = best_pipe.predict_proba(X_train_original)[:, 1]
        y_train_pred = best_pipe.predict(X_train_original)
        y_test_proba = best_pipe.predict_proba(X_test)[:, 1]
        y_test_pred = best_pipe.predict(X_test)

        def compute_metrics(y_true, y_proba, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            spec = tn / (tn+fp) if (tn+fp) > 0 else 0
            auc_val = roc_auc_score(y_true, y_proba)
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
            auprc = auc(recall_curve, precision_curve) if len(precision_curve) > 1 else 0
            brier = brier_score_loss(y_true, y_proba)
            brier_ci = calculate_brier_ci(y_true, y_proba, brier)
            auc_low, auc_high = auc_ci_bootstrap(y_true, y_proba)
            return {
                'AUC': auc_val, 'AUC_CI': (auc_low, auc_high),
                'Accuracy': accuracy_score(y_true, y_pred),
                'Sensitivity': recall_score(y_true, y_pred),
                'Specificity': spec,
                'Precision': precision_score(y_true, y_pred),
                'F1': f1_score(y_true, y_pred),
                'AUPRC': auprc,
                'Brier': brier, 'Brier_CI': brier_ci
            }

        train_metrics = compute_metrics(y_train_original, y_train_proba, y_train_pred)
        test_metrics = compute_metrics(y_test, y_test_proba, y_test_pred)

        train_hl_stat, train_hl_p = hosmer_lemeshow_test(y_train_original, y_train_proba)
        test_hl_stat, test_hl_p = hosmer_lemeshow_test(y_test, y_test_proba)

        prec, rec, thresh = precision_recall_curve(y_train_original, y_train_proba)
        f1_vals = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
        best_idx = np.argmax(f1_vals)
        best_th = thresh[best_idx] if len(thresh) > 0 else 0.5
        train_opt_f1 = f1_score(y_train_original, (y_train_proba >= best_th).astype(int))
        test_opt_f1 = f1_score(y_test, (y_test_proba >= best_th).astype(int))

        results[name] = {
            'Model': best_pipe,
            'Train_AUC': train_metrics['AUC'],
            'Train_AUC_CI': train_metrics['AUC_CI'],
            'Train_Accuracy': train_metrics['Accuracy'],
            'Train_Sensitivity': train_metrics['Sensitivity'],
            'Train_Specificity': train_metrics['Specificity'],
            'Train_Precision': train_metrics['Precision'],
            'Train_F1': train_metrics['F1'],
            'Train_Optimized_F1': train_opt_f1,
            'Train_Best_Threshold': best_th,
            'Train_AUPRC': train_metrics['AUPRC'],
            'Train_Brier': train_metrics['Brier'],
            'Train_Brier_CI': train_metrics['Brier_CI'],
            'Train_HL_statistic': train_hl_stat,
            'Train_HL_pvalue': train_hl_p,
            'Test_AUC': test_metrics['AUC'],
            'Test_AUC_CI': test_metrics['AUC_CI'],
            'Test_Accuracy': test_metrics['Accuracy'],
            'Test_Sensitivity': test_metrics['Sensitivity'],
            'Test_Specificity': test_metrics['Specificity'],
            'Test_Precision': test_metrics['Precision'],
            'Test_F1': test_metrics['F1'],
            'Test_Optimized_F1': test_opt_f1,
            'Test_Best_Threshold': best_th,
            'Test_AUPRC': test_metrics['AUPRC'],
            'Test_Brier': test_metrics['Brier'],
            'Test_Brier_CI': test_metrics['Brier_CI'],
            'Test_HL_statistic': test_hl_stat,
            'Test_HL_pvalue': test_hl_p,
            'F1_Difference': train_metrics['F1'] - test_metrics['F1'],
            'Best_Params': grid.best_params_
        }
        logger.info(f"{name} Train AUC {train_metrics['AUC']:.3f} [{train_metrics['AUC_CI'][0]:.3f}-{train_metrics['AUC_CI'][1]:.3f}], "
                    f"Test AUC {test_metrics['AUC']:.3f} [{test_metrics['AUC_CI'][0]:.3f}-{test_metrics['AUC_CI'][1]:.3f}]")
    return results

# ========================== 绘图函数 (保留全部) ==========================
def plot_roc_curves(results, X, y, dataset_name, set_type):
    plt.figure(figsize=(10, 8))
    auc_key = 'Train_AUC' if set_type == "Training" else 'Test_AUC'
    ci_key  = 'Train_AUC_CI' if set_type == "Training" else 'Test_AUC_CI'
    for name, res in sorted(results.items(), key=lambda x: x[1][auc_key], reverse=True):
        proba = res['Model'].predict_proba(X)[:, 1]
        auc_val = roc_auc_score(y, proba)
        auc_low, auc_high = res[ci_key]
        color = MODEL_COLORS.get(name, '#%02x%02x%02x' % tuple(np.random.randint(0,255,3)))
        fpr, tpr, _ = roc_curve(y, proba)
        plt.plot(fpr, tpr, color=color, linewidth=2,
                 label=f'{name} (AUC={auc_val:.3f} [{auc_low:.3f}-{auc_high:.3f}])')
    plt.plot([0,1],[0,1],'k--', linewidth=1.5)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'{set_type} Set - ROC Curves'); plt.legend(loc='lower right'); plt.grid(True, alpha=0.7); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'ROC_Curves_{dataset_name}_{set_type}.png'), dpi=600, bbox_inches='tight')
    plt.close()

def plot_pr_curves(results, X, y, dataset_name, set_type):
    plt.figure(figsize=(10,8))
    baseline = y.mean()
    for name, res in results.items():
        proba = res['Model'].predict_proba(X)[:, 1]
        prec, rec, _ = precision_recall_curve(y, proba)
        auprc = auc(rec, prec)
        color = MODEL_COLORS.get(name, '#%02x%02x%02x' % tuple(np.random.randint(0,255,3)))
        plt.plot(rec, prec, color=color, linewidth=2, label=f'{name} (AUPRC={auprc:.3f})')
    plt.axhline(baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.2f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'{set_type} Set - Precision-Recall Curves')
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.7); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'PR_Curves_{dataset_name}_{set_type}.png'), dpi=600, bbox_inches='tight')
    plt.close()

def plot_calibration_curves(results, X, y, dataset_name, set_type):
    plt.figure(figsize=(10,8))
    plt.plot([0,1],[0,1],'k:', label='Perfectly calibrated')
    for name, res in results.items():
        proba = res['Model'].predict_proba(X)[:, 1]
        frac, mean_pred = calibration_curve(y, proba, n_bins=10, strategy='quantile')
        color = MODEL_COLORS.get(name, '#%02x%02x%02x' % tuple(np.random.randint(0,255,3)))
        plt.plot(mean_pred, frac, 's-', color=color, markersize=8, label=name)
    plt.xlabel('Mean predicted probability'); plt.ylabel('Fraction of positives')
    plt.ylim([-0.05, 1.05]); plt.legend(loc='upper left'); plt.grid(True, alpha=0.7); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'Calibration_Curves_{dataset_name}_{set_type}.png'), dpi=600, bbox_inches='tight')
    plt.close()

def plot_dca_curve(results, X, y, dataset_name, set_type):
    y = np.array(y)
    n = len(y)
    thresholds = np.linspace(0.01, 0.99, 50)
    plt.figure(figsize=(10,8))
    net_all = [np.mean(y) - (1-np.mean(y))*(pt/(1-pt)) for pt in thresholds]
    plt.plot(thresholds, net_all, 'k-', label='Treat All')
    plt.plot(thresholds, [0]*len(thresholds), 'k--', label='Treat None')
    for name, res in results.items():
        proba = res['Model'].predict_proba(X)[:, 1]
        nb = [np.sum((proba>=pt)&(y==1))/n - np.sum((proba>=pt)&(y==0))/n*(pt/(1-pt)) for pt in thresholds]
        color = MODEL_COLORS.get(name, '#%02x%02x%02x' % tuple(np.random.randint(0,255,3)))
        plt.plot(thresholds, nb, color=color, label=name)
    plt.xlim([0,1]); plt.ylim([-0.05, max(net_all)*1.1])
    plt.xlabel('Threshold Probability'); plt.ylabel('Net Benefit');
    plt.title(f'{set_type} Set - Decision Curve Analysis'); plt.legend(); plt.grid(True, alpha=0.7); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'DCA_{dataset_name}_{set_type}.png'), dpi=600, bbox_inches='tight')
    plt.close()

def plot_model_comparison(results, dataset_name, set_type):
    prefix = 'Train' if set_type == "Training" else 'Test'
    metrics = [f'{prefix}_AUC', f'{prefix}_Accuracy', f'{prefix}_Sensitivity',
               f'{prefix}_Specificity', f'{prefix}_F1', f'{prefix}_Optimized_F1', f'{prefix}_Brier']
    data = []
    for model_name, res in results.items():
        if f'{prefix}_AUC_CI' in res:
            auc_ci = f"{res[f'{prefix}_AUC']:.3f} [{res[f'{prefix}_AUC_CI'][0]:.3f}-{res[f'{prefix}_AUC_CI'][1]:.3f}]"
            data.append({'Model': model_name, 'Metric': f'{prefix}_AUC', 'Value': res[f'{prefix}_AUC'], 'Display': auc_ci})
        else:
            data.append({'Model': model_name, 'Metric': f'{prefix}_AUC', 'Value': res[f'{prefix}_AUC'], 'Display': f"{res[f'{prefix}_AUC']:.3f}"})
        for metric in metrics:
            if metric != f'{prefix}_AUC' and metric in res:
                data.append({'Model': model_name, 'Metric': metric, 'Value': res[metric], 'Display': f"{res[metric]:.3f}"})
    df = pd.DataFrame(data)
    plt.figure(figsize=(16,8))
    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df, palette=MODEL_COLORS)
    plt.title(f'{set_type} Set - Model Performance Comparison')
    plt.axhline(0.5, color='gray', linestyle='--'); plt.axhline(0.25, color='r', linestyle='--')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4); plt.grid(axis='y', alpha=0.7); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'Model_Performance_Comparison_{dataset_name}_{set_type}.png'), dpi=600, bbox_inches='tight')
    plt.close()

def plot_f1_comparison(results, dataset_name, set_type):
    prefix = 'Train' if set_type == "Training" else 'Test'
    f1_key = f'{prefix}_F1'; opt_key = f'{prefix}_Optimized_F1'
    base_f1 = [res[f1_key] for res in results.values()]
    opt_f1 = [res[opt_key] for res in results.values()]
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.35
    plt.figure(figsize=(12,8))
    plt.bar(x - width/2, base_f1, width, label='Default Threshold (0.5)', color='lightblue')
    plt.bar(x + width/2, opt_f1, width, label='Optimized Threshold', color='orange', alpha=0.7)
    for i, (b, o) in enumerate(zip(base_f1, opt_f1)):
        plt.text(i-0.2, b+0.01, f'{b:.3f}', ha='center', fontsize=9)
        plt.text(i+0.2, o+0.01, f'{o:.3f}', ha='center', fontsize=9)
    plt.axhline(0.5, color='r', linestyle='--', label='Clinical Minimum')
    plt.xticks(x, models, rotation=45, ha='right'); plt.ylabel('F1 Score')
    plt.title(f'{set_type} Set - F1 Score Comparison'); plt.legend(); plt.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'F1_Comparison_{dataset_name}_{set_type}.png'), dpi=600, bbox_inches='tight')
    plt.close()

def plot_performance_table(results, dataset_name, set_type):
    prefix = 'Train' if set_type == "Training" else 'Test'
    table_data = []
    for name, res in results.items():
        auc_ci = res.get(f'{prefix}_AUC_CI', (0,0))
        auc_disp = f"{res[f'{prefix}_AUC']:.3f} [{auc_ci[0]:.3f}-{auc_ci[1]:.3f}]"
        brier_ci = res.get(f'{prefix}_Brier_CI', (0,0))
        brier_disp = f"{res[f'{prefix}_Brier']:.4f} [{brier_ci[0]:.4f}-{brier_ci[1]:.4f}]"
        hl_stat = res.get(f'{prefix}_HL_statistic', np.nan)
        hl_p = res.get(f'{prefix}_HL_pvalue', np.nan)
        hl_disp = f"{hl_stat:.2f} (p={hl_p:.4f})" if not np.isnan(hl_stat) else "N/A"
        table_data.append([
            name, auc_disp, f"{res[f'{prefix}_Accuracy']:.3f}", f"{res[f'{prefix}_Sensitivity']:.3f}",
            f"{res[f'{prefix}_Specificity']:.3f}", f"{res[f'{prefix}_Precision']:.3f}",
            f"{res[f'{prefix}_F1']:.3f}", f"{res[f'{prefix}_Optimized_F1']:.3f}",
            f"{res[f'{prefix}_Best_Threshold']:.3f}", brier_disp, hl_disp
        ])
    plt.figure(figsize=(18,6))
    columns = ['Model', 'AUC (95% CI)', 'Accuracy', 'Sensitivity', 'Specificity',
               'Precision', 'F1', 'Optimized F1', 'Best Threshold', 'Brier (95% CI)',
               'H-L χ² (p)']
    table = plt.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.5)
    for (i,j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4F81BD')
    plt.axis('off'); plt.title(f'{set_type} Set - Model Performance Metrics'); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'Performance_Table_{dataset_name}_{set_type}.png'), dpi=600, bbox_inches='tight')
    plt.close()

def create_performance_matrix(results, dataset_name, set_type):
    prefix = 'Train' if set_type == "Training" else 'Test'
    data = []
    for name, res in results.items():
        auc_ci = res.get(f'{prefix}_AUC_CI', (0,0))
        auc_disp = f"{res[f'{prefix}_AUC']:.3f} [{auc_ci[0]:.3f}-{auc_ci[1]:.3f}]"
        brier_ci = res.get(f'{prefix}_Brier_CI', (0,0))
        brier_disp = f"{res[f'{prefix}_Brier']:.4f} [{brier_ci[0]:.4f}-{brier_ci[1]:.4f}]"
        hl_stat = res.get(f'{prefix}_HL_statistic', np.nan)
        hl_p = res.get(f'{prefix}_HL_pvalue', np.nan)
        hl_disp = f"{hl_stat:.2f}" if not np.isnan(hl_stat) else "N/A"
        hl_p_disp = f"{hl_p:.4f}" if not np.isnan(hl_p) else "N/A"
        data.append({
            'Model': name,
            'AUC (95% CI)': auc_disp,
            'Accuracy': f"{res[f'{prefix}_Accuracy']:.3f}",
            'Sensitivity': f"{res[f'{prefix}_Sensitivity']:.3f}",
            'Specificity': f"{res[f'{prefix}_Specificity']:.3f}",
            'Precision': f"{res[f'{prefix}_Precision']:.3f}",
            'F1': f"{res[f'{prefix}_F1']:.3f}",
            'Optimized_F1': f"{res[f'{prefix}_Optimized_F1']:.3f}",
            'Best_Threshold': f"{res[f'{prefix}_Best_Threshold']:.3f}",
            'AUPRC': f"{res[f'{prefix}_AUPRC']:.3f}",
            'Brier_Score (95% CI)': brier_disp,
            'Hosmer-Lemeshow χ²': hl_disp,
            'H-L p-value': hl_p_disp
        })
    df = pd.DataFrame(data)
    save_path = os.path.join(OUTPUT_DIR, f'performance_matrix_{dataset_name}_{set_type}.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    logger.info(f"Saved performance matrix: {save_path}")
    return df

def internal_validation(best_pipe, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, f1s = [], []
    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        clone_pipe = clone(best_pipe)
        clone_pipe.fit(X_tr, y_tr)
        proba = clone_pipe.predict_proba(X_te)[:, 1]
        pred = clone_pipe.predict(X_te)
        aucs.append(roc_auc_score(y_te, proba))
        f1s.append(f1_score(y_te, pred))
    return np.mean(aucs), np.std(aucs), np.mean(f1s), np.std(f1s)

# ========================== SHAP 解释（含数据表格输出） ==========================
def create_shap_explainer(model, X, model_name):
    logger.info(f"Creating SHAP explainer for {model_name}")
    try:
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
        else:
            classifier = model
        model_type = str(type(classifier)).lower()
        if 'svm' in model_type or 'svc' in model_name.lower():
            n_bg = min(100, len(X))
            if n_bg < 10:
                bg = X
            else:
                kmeans = KMeans(n_clusters=min(10, n_bg), n_init=10, random_state=42)
                kmeans.fit(X)
                bg = kmeans.cluster_centers_
            predict_fn = lambda x: model.predict_proba(x)
            explainer = shap.KernelExplainer(predict_fn, bg)
        elif 'lightgbm' in model_type or 'lgbm' in model_name.lower():
            explainer = shap.TreeExplainer(classifier, feature_perturbation="tree_path_dependent")
        elif 'xgboost' in model_type or 'xgb' in model_name.lower():
            explainer = shap.TreeExplainer(classifier)
        elif 'catboost' in model_type or 'cat' in model_name.lower():
            explainer = shap.TreeExplainer(classifier)
        elif 'randomforest' in model_type or 'rf' in model_name.lower():
            explainer = shap.TreeExplainer(classifier)
        elif 'logistic' in model_type or 'lr' in model_name.lower():
            explainer = shap.LinearExplainer(classifier, X)
        else:
            try:
                explainer = shap.TreeExplainer(classifier)
            except:
                bg = shap.sample(X, min(100, len(X)))
                predict_fn = lambda x: model.predict_proba(x)
                explainer = shap.KernelExplainer(predict_fn, bg)
        return explainer
    except Exception as e:
        logger.error(f"SHAP explainer creation failed: {str(e)}")
        return None

def plot_shap_for_paper(model, model_name, X, feature_names, dataset_name, set_type):
    logger.info(f"Generating SHAP plots for {model_name} on {set_type}")
    explainer = create_shap_explainer(model, X, model_name)
    if explainer is None:
        return
    sample_size = min(500, len(X))
    X_sample = X.sample(sample_size, random_state=42) if len(X) > sample_size else X
    try:
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        logger.error(f"SHAP value calculation failed: {str(e)}")
        return
    shap_vals_use = None
    if isinstance(shap_values, list):
        shap_vals_use = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_vals_use = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values[:, :, 0]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:
        shap_vals_use = shap_values
    if shap_vals_use is None or shap_vals_use.shape[0] != X_sample.shape[0] or shap_vals_use.shape[1] != X_sample.shape[1]:
        return

    # 输出 SHAP 值数据表格
    shap_df = pd.DataFrame(shap_vals_use, columns=feature_names)
    shap_csv_path = os.path.join(FIGURES_DIR, f'SHAP_values_{model_name}_{dataset_name}_{set_type}.csv')
    shap_df.to_csv(shap_csv_path, index=False)
    logger.info(f"SHAP values table saved: {shap_csv_path}")

    # 输出特征重要性表格
    mean_abs_shap = np.abs(shap_vals_use).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    importance_csv_path = os.path.join(FIGURES_DIR, f'SHAP_Importance_{model_name}_{dataset_name}_{set_type}.csv')
    importance_df.to_csv(importance_csv_path, index=False)
    logger.info(f"SHAP importance table saved: {importance_csv_path}")

    mapped_names = [FEATURE_NAME_MAP.get(f, f) for f in feature_names]

    plt.figure(figsize=(10, 8), dpi=600)
    shap.summary_plot(shap_vals_use, X_sample, feature_names=mapped_names, plot_type="dot", max_display=15, show=False)
    plt.title(f"SHAP Summary Plot - {model_name} ({set_type} Set)"); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'SHAP_Swarm_{model_name}_{dataset_name}_{set_type}.png'), dpi=600, bbox_inches='tight'); plt.close()

    plt.figure(figsize=(10, 6), dpi=600)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    top_n = min(15, len(sorted_idx))
    plt.barh([mapped_names[i] for i in sorted_idx[:top_n]][::-1], mean_abs_shap[sorted_idx[:top_n]][::-1], color='#1f77b4')
    plt.xlabel("Mean |SHAP Value|"); plt.title(f"Feature Importance - {model_name} ({set_type})"); plt.grid(axis='x', alpha=0.7); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'SHAP_Importance_{model_name}_{dataset_name}_{set_type}.png'), dpi=600, bbox_inches='tight'); plt.close()

# ========================== 主流程 ==========================
def main():
    try:
        logger.info(f"Initial memory: {psutil.virtual_memory().percent}%")
        df = load_and_preprocess_data(DATA_PATH)
        for col in df.columns:
            if df[col].dtype == 'float64': df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64': df[col] = df[col].astype('int32')
        X = df.drop(columns=['Outcome'])
        y = df['Outcome']
        logger.info(f"Class balance:\n{y.value_counts(normalize=True)}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # 直接使用原始数据，无缺失值填补
        sel_feat, lasso_model = feature_selection_with_lasso(X_train, y_train, 10, 12)
        X_train_sel = X_train[sel_feat]
        X_test_sel = X_test[sel_feat]

        plot_lasso_cv_results(lasso_model, lasso_model.X_train_, lasso_model.y_train_, FIGURES_DIR, "Optimized")
        plot_lasso_path(lasso_model, FIGURES_DIR, "Optimized", sel_feat)
        detect_multicollinearity(X_train_sel, FIGURES_DIR, "Optimized", "Train_")

        results = train_models_with_cv(X_train_sel, y_train, X_test_sel, y_test)

        logger.info("\n=== Hosmer-Lemeshow Calibration Test ===")
        for name, res in results.items():
            train_hl = res.get('Train_HL_statistic', np.nan)
            train_p = res.get('Train_HL_pvalue', np.nan)
            test_hl = res.get('Test_HL_statistic', np.nan)
            test_p = res.get('Test_HL_pvalue', np.nan)
            logger.info(f"{name}: Train H-L χ²={train_hl:.2f}, p={train_p:.4f} | Test H-L χ²={test_hl:.2f}, p={test_p:.4f}")

        create_performance_matrix(results, "Optimized", "Training")
        create_performance_matrix(results, "Optimized", "Test")

        for stype in ['Training', 'Test']:
            X_plot = X_train_sel if stype == 'Training' else X_test_sel
            y_plot = y_train if stype == 'Training' else y_test
            plot_roc_curves(results, X_plot, y_plot, "Optimized", stype)
            plot_pr_curves(results, X_plot, y_plot, "Optimized", stype)
            plot_calibration_curves(results, X_plot, y_plot, "Optimized", stype)
            plot_dca_curve(results, X_plot, y_plot, "Optimized", stype)
            plot_model_comparison(results, "Optimized", stype)
            plot_f1_comparison(results, "Optimized", stype)
            plot_performance_table(results, "Optimized", stype)

        logger.info("\n=== AUC Pairwise Comparison (Bootstrap DeLong) ===")
        test_y = y_test
        names = list(results.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                n1, n2 = names[i], names[j]
                prob1 = results[n1]['Model'].predict_proba(X_test_sel)[:, 1]
                prob2 = results[n2]['Model'].predict_proba(X_test_sel)[:, 1]
                diff, p = delong_roc_test(test_y, prob1, prob2)
                logger.info(f"{n1} vs {n2}: diff={diff:.4f}, p={p:.4f}")

        best_name = max(results, key=lambda n: results[n]['Test_Optimized_F1'])
        best_pipe = results[best_name]['Model']
        logger.info(f"Best model: {best_name}")

        mean_auc, std_auc, mean_f1, std_f1 = internal_validation(best_pipe, X_train_sel, y_train)
        logger.info(f"CV AUC: {mean_auc:.3f}±{std_auc:.3f}, F1: {mean_f1:.3f}±{std_f1:.3f}")

        plot_shap_for_paper(best_pipe, best_name, X_train_sel, sel_feat, "Optimized", "Training")
        plot_shap_for_paper(best_pipe, best_name, X_test_sel, sel_feat, "Optimized", "Test")

        joblib.dump({'pipe': best_pipe, 'features': sel_feat, 'threshold': results[best_name]['Test_Best_Threshold']},
                    os.path.join(OUTPUT_DIR, 'best_model.pkl'))

        logger.info("All analyses completed successfully.")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()