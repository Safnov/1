# -*- coding: utf-8 -*-
"""
时间验证 (Temporal Validation) – 修正增强版
使用前 70% 时间的数据训练，后 30% 的数据测试。
所有预处理参数均仅从训练集学习，严格避免数据泄漏。

新增功能：
- 混淆矩阵类别完整性检查
- 特征选择边界保护
- 绘制并保存 ROC 曲线
- 保存详细评估结果到 CSV 文件
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
                             precision_score, f1_score, brier_score_loss,
                             confusion_matrix, roc_curve, auc)
import matplotlib.pyplot as plt
import warnings
import logging
import os

# ======================== 配置 ========================
DATA_PATH = r"D:\desktop\1234.csv"          # 原始数据（含 time 列）
OUTPUT_DIR = r"D:\desktop\temporal_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
warnings.filterwarnings('ignore')
np.random.seed(42)

# ======================== 工具函数 ========================
def calculate_brier_ci(y_true, y_proba, n_bootstraps=2000):
    """使用Bootstrap计算Brier分数的95%置信区间"""
    rng = np.random.default_rng(42)
    # 统一为numpy数组，避免隐式转换警告
    y_true_arr = y_true.values if hasattr(y_true, 'values') else np.asarray(y_true)
    n = len(y_true_arr)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        scores.append(brier_score_loss(y_true_arr[idx], y_proba[idx]))
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

def auc_ci_bootstrap(y_true, y_proba, n_bootstraps=2000):
    """使用Bootstrap计算AUC的95%置信区间"""
    rng = np.random.default_rng(42)
    y_true_arr = y_true.values if hasattr(y_true, 'values') else np.asarray(y_true)
    n = len(y_true_arr)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        scores.append(roc_auc_score(y_true_arr[idx], y_proba[idx]))
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    if 'Outcome' not in df.columns:
        raise ValueError("Missing 'Outcome' column")
    df['Outcome'] = df['Outcome'].astype(int)
    if df['time'].dtype == object:
        df['time'] = pd.to_datetime(df['time'])
    for col in df.columns:
        if col in ('Outcome', 'time'): continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def feature_selection_lasso(X_train, y_train, min_feats=10, max_feats=12):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    alphas = np.logspace(-6, 6, 100)
    lasso = LassoCV(alphas=alphas, cv=10, max_iter=10000, random_state=42, n_jobs=1)
    lasso.fit(X_scaled, y_train)
    coefs = np.abs(lasso.coef_)
    n_nonzero = np.sum(coefs > 1e-5)
    # 动态计算所选特征数，并限制不超过总特征数
    n_sel = max(min_feats, min(max_feats, n_nonzero))
    n_sel = min(n_sel, X_train.shape[1])
    selected = X_train.columns[np.argsort(coefs)[::-1][:n_sel]].tolist()
    logger.info(f"Temporal CV: selected {len(selected)} features (min_coef={coefs[np.argsort(coefs)[::-1][:n_sel]].min():.6f}): {selected}")
    return selected

def save_results_to_csv(results_dict, filepath):
    """将评估指标保存为CSV文件"""
    df = pd.DataFrame([results_dict])
    # 列顺序优化
    cols_order = ['AUC', 'AUC_95CI_Low', 'AUC_95CI_High', 
                  'Brier', 'Brier_95CI_Low', 'Brier_95CI_High',
                  'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1',
                  'Train_Size', 'Test_Size']
    df = df[[c for c in cols_order if c in df.columns]]
    df.to_csv(filepath, index=False)
    logger.info(f"Results saved to {filepath}")

def plot_roc_curve(y_true, y_proba, auc_val, save_path):
    """绘制并保存ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Temporal Validation ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"ROC curve saved to {save_path}")

# ======================== 主流程 ========================
def main():
    # 1. 加载数据
    df = load_and_preprocess(DATA_PATH)
    logger.info(f"Loaded {df.shape[0]} records")

    # 2. 按时间排序并划分
    df = df.sort_values('time').reset_index(drop=True)
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    logger.info(f"Train period: {train_df['time'].min()} to {train_df['time'].max()}")
    logger.info(f"Test period:  {test_df['time'].min()} to {test_df['time'].max()}")
    logger.info(f"Sizes: train={len(train_df)}, test={len(test_df)}")

    feature_cols = [c for c in df.columns if c not in ('Outcome', 'time')]
    X_train_raw = train_df[feature_cols]
    y_train = train_df['Outcome']
    X_test_raw = test_df[feature_cols]
    y_test = test_df['Outcome']

    # 3. 缺失值处理（仅从训练集学习）
    imputer = IterativeImputer(max_iter=20, random_state=42, skip_complete=True)
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=feature_cols)
    X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=feature_cols)

    # 4. 特征选择（仅从训练集学习）
    selected = feature_selection_lasso(X_train_imp, y_train, min_feats=10, max_feats=12)
    X_train_sel = X_train_imp[selected]
    X_test_sel = X_test_imp[selected]

    # 5. 标准化（仅从训练集学习）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    # 6. 训练 SVM (最佳参数)
    svm = SVC(probability=True, kernel='linear', C=0.01, gamma='scale',
              class_weight='balanced', random_state=42)
    svm.fit(X_train_scaled, y_train)

    # 7. 测试集评估
    y_proba = svm.predict_proba(X_test_scaled)[:, 1]
    y_pred = svm.predict(X_test_scaled)

    # 混淆矩阵安全解包
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn+fp) if (tn+fp) > 0 else 0.0
        sens = recall_score(y_test, y_pred)  # 等同于 tp/(tp+fn)
    else:
        # 测试集仅包含单一类别时的降级处理
        logger.warning("Confusion matrix is not 2x2, metrics will be limited.")
        tn = fp = fn = tp = 0
        spec = 0.0
        sens = recall_score(y_test, y_pred, zero_division=0)
    
    auc_val = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    auc_ci = auc_ci_bootstrap(y_test, y_proba)
    brier_ci = calculate_brier_ci(y_test, y_proba)

    # 收集所有指标
    results = {
        'AUC': round(auc_val, 4),
        'AUC_95CI_Low': round(auc_ci[0], 4),
        'AUC_95CI_High': round(auc_ci[1], 4),
        'Brier': round(brier, 4),
        'Brier_95CI_Low': round(brier_ci[0], 4),
        'Brier_95CI_High': round(brier_ci[1], 4),
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Sensitivity': round(sens, 4),
        'Specificity': round(spec, 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'F1': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'Train_Size': len(y_train),
        'Test_Size': len(y_test)
    }

    # 8. 输出结果到控制台
    logger.info("===== Temporal Validation Results =====")
    for k, v in results.items():
        logger.info(f"{k}: {v}")

    # 9. 保存结果 CSV
    csv_path = os.path.join(OUTPUT_DIR, 'temporal_validation_results.csv')
    save_results_to_csv(results, csv_path)

    # 10. 绘制并保存 ROC 曲线
    roc_path = os.path.join(OUTPUT_DIR, 'temporal_roc_curve.png')
    plot_roc_curve(y_test, y_proba, auc_val, roc_path)

if __name__ == "__main__":
    main()