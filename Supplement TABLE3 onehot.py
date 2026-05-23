# -*- coding: utf-8 -*-
"""
敏感性分析：将 mRS 和 NIHSS 作为分类变量（独热编码）重新建模
与原始连续变量模型对比，生成 Supplementary Table 3

修正说明：
- OneHotEncoder 的 sparse 参数改为 sparse_output
- 移除 XGBoost 已废弃的 use_label_encoder 参数
- 修正 SMOTE 的 k_neighbors 计算方式（基于少数类样本数）
- 增加特征选择的边界保护与混淆矩阵安全检查
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
                             precision_score, f1_score, brier_score_loss,
                             confusion_matrix)
from sklearn.calibration import calibration_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import os, warnings, logging

# ======================= 配置 =======================
DATA_PATH = r"D:\desktop\datafinalversion.csv"
OUTPUT_DIR = r"D:\desktop\sensitivity_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
warnings.filterwarnings('ignore')
np.random.seed(42)

# 模型颜色（绘图用不到，保留兼容）
MODEL_COLORS = {
    'XGBoost': '#1f77b4', 'SVM': '#ff7f0e', 'LightGBM': '#2ca02c',
    'LR': '#d62728', 'RF': '#9467bd', 'MLP': '#8c564b',
    'GBDT': '#e377c2', 'CatBoost': '#7f7f7f'
}

# ======================= 工具函数 =======================
def load_and_clean_data(path):
    df = pd.read_csv(path)
    df['Outcome'] = df['Outcome'].astype(int)
    for col in df.columns:
        if col == 'Outcome': continue
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                df[col] = pd.Categorical(df[col]).codes
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def binarize_mRS_NIHSS(X):
    """
    将 mRS 和 NIHSS 转换为分类变量（独热编码）。
    切点选择：
    - mRS: 0-2 (轻度) vs 3-6 (中重度) -> 二分类
    - NIHSS: 0-4 (轻度) vs 5-15 (中度) vs 16+ (重度) -> 三分类
    这里为简洁展示，均做二分类 + 独热编码，也可使用更细分箱。
    独热编码删除第一个类别以避免共线性（drop='first'）。
    """
    X = X.copy()
    # mRS 二分类
    X['mRS_cat'] = np.where(X['mRS'] <= 2, 0, 1)
    # NIHSS 二分类（采用常见轻度 vs 中重度切点 5）
    X['NIHSS_cat'] = np.where(X['NIHSS'] <= 5, 0, 1)
    # 独热编码 (sparse_output=False)
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded = encoder.fit_transform(X[['mRS_cat', 'NIHSS_cat']])
    cat_cols = [f"mRS_3to6", f"NIHSS_6plus"]  # 新特征名
    X_encoded = pd.DataFrame(encoded, columns=cat_cols, index=X.index)
    # 删除原始连续变量并合并编码特征
    X = X.drop(columns=['mRS', 'NIHSS'])
    X = pd.concat([X, X_encoded], axis=1)
    return X, encoder

def feature_selection_lasso(X_train, y_train, min_feat=10, max_feat=12):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lasso_cv = LassoCV(alphas=np.logspace(-6, 6, 100), cv=10, max_iter=10000, random_state=42)
    lasso_cv.fit(X_scaled, y_train)
    coefs = np.abs(lasso_cv.coef_)
    # 修正：确保选择的特征数不超过实际特征数
    n_nonzero = np.sum(coefs > 1e-5)
    n = max(min_feat, min(max_feat, n_nonzero))
    n = min(n, X_train.shape[1])  # 边界保护
    # 取系数绝对值最大的前 n 个特征
    selected_idx = np.argsort(coefs)[::-1][:n]
    selected = X_train.columns[selected_idx].tolist()
    logger.info(f"Selected {len(selected)} features: {selected}")
    return selected, lasso_cv

def train_models(X_train, y_train, X_test, y_test):
    """与 V15 相同的模型和超参数网格，并修正了 SMOTE 的 k_neighbors 参数"""
    # 计算少数类的最小样本数，用于动态设置 SMOTE 的 k_neighbors
    class_counts = y_train.value_counts()
    min_class_count = class_counts.min()
    # k_neighbors 默认 5，但不能超过 min_class_count-1
    k_neighbors_smote = min(5, min_class_count - 1) if min_class_count > 1 else 1

    models = {
        'LR': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'SVM': SVC(probability=True, kernel='linear', class_weight='balanced', random_state=42),
        'MLP': MLPClassifier(max_iter=1000, early_stopping=True, hidden_layer_sizes=(30,15), alpha=0.5, random_state=42),
        'LightGBM': LGBMClassifier(verbose=-1, reg_alpha=3.0, reg_lambda=3.0, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', reg_alpha=3.0, reg_lambda=3.0, random_state=42),
        'RF': RandomForestClassifier(class_weight='balanced', max_depth=5, min_samples_leaf=20, random_state=42),
        'GBDT': GradientBoostingClassifier(max_depth=3, min_samples_leaf=20, random_state=42),
        'CatBoost': CatBoostClassifier(verbose=False, depth=4, l2_leaf_reg=3.0, random_seed=42)
    }
    param_grids = {
        'LR': {'classifier__C': [0.01, 0.1, 1, 10], 'classifier__penalty': ['l1','l2'], 'classifier__solver': ['liblinear']},
        'SVM': {'classifier__C': [0.01, 0.1, 1, 10], 'classifier__gamma': ['scale','auto'], 'classifier__kernel': ['linear']},
        'MLP': {'classifier__hidden_layer_sizes': [(30,),(50,),(30,15)], 'classifier__alpha': [0.01,0.1], 'classifier__learning_rate_init': [0.001,0.01], 'classifier__early_stopping': [True]},
        'LightGBM': {'classifier__learning_rate': [0.05], 'classifier__n_estimators': [200], 'classifier__num_leaves': [15,31], 'classifier__max_depth': [5,7], 'classifier__min_child_samples': [50], 'classifier__reg_alpha': [1.0,3.0], 'classifier__reg_lambda': [1.0,3.0], 'classifier__subsample': [0.8]},
        'XGBoost': {'classifier__learning_rate': [0.05], 'classifier__max_depth': [3,5], 'classifier__n_estimators': [100], 'classifier__subsample': [0.8], 'classifier__colsample_bytree': [0.6,0.8], 'classifier__reg_alpha': [1.0], 'classifier__reg_lambda': [1.0], 'classifier__gamma': [0.1]},
        'RF': {'classifier__n_estimators': [200], 'classifier__max_depth': [5,10], 'classifier__min_samples_split': [10], 'classifier__min_samples_leaf': [4], 'classifier__max_features': ['sqrt']},
        'GBDT': {'classifier__learning_rate': [0.05], 'classifier__n_estimators': [100], 'classifier__max_depth': [3], 'classifier__min_samples_leaf': [20], 'classifier__subsample': [0.8]},
        'CatBoost': {'classifier__iterations': [200], 'classifier__learning_rate': [0.05], 'classifier__depth': [6,8], 'classifier__l2_leaf_reg': [3.0,5.0], 'classifier__border_count': [64], 'classifier__subsample': [0.8]}
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        # 使用动态的 k_neighbors 构建 SMOTE
        pipe = ImbPipeline([('scaler', StandardScaler()),
                            ('smote', SMOTE(random_state=42, k_neighbors=k_neighbors_smote)),
                            ('classifier', model)])
        grid = GridSearchCV(pipe, param_grids[name], cv=cv, scoring='roc_auc', n_jobs=1, verbose=0)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:,1]
        # 计算混淆矩阵，增加类别数量检查
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            # 若测试集只有一个类别，将对应指标置0或计算有偏值
            tn = fp = fn = tp = 0
            specificity = 0.0
        results[name] = {
            'AUC': roc_auc_score(y_test, y_prob),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Sensitivity': recall_score(y_test, y_pred),
            'Specificity': specificity,
            'F1': f1_score(y_test, y_pred),
            'Brier': brier_score_loss(y_test, y_prob)
        }
    return results

# ======================= 主流程 =======================
def main():
    df = load_and_clean_data(DATA_PATH)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # 分成训练/测试（固定种子）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # ---- 原始连续变量 ----
    sel_cont, _ = feature_selection_lasso(X_train, y_train)
    X_train_cont = X_train[sel_cont]
    X_test_cont = X_test[sel_cont]
    results_cont = train_models(X_train_cont, y_train, X_test_cont, y_test)

    # ---- 分类变量（mRS, NIHSS 独热编码） ----
    X_cat, encoder = binarize_mRS_NIHSS(X)
    # 重新划分（使用相同随机种子确保可比性）
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
        X_cat, y, test_size=0.3, stratify=y, random_state=42)
    sel_cat, _ = feature_selection_lasso(X_train_cat, y_train_cat)
    X_train_cat_sel = X_train_cat[sel_cat]
    X_test_cat_sel = X_test_cat[sel_cat]
    results_cat = train_models(X_train_cat_sel, y_train_cat, X_test_cat_sel, y_test_cat)

    # ---- 构建对比表 (Supplementary Table 3) ----
    models_list = list(results_cont.keys())
    rows = []
    for model in models_list:
        for metric in ['AUC','Accuracy','Sensitivity','Specificity','F1','Brier']:
            rows.append([model, metric,
                         f"{results_cont[model][metric]:.3f}",
                         f"{results_cat[model][metric]:.3f}"])
    df_table = pd.DataFrame(rows, columns=['Model','Metric','Continuous','Categorical'])
    csv_path = os.path.join(OUTPUT_DIR, 'Supplementary_Table3_Sensitivity.csv')
    df_table.to_csv(csv_path, index=False)
    logger.info(f"Supplementary Table 3 saved to {csv_path}")

    # 打印最佳模型信息
    best_cont = max(models_list, key=lambda m: results_cont[m]['AUC'])
    best_cat = max(models_list, key=lambda m: results_cat[m]['AUC'])
    logger.info(f"Best continuous model: {best_cont} AUC={results_cont[best_cont]['AUC']:.3f}")
    logger.info(f"Best categorical model: {best_cat} AUC={results_cat[best_cat]['AUC']:.3f}")

if __name__ == "__main__":
    main()