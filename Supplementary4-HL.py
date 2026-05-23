# -*- coding: utf-8 -*-
"""
校准指标提取脚本 —— 输出 Supplementary Table 4
- Hosmer‑Lemeshow χ² 和 p 值
- 校准斜率 (calibration slope) 和截距 (calibration intercept)

修正说明：
- 修复 SMOTE 的 k_neighbors 参数，基于少数类样本数计算
- 删除 XGBoost 已彻底移除的 use_label_encoder 参数
- 特征选择增加边界保护
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import chi2
import warnings, logging, os

# ======================= 配置 =======================
DATA_PATH = r"D:\desktop\datafinalversion.csv"
OUTPUT_DIR = r"D:\desktop\sensitivity_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
warnings.filterwarnings('ignore')
np.random.seed(42)

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

def lasso_feature_selection(X_train, y_train, min_feat=10, max_feat=12):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lasso = LassoCV(alphas=np.logspace(-6, 6, 100), cv=10, max_iter=10000, random_state=42)
    lasso.fit(X_scaled, y_train)
    coefs = np.abs(lasso.coef_)
    n_nonzero = np.sum(coefs > 1e-5)
    n = max(min_feat, min(max_feat, n_nonzero))
    n = min(n, X_train.shape[1])  # 边界保护
    selected = X_train.columns[np.argsort(coefs)[::-1][:n]].tolist()
    logger.info(f"Selected features: {selected}")
    return selected, lasso

def hosmer_lemeshow_test(y_true, y_proba, n_bins=10):
    bins = np.percentile(y_proba, np.linspace(0, 100, n_bins+1))
    bins[-1] += 1e-8
    binned = np.digitize(y_proba, bins) - 1
    binned = np.clip(binned, 0, n_bins-1)
    obs = np.array([y_true[binned==i].sum() for i in range(n_bins)])
    exp = np.array([y_proba[binned==i].sum() for i in range(n_bins)])
    cnt = np.array([(binned==i).sum() for i in range(n_bins)])
    mask = cnt > 0
    obs, exp, cnt = obs[mask], exp[mask], cnt[mask]
    if len(cnt) < 2:
        return np.nan, np.nan
    # 添加微小值避免分母为0
    denom = exp * (1 - exp/cnt) + 1e-10
    hl = np.sum((obs - exp)**2 / denom)
    df = len(cnt) - 2
    p = chi2.sf(hl, df) if df > 0 else np.nan
    return hl, p

def calibration_slope_intercept(y_true, y_proba):
    """用 logistic 回归拟合 logit(y) ~ logit(p)，返回斜率 (slope) 和截距 (intercept)"""
    eps = 1e-8
    logit_p = np.log(np.clip(y_proba, eps, 1-eps) / (1 - np.clip(y_proba, eps, 1-eps)))
    mask = np.isfinite(logit_p)
    if np.sum(mask) < 10:
        return np.nan, np.nan
    logit_p_finite = logit_p[mask]
    y_finite = y_true[mask]
    cal_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    cal_model.fit(logit_p_finite.reshape(-1, 1), y_finite)
    return cal_model.coef_[0][0], cal_model.intercept_[0]

# ======================= 模型训练 =======================
def train_all_models(X_train, y_train, X_test, y_test):
    # 动态计算 SMOTE 的 k_neighbors
    class_counts = y_train.value_counts()
    min_class_count = class_counts.min()
    k_neighbors_smote = min(5, min_class_count - 1) if min_class_count > 1 else 1

    models = {
        'LR': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'SVM': SVC(probability=True, kernel='linear', class_weight='balanced', random_state=42),
        'MLP': MLPClassifier(max_iter=1000, early_stopping=True, hidden_layer_sizes=(30,15),
                             alpha=0.5, random_state=42),
        'LightGBM': LGBMClassifier(verbose=-1, reg_alpha=3.0, reg_lambda=3.0, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', reg_alpha=3.0, reg_lambda=3.0,
                                 random_state=42),  # 移除 use_label_encoder
        'RF': RandomForestClassifier(class_weight='balanced', max_depth=5,
                                     min_samples_leaf=20, random_state=42),
        'GBDT': GradientBoostingClassifier(max_depth=3, min_samples_leaf=20, random_state=42),
        'CatBoost': CatBoostClassifier(verbose=False, depth=4, l2_leaf_reg=3.0, random_seed=42)
    }
    param_grids = {
        'LR': {'classifier__C': [0.01, 0.1, 1, 10], 'classifier__penalty': ['l1','l2'],
               'classifier__solver': ['liblinear']},
        'SVM': {'classifier__C': [0.01, 0.1, 1, 10], 'classifier__gamma': ['scale','auto'],
                'classifier__kernel': ['linear']},
        'MLP': {'classifier__hidden_layer_sizes': [(30,),(50,),(30,15)],
                'classifier__alpha': [0.01,0.1], 'classifier__learning_rate_init': [0.001,0.01],
                'classifier__early_stopping': [True]},
        'LightGBM': {'classifier__learning_rate': [0.05], 'classifier__n_estimators': [200],
                     'classifier__num_leaves': [15,31], 'classifier__max_depth': [5,7],
                     'classifier__min_child_samples': [50], 'classifier__reg_alpha': [1.0,3.0],
                     'classifier__reg_lambda': [1.0,3.0], 'classifier__subsample': [0.8]},
        'XGBoost': {'classifier__learning_rate': [0.05], 'classifier__max_depth': [3,5],
                    'classifier__n_estimators': [100], 'classifier__subsample': [0.8],
                    'classifier__colsample_bytree': [0.6,0.8], 'classifier__reg_alpha': [1.0],
                    'classifier__reg_lambda': [1.0], 'classifier__gamma': [0.1]},
        'RF': {'classifier__n_estimators': [200], 'classifier__max_depth': [5,10],
               'classifier__min_samples_split': [10], 'classifier__min_samples_leaf': [4],
               'classifier__max_features': ['sqrt']},
        'GBDT': {'classifier__learning_rate': [0.05], 'classifier__n_estimators': [100],
                 'classifier__max_depth': [3], 'classifier__min_samples_leaf': [20],
                 'classifier__subsample': [0.8]},
        'CatBoost': {'classifier__iterations': [200], 'classifier__learning_rate': [0.05],
                     'classifier__depth': [6,8], 'classifier__l2_leaf_reg': [3.0,5.0],
                     'classifier__border_count': [64], 'classifier__subsample': [0.8]}
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    calib_results = []
    for name, model in models.items():
        pipe = ImbPipeline([('scaler', StandardScaler()),
                            ('smote', SMOTE(random_state=42, k_neighbors=k_neighbors_smote)),
                            ('classifier', model)])
        grid = GridSearchCV(pipe, param_grids[name], cv=cv, scoring='roc_auc', n_jobs=1, verbose=0)
        grid.fit(X_train, y_train)
        best_pipe = grid.best_estimator_
        y_prob = best_pipe.predict_proba(X_test)[:, 1]
        hl_stat, hl_p = hosmer_lemeshow_test(y_test, y_prob)
        slope, intercept = calibration_slope_intercept(y_test, y_prob)
        calib_results.append({
            'Model': name,
            'H-L χ²': f"{hl_stat:.2f}" if not np.isnan(hl_stat) else "N/A",
            'H-L p-value': f"{hl_p:.4f}" if not np.isnan(hl_p) else "N/A",
            'Calibration Slope': f"{slope:.3f}" if not np.isnan(slope) else "N/A",
            'Calibration Intercept': f"{intercept:.3f}" if not np.isnan(intercept) else "N/A"
        })
        logger.info(f"{name}: H-L χ²={hl_stat:.2f}, p={hl_p:.4f}, Slope={slope:.3f}, Intercept={intercept:.3f}")
    return pd.DataFrame(calib_results)

# ======================= 主流程 =======================
def main():
    df = load_and_clean_data(DATA_PATH)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # LASSO 特征选择
    sel_feat, _ = lasso_feature_selection(X_train, y_train)
    X_train_sel = X_train[sel_feat]
    X_test_sel = X_test[sel_feat]

    # 训练并计算校准指标
    calib_df = train_all_models(X_train_sel, y_train, X_test_sel, y_test)

    # 保存
    csv_path = os.path.join(OUTPUT_DIR, 'Supplementary_Table4_Calibration.csv')
    calib_df.to_csv(csv_path, index=False)
    logger.info(f"Supplementary Table 4 saved to {csv_path}")
    print(calib_df)

if __name__ == "__main__":
    main()