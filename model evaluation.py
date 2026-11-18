import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, LeaveOneOut, KFold
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
from joblib import Parallel, delayed


# ---------- 0. 最优超参数(CatBoost训练得到的) ----------
BEST_PARAMS = {
    'bootstrap_type': 'Bernoulli',
    'border_count': 64,
    'depth': 6,
    'iterations': 500,
    'l2_leaf_reg': 12,
    'leaf_estimation_iterations': 1,
    'learning_rate': 0.12,
    'loss_function': 'RMSE',
    'model_size_reg': 0.5,
    'random_strength': 5,
    'sampling_frequency': 'PerTree',
    'subsample': 0.6,
    'task_type': 'CPU',
    'random_seed': 42,
    'thread_count': -1,
    'verbose': False
}

# ---------- 1. 读原始 SMILES ----------
df = pd.read_csv('SMILES.csv')
df = df.dropna(subset=[df.columns[0], df.columns[1]])
df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
df.iloc[:, 1] = df.iloc[:, 1].astype(str).str.strip()
df = df[(df.iloc[:, 0] != '') & (df.iloc[:, 1] != '')]

# ---------- 2. 计算描述符 ----------
all_desc = [x[0] for x in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(all_desc)

def calc_desc(smiles, prefix):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series([np.nan]*len(all_desc), index=[f'{prefix}_{d}' for d in all_desc])
    vals = calc.CalcDescriptors(mol)
    return pd.Series(vals, index=[f'{prefix}_{d}' for d in all_desc])

results = []
for _, row in df.iterrows():
    host_smiles, guest_smiles, lgks = row.iloc[0], row.iloc[1], row.iloc[2]
    host_s = calc_desc(host_smiles, 'host')
    guest_s = calc_desc(guest_smiles, 'guest')
    com_s = {}
    for d in all_desc:
        hv, gv = host_s[f'host_{d}'], guest_s[f'guest_{d}']
        if pd.isna(hv) or pd.isna(gv):
            com_s[f'com_{d}'] = np.nan
        else:
            prod = hv * gv
            com_s[f'com_{d}'] = np.sqrt(abs(prod)) if prod >= 0 else -np.sqrt(abs(prod))
    summary = pd.concat([host_s, guest_s, pd.Series(com_s)])
    summary['lgKs'] = lgks
    results.append(summary)

data = pd.DataFrame(results)

# ---------- 3. 加载GA筛选的特征描述符 ----------
sel_feat = pd.read_csv('selected_features.csv')['Name'].tolist()
missing = [c for c in sel_feat if c not in data.columns]
if missing:
    raise RuntimeError(f'缺失特征列：{missing}')
X_df = data[sel_feat].copy()
y = data['lgKs'].astype(float)

# ---------- 4. 训练/测试 split ----------
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# ---------- 5. 缺失值+标准化 ----------
pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])
X_tr_imp = pipe.fit_transform(X_train)
X_te_imp = pipe.transform(X_test)

# ---------- 6. 训练并保存模型 ----------
model = CatBoostRegressor(**BEST_PARAMS)
model.fit(X_tr_imp, y_train)
model.save_model('0.81boost.cbm')
print('>>> 0.81boost.cbm 已保存完成！')

y_tr_pred = model.predict(X_tr_imp)
y_te_pred = model.predict(X_te_imp)

r2_tr = r2_score(y_train, y_tr_pred)
r2_te = r2_score(y_test, y_te_pred)

print(f'训练集 R²: {r2_tr:.4f}')
print(f'测试集  R²: {r2_te:.4f}')

# ---------- 6-1. 保存最佳参数 ----------
params_json = 'best_params.json'
with open(params_json, 'w', encoding='utf-8') as f:
    json.dump(BEST_PARAMS, f, ensure_ascii=False, indent=4, default=str)
print(f'>>> 最佳参数已保存到 {params_json}')

# ---------- 7. 评价指标 ----------
def reg_metrics(y_true, y_pred, tag):
    return {
        'Set': tag,
        'R²': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
    }

# 7-1 训练 / 测试
train_metrics = reg_metrics(y_train, y_tr_pred, 'Train')
test_metrics  = reg_metrics(y_test, y_te_pred, 'Test')

# ============================================================
# 7-2. 五折交叉验证
#     报告每一折的 R² / MAE / RMSE
#     最后以「均值 ± 标准差」形式汇总
# ============================================================
print('\n' + '=' * 70)
print('五折交叉验证（5-Fold CV）')
print('=' * 70)

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=12)
y_cv_pred = np.empty(len(y))

fold_r2 = []
fold_mae = []
fold_rmse = []
fold_records = []

for fold, (tr_idx, te_idx) in enumerate(kf.split(X_df), 1):
    pipe_fold = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    X_tr_fold = pipe_fold.fit_transform(X_df.iloc[tr_idx])
    X_te_fold = pipe_fold.transform(X_df.iloc[te_idx])

    m = CatBoostRegressor(**BEST_PARAMS)
    m.fit(X_tr_fold, y.iloc[tr_idx])
    y_cv_pred[te_idx] = m.predict(X_te_fold)

    y_te_fold = y.iloc[te_idx]
    y_pred_fold = y_cv_pred[te_idx]

    r2_fold = r2_score(y_te_fold, y_pred_fold)
    mae_fold = mean_absolute_error(y_te_fold, y_pred_fold)
    rmse_fold = np.sqrt(mean_squared_error(y_te_fold, y_pred_fold))

    fold_r2.append(r2_fold)
    fold_mae.append(mae_fold)
    fold_rmse.append(rmse_fold)

    fold_records.append({
        'Fold': fold,
        'N_train': len(tr_idx),
        'N_val': len(te_idx),
        'R²': r2_fold,
        'MAE': mae_fold,
        'RMSE': rmse_fold
    })

    print(f'  Fold {fold}/{N_FOLDS}:  '
          f'N_train={len(tr_idx):4d}  N_val={len(te_idx):4d}  '
          f'R²={r2_fold:.4f}  MAE={mae_fold:.4f}  RMSE={rmse_fold:.4f}')

# 计算均值和样本标准差 
mean_r2, std_r2 = np.mean(fold_r2), np.std(fold_r2, ddof=1)
mean_mae, std_mae = np.mean(fold_mae), np.std(fold_mae, ddof=1)
mean_rmse, std_rmse = np.mean(fold_rmse), np.std(fold_rmse, ddof=1)

print('-' * 70)
print(f'  Mean ± Std:  '
      f'R²={mean_r2:.4f} ± {std_r2:.4f}  '
      f'MAE={mean_mae:.4f} ± {std_mae:.4f}  '
      f'RMSE={mean_rmse:.4f} ± {std_rmse:.4f}')

# 保存每折结果
fold_df = pd.DataFrame(fold_records)
fold_df.to_csv('5fold_cv_folds.csv', index=False, float_format='%.6f')
print('>>> 每折明细已保存到 5fold_cv_folds.csv')

# 保存均值 ± 标准差
pd.DataFrame([{
    'Set':       f'{N_FOLDS}-Fold CV (mean ± std)',
    'R²_mean':   mean_r2,
    'R²_std':    std_r2,
    'MAE_mean':  mean_mae,
    'MAE_std':   std_mae,
    'RMSE_mean': mean_rmse,
    'RMSE_std':  std_rmse,
}]).to_csv('5fold_cv_summary.csv', index=False, float_format='%.6f')
print('>>> 均值 ± 标准差已保存到 5fold_cv_summary.csv')

fivefold_metrics = {
    'Set':  f'{N_FOLDS}-Fold CV',
    'R²':   mean_r2,
    'MAE':  mean_mae,
    'RMSE': mean_rmse,
}

# ============================================================
# 7-3. 留一法交叉验证
# ============================================================
print('\n>>> 正在留一法交叉验证 ……')
loo = LeaveOneOut()
y_loo_pred = np.empty(len(y_train))
for tr_idx, te_idx in loo.split(X_tr_imp):
    m = CatBoostRegressor(**BEST_PARAMS)
    m.fit(X_tr_imp[tr_idx], y_train.iloc[tr_idx])
    y_loo_pred[te_idx] = m.predict(X_tr_imp[te_idx])

loo_metrics = reg_metrics(y_train, y_loo_pred, 'LOO-CV')

# ---------- 7-4. 汇总打印----------
summary = pd.DataFrame([train_metrics, test_metrics, fivefold_metrics, loo_metrics])
print('\n' + '=' * 70)
print('汇总指标')
print('=' * 70)
print(summary.to_string(index=False))

# ---------- 7-5. 保存 combined_metrics.csv ----------
summary_with_n = summary.copy()
summary_with_n.insert(
    1, 'N',
    [len(y_train), len(y_test), len(y), len(y_train)]
)
summary_with_n.to_csv('combined_metrics.csv', index=False, float_format='%.4f')
print('>>> 汇总指标已保存到 combined_metrics.csv')


# ============================================================
# 8. 通用 plot_parity：LOO / 外部验证 / 整体验证
# ============================================================
def plot_parity(y_true, y_pred, title, fig_path, csv_path,
                color='#1f77b4', label='Samples',
                extra_cols=None, xlabel='Experimental lgKs',
                ylabel='Predicted lgKs'):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, c=color, edgecolors='black',
               alpha=0.6, s=60, label=label)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.8, label='Ideal Line')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f'{title}\nR²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}',
        fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'>>> 图像已保存: {fig_path}')

    data_dict = {}
    if extra_cols:
        for k, v in extra_cols.items():
            data_dict[k] = list(v)
    data_dict['Experimental_lgKs'] = y_true
    data_dict['Predicted_lgKs']    = y_pred
    pd.DataFrame(data_dict).to_csv(csv_path, index=False, float_format='%.6f')
    print(f'>>> 数据已保存: {csv_path}')

    return {'R²': r2, 'MAE': mae, 'RMSE': rmse}

# ---------- 8-1. 留一法交叉验证可视化与数据保存 ----------
print('\n>>> 正在处理留一法交叉验证结果...')

loo_plot_metrics = plot_parity(
    y_train, y_loo_pred,
    title='Leave-One-Out Cross Validation',
    fig_path='LOO_CV_Parity_Plot.png',
    csv_path='留一法交叉验证结果.csv',
    color='#2ca02c',
    label='LOO Prediction'
)
print(f'  LOO 指标: R²={loo_plot_metrics["R²"]:.4f}, '
      f'MAE={loo_plot_metrics["MAE"]:.4f}, '
      f'RMSE={loo_plot_metrics["RMSE"]:.4f}')

# ---------- 8-2. 训练集 + 测试集外部验证可视化与数据保存 ----------
print('\n>>> 正在处理外部验证结果...')

y_train_exp  = np.asarray(y_train)
y_test_exp   = np.asarray(y_test)
y_tr_pred_np = np.asarray(y_tr_pred)
y_te_pred_np = np.asarray(y_te_pred)

# 合并用于画图边界
all_y_true = np.concatenate([y_train_exp, y_test_exp])
all_y_pred = np.concatenate([y_tr_pred_np, y_te_pred_np])
all_set    = ['Train'] * len(y_train_exp) + ['Test'] * len(y_test_exp)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_train_exp, y_tr_pred_np, c='blue', edgecolors='black',
           alpha=0.6, s=60, label='Training Set')
ax.scatter(y_test_exp, y_te_pred_np, c='red', edgecolors='black',
           alpha=0.6, s=60, marker='s', label='Test Set')
mn = min(all_y_true.min(), all_y_pred.min())
mx = max(all_y_true.max(), all_y_pred.max())
ax.plot([mn, mx], [mn, mx], 'k--', lw=2, label='Ideal Line')

# 分别计算 Train 和 Test 的指标
train_r2_ext   = r2_score(y_train_exp, y_tr_pred_np)
train_mae_ext  = mean_absolute_error(y_train_exp, y_tr_pred_np)
train_rmse_ext = np.sqrt(mean_squared_error(y_train_exp, y_tr_pred_np))

test_r2_ext    = r2_score(y_test_exp, y_te_pred_np)
test_mae_ext   = mean_absolute_error(y_test_exp, y_te_pred_np)
test_rmse_ext  = np.sqrt(mean_squared_error(y_test_exp, y_te_pred_np))

ax.set_xlabel('Experimental lgKs', fontsize=12)
ax.set_ylabel('Predicted lgKs', fontsize=12)
# 图表标题同时展示两部分指标
ax.set_title(
    f'External Validation (Train & Test)\n'
    f'Train: R²={train_r2_ext:.4f}, MAE={train_mae_ext:.4f}, RMSE={train_rmse_ext:.4f}\n'
    f'Test:  R²={test_r2_ext:.4f}, MAE={test_mae_ext:.4f}, RMSE={test_rmse_ext:.4f}',
    fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('External_Validation_Parity_Plot.png', dpi=300, bbox_inches='tight')
plt.close()
print('>>> 图像已保存: External_Validation_Parity_Plot.png')

external_df = pd.DataFrame({
    'Set':              all_set,
    'Experimental_lgKs': all_y_true,
    'Predicted_lgKs':   all_y_pred
})
external_df.to_csv('外部验证.csv', index=False, float_format='%.6f')
print('>>> 数据已保存: 外部验证.csv')

print(f'  Train {train_r2_ext:.6f} {train_mae_ext:.6f} {train_rmse_ext:.6f}')
print(f'  Test  {test_r2_ext:.6f} {test_mae_ext:.6f} {test_rmse_ext:.6f}')

# ---------- 8-3. 整体数据集评估 ----------
print('\n>>> 正在处理整体数据集评估...')

X_all_imp = pipe.transform(data[sel_feat])
y_all_pred = model.predict(X_all_imp)

overall_plot_metrics = plot_parity(
    y.values, y_all_pred,
    title='Overall Dataset (All Samples)',
    fig_path='Overall_Parity_Plot.png',
    csv_path='整体验证.csv',
    color='#9467bd',
    label='All Samples'
)
print(f'  整体数据集 (共 {len(y)} 个样本) 指标: '
      f'R²={overall_plot_metrics["R²"]:.4f}, '
      f'MAE={overall_plot_metrics["MAE"]:.4f}, '
      f'RMSE={overall_plot_metrics["RMSE"]:.4f}')

# ============================================================
# 9. Y 随机化验证（1000 次）
# ============================================================
print('\n>>> 开始 Y 随机化验证 (1000 次)')

N_PERM = 1000

X_tr_np = np.asarray(X_tr_imp)
y_tr_np = np.asarray(y_train)
N_TRAIN = len(y_tr_np)


def one_permutation(seed):
    """一次随机化：返回 (R²_train, Q²_LOO)"""
    rng = np.random.RandomState(seed)
    y_perm = y_tr_np[rng.permutation(N_TRAIN)]

    m = CatBoostRegressor(**BEST_PARAMS)
    m.fit(X_tr_np, y_perm)
    r2_train = r2_score(y_perm, m.predict(X_tr_np))

    y_loo_perm_pred = np.empty(N_TRAIN)
    for i in range(N_TRAIN):
        mask = np.ones(N_TRAIN, dtype=bool)
        mask[i] = False
        mm = CatBoostRegressor(**BEST_PARAMS)
        mm.fit(X_tr_np[mask], y_perm[mask])
        y_loo_perm_pred[i] = mm.predict(X_tr_np[[i]])[0]
    q2_loo = r2_score(y_perm, y_loo_perm_pred)

    return r2_train, q2_loo


results_perm = Parallel(n_jobs=-1, verbose=10)(
    delayed(one_permutation)(i) for i in range(N_PERM)
)

r2_perm_list = [r[0] for r in results_perm]
q2_perm_list = [r[1] for r in results_perm]

yrand_df = pd.DataFrame({
    'R²': r2_perm_list,
    'Q²': q2_perm_list,
})
yrand_df.to_csv('Y随机化验证.csv', index=False, float_format='%.6f')
print('>>> 数据已保存: Y随机化验证.csv')

avg_r2 = float(np.mean(r2_perm_list))
avg_q2 = float(np.mean(q2_perm_list))
print(f'  随机化平均 R²(Train) = {avg_r2:.4f}')
print(f'  随机化平均 Q²(LOO)   = {avg_q2:.4f}')

orig_r2 = r2_tr
orig_q2 = loo_metrics['R²']

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(r2_perm_list, q2_perm_list,
           c='gray', alpha=0.5, s=30, label='Randomized models')
ax.scatter([orig_r2], [orig_q2],
           c='red', s=200, marker='*', edgecolors='black',
           label='Original model', zorder=5)

ax.set_xlabel('R² (training)', fontsize=12)
ax.set_ylabel('Q² (LOO-CV)', fontsize=12)
ax.set_title(
    f'Y-Randomization ({N_PERM} permutations)\n'
    f'Avg R²={avg_r2:.4f}, Avg Q²={avg_q2:.4f}',
    fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()

x_min = min(min(r2_perm_list), orig_r2) - 0.05
x_max = max(max(r2_perm_list), orig_r2) + 0.05
y_min = min(min(q2_perm_list), orig_q2) - 0.05
y_max = max(max(q2_perm_list), orig_q2) + 0.05
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('Y_Randomization_Plot.png', dpi=300, bbox_inches='tight')
plt.close()
print('>>> 图像已保存: Y_Randomization_Plot.png')

print('\n>>> 所有任务完成！输出文件：')
print('    5fold_cv_folds.csv       → 五折每折明细（整个数据集）')
print('    5fold_cv_summary.csv     → 五折均值 ± 标准差（整个数据集）')
print('    LOO_CV_Parity_Plot.png                + 留一法交叉验证结果.csv')
print('    External_Validation_Parity_Plot.png   + 外部验证.csv')
print('    Overall_Parity_Plot.png               + 整体验证.csv')
print('    Y_Randomization_Plot.png              + Y随机化验证.csv')
print('    combined_metrics.csv（Train/Test/5-Fold/LOO，R²/MAE/RMSE）')
print('    best_params.json / 0.81boost.cbm')