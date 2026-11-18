import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, ParameterGrid
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import spearmanr
from catboost import CatBoostRegressor


# ============================================================
# 第一部分：数据加载和预处理
# ============================================================
csv_path = 'calculated descriptors.csv'
data = pd.read_csv(csv_path)

exclude_cols = {'host_smiles', 'guest_smiles', 'lgKs'}
num_cols = [c for c in data.columns
            if c.startswith(('host_', 'guest_', 'com_'))
            and c not in exclude_cols]

if not num_cols:
    raise ValueError('未找到任何描述符列，请检查列名前缀！')

data = data[num_cols + ['lgKs']].dropna(axis=1, how='all')
num_cols = [c for c in data.columns if c != 'lgKs']
print(f'可用数值描述符列数: {len(num_cols)}')

X_df = data[num_cols].copy()
y = data['lgKs'].astype(float)

if X_df.dtypes.eq('object').any():
    obj_cols = X_df.columns[X_df.dtypes.eq('object')].tolist()
    raise TypeError(f'以下列仍为字符串，已剔除 → {obj_cols}')

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42)

X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan)
X_test_df  = X_test_df.replace([np.inf, -np.inf], np.nan)

impute_inf = SimpleImputer(strategy='median')
X_train_df = pd.DataFrame(
    impute_inf.fit_transform(X_train_df),
    columns=X_train_df.columns, index=X_train_df.index)
X_test_df = pd.DataFrame(
    impute_inf.transform(X_test_df),
    columns=X_test_df.columns, index=X_test_df.index)

vt = VarianceThreshold(threshold=0.05)
vt.fit(X_train_df)
X_train_vt = vt.transform(X_train_df)
X_test_vt  = vt.transform(X_test_df)
vt_cols = X_train_df.columns[vt.get_support()]
print(f'VarianceThreshold 后特征数: {len(vt_cols)}')


def spearman_cluster(X, names, rho_max=0.95):
    mask = np.std(X, axis=0) > 0
    X = X[:, mask]
    names = names[mask]
    if X.shape[1] < 2:
        return names
    corr, _ = spearmanr(X, axis=0)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 0)
    keep_mask = np.ones(corr.shape[0], dtype=bool)
    for i in range(corr.shape[0]):
        if not keep_mask[i]:
            continue
        dupes = np.where(corr[i, :] > rho_max)[0]
        keep_mask[dupes] = False
        keep_mask[i] = True
    return names[keep_mask]


keep_names = spearman_cluster(X_train_vt, vt_cols, 0.95)
print(f'Spearman 去冗余后特征数: {len(keep_names)}')

X_train_clu = pd.DataFrame(X_train_vt, columns=vt_cols)[keep_names]
X_test_clu  = pd.DataFrame(X_test_vt,  columns=vt_cols)[keep_names]

pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])
X_train_imp = pipe.fit_transform(X_train_clu)
X_test_imp  = pipe.transform(X_test_clu)

print(f'总样本 {len(y)}  训练 {len(y_train)}  测试 {len(y_test)}  特征 {X_train_imp.shape[1]}')


# ============================================================
# 第二部分：从 selected_features.csv 读取 GA 选中的特征描述符
# ============================================================
SEL_FEAT_PATH = 'selected_features.csv'
if not os.path.exists(SEL_FEAT_PATH):
    raise FileNotFoundError(f'未找到 {SEL_FEAT_PATH}，请先运行 GA 阶段')

sel_df = pd.read_csv(SEL_FEAT_PATH)
print(f'\n[读入] {SEL_FEAT_PATH}  shape = {sel_df.shape}')
print(f'  列: {list(sel_df.columns)}')

name_col = None
for c in sel_df.columns:
    if str(c).strip().lower() == 'name':
        name_col = c
        break
if name_col is None:
    name_col = sel_df.columns[-1]
    print(f'  [提示] 未找到 Name 列，默认使用最后一列 "{name_col}"')

selected_name = sel_df[name_col].astype(str).str.strip().tolist()

seen = set()
selected_name_unique = []
for x in selected_name:
    if x and x != 'nan' and x not in seen:
        seen.add(x)
        selected_name_unique.append(x)
selected_name = selected_name_unique
print(f'  解析得到 {len(selected_name)} 个唯一特征名')

missing = [c for c in selected_name if c not in X_train_clu.columns]
if missing:
    print(f'\n[警告] 以下 {len(missing)} 个特征不在 X_train_clu 中：')
    for m in missing[:20]:
        print(f'    {m}')
    if len(missing) > 20:
        print(f'    ... 共 {len(missing)} 个')

existing = [c for c in selected_name if c in X_train_clu.columns]
if not existing:
    raise RuntimeError('没有任何选中特征匹配 X_train_clu.columns，请核对列名')

print(f'\n[使用] 实际可用特征 {len(existing)} 个：')
for f in existing:
    print('   ', f)


# ============================================================
# 第三部分：按 Name 切出选中特征 + 标准化
# ============================================================
X_tr_sel_raw = X_train_clu[existing].reset_index(drop=True)
X_te_sel_raw = X_test_clu[existing].reset_index(drop=True)

pipe_sel = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])
X_tr_sel = pd.DataFrame(pipe_sel.fit_transform(X_tr_sel_raw),
                        columns=existing).reset_index(drop=True)
X_te_sel = pd.DataFrame(pipe_sel.transform(X_te_sel_raw),
                        columns=existing).reset_index(drop=True)

y_tr = y_train.reset_index(drop=True)
y_te = y_test.reset_index(drop=True)


# ============================================================
# 第四部分：网格搜索
# ============================================================
param_grid = {
    'depth':                       [4, 5, 6],
    'learning_rate':               [0.08, 0.10, 0.12],
    'l2_leaf_reg':                 [10, 12, 15],
    'border_count':                [32, 56, 64],
    'iterations':                  [300, 350, 400, 450, 500],
    'subsample':                   [0.6, 0.7, 0.8],
    'random_strength':             [5, 10, 15],
    'model_size_reg':              [0.5, 1, 2],
    'leaf_estimation_iterations':  [1],
    'bootstrap_type':              ['Bernoulli'],
    'sampling_frequency':          ['PerTree'],
    'loss_function':               ['RMSE'],
    'task_type':                   ['CPU'],
}


def _to_native(obj):
    """递归把 numpy 类型转换为原生 Python 类型，便于 json.dump"""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


records = []
print('\n>>> 开始网格搜索……')
for i, p in enumerate(ParameterGrid(param_grid), 1):
    cb = CatBoostRegressor(**p, verbose=False, random_seed=42, thread_count=-1)
    cb.fit(X_tr_sel, y_tr)
    r2_tr = float(r2_score(y_tr, cb.predict(X_tr_sel)))
    r2_te = float(r2_score(y_te, cb.predict(X_te_sel)))
    records.append({
        'params':   p,
        'train_r2': r2_tr,
        'test_r2':  r2_te,
        'delta_r2': r2_tr - r2_te,
    })
    print(f'[{i:5d}] 训练 R² = {r2_tr:.4f} | 测试 R² = {r2_te:.4f} '
          f'| Δ = {r2_tr - r2_te:+.4f}')

candidates = [r for r in records if r['train_r2'] > r['test_r2']]
if candidates:
    best_rec   = max(candidates, key=lambda r: r['test_r2'])
    sel_reason = 'train_r2 > test_r2 且 test_r2 最大'
else:
    print('\n[警告] 所有组合都不满足 train_r2 > test_r2，退化为「test_r2 最大」')
    best_rec   = max(records, key=lambda r: r['test_r2'])
    sel_reason = '（退化）test_r2 最大'

best_params = best_rec['params']
best_r2_tr  = best_rec['train_r2']
best_r2_te  = best_rec['test_r2']
best_delta  = best_rec['delta_r2']

print(f'\n[最终选择] 准则: {sel_reason}')
print(f'   Train R² = {best_r2_tr:.4f}')
print(f'   Test  R² = {best_r2_te:.4f}')
print(f'   ΔR²      = {best_delta:+.4f}')
print(f'   最优参数: {best_params}')


# ============================================================
# 第五部分：保存最优超参数
# ============================================================
BEST_PARAMS = _to_native({
    **best_params,
    'random_seed':  42,
    'thread_count': -1,
    'verbose':      False,
})

params_json = 'best_params.json'
with open(params_json, 'w', encoding='utf-8') as f:
    json.dump(BEST_PARAMS, f, ensure_ascii=False, indent=4, default=str)
print(f'\n>>> 最佳参数已保存到 {params_json}')


# ============================================================
# 第六部分：用最优参数训练最终模型 + 保存 .cbm
# ============================================================
final_model = CatBoostRegressor(**best_params, verbose=False,
                                random_seed=42, thread_count=-1)
final_model.fit(X_tr_sel, y_tr)
final_model.save_model('0.81boost.cbm')
print('\n>>> 最终模型已保存到 0.81boost.cbm')

y_tr_pred = final_model.predict(X_tr_sel)
y_te_pred = final_model.predict(X_te_sel)


# ============================================================
# 第七部分：计算并输出最终指标 (Train & Test)
# ============================================================
def calc_metrics(y_true, y_pred, tag):
    return {
        'Method': tag,
        'N':      len(y_true),
        'R²':     r2_score(y_true, y_pred),
        'MAE':    mean_absolute_error(y_true, y_pred),
        'RMSE':   np.sqrt(mean_squared_error(y_true, y_pred)),
    }

tr_metrics = calc_metrics(y_tr, y_tr_pred, 'External-Train')
te_metrics = calc_metrics(y_te, y_te_pred, 'External-Test')

summary_df = pd.DataFrame([tr_metrics, te_metrics])
summary_df.to_csv('combined_metrics.csv', index=False, float_format='%.4f')

print('\n' + '=' * 60)
print('模型最终预测指标：')
print('=' * 60)
print(summary_df.to_string(index=False))
print('=' * 60)
print('\n>>> 最终指标已保存到 combined_metrics.csv')


# ============================================================
# 第八部分：绘图（训练集与测试集 Parity 图）
# ============================================================
plt.figure(figsize=(8, 8))

# 绘制训练集
plt.scatter(y_tr, y_tr_pred, c='blue', edgecolors='black', alpha=0.6, s=60, label='Training Set')
# 绘制测试集
plt.scatter(y_te, y_te_pred, c='red', edgecolors='black', alpha=0.6, s=60, marker='s', label='Test Set')

# 绘制对角线
all_y_true = np.concatenate([y_tr, y_te])
all_y_pred = np.concatenate([y_tr_pred, y_te_pred])
mn = min(all_y_true.min(), all_y_pred.min())
mx = max(all_y_true.max(), all_y_pred.max())
plt.plot([mn, mx], [mn, mx], 'k--', lw=2, label='Ideal Line')

plt.xlabel('Experimental lgKs', fontsize=12)
plt.ylabel('Predicted lgKs', fontsize=12)
plt.title(
    f'External Validation (Train & Test)\n'
    f'Train: R²={tr_metrics["R²"]:.3f}, MAE={tr_metrics["MAE"]:.3f}, RMSE={tr_metrics["RMSE"]:.3f}\n'
    f'Test:  R²={te_metrics["R²"]:.3f}, MAE={te_metrics["MAE"]:.3f}, RMSE={te_metrics["RMSE"]:.3f}',
    fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.gca().set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('Train_Test_Parity_Plot.png', dpi=300, bbox_inches='tight')
plt.close()
print('\n>>> 综合散点图已保存到 Train_Test_Parity_Plot.png')


# ============================================================
# 第九部分：保存预测结果
# ============================================================
os.makedirs('pred_results', exist_ok=True)
pd.DataFrame({'y_true_train': y_tr.values, 'y_pred_train': y_tr_pred}) \
    .to_csv('pred_results/train_pred.csv', index=False, float_format='%.6f')
pd.DataFrame({'y_true_test': y_te.values, 'y_pred_test': y_te_pred}) \
    .to_csv('pred_results/test_pred.csv', index=False, float_format='%.6f')
print('\n>>> 预测结果已保存至 pred_results/ 目录')