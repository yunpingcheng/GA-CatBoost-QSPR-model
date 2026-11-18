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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

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


# ============================================================
# 7. SHAP特征重要性分析
# ============================================================
print('\n>>> 开始SHAP特征重要性分析...')

try:
    import shap
except ImportError:
    print("正在安装shap库...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'shap'])
    import shap

# 创建SHAP解释器
explainer = shap.TreeExplainer(model)

# 计算SHAP值
print("正在计算SHAP值...")
shap_values = explainer.shap_values(X_tr_imp)

# 获取特征名称
feature_names = sel_feat

# ---------- 7-1. SHAP Summary Plot (蜂群图) ----------
print("\n>>> 生成SHAP Summary Plot...")

plt.clf()
fig, ax = plt.subplots(figsize=(14, 10))

shap.summary_plot(
    shap_values, 
    X_tr_imp, 
    feature_names=feature_names,
    show=False,
    max_display=15,
    plot_size=None
)

ax = plt.gca()
ax.set_xlim(-0.45, 0.85)
ax.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
ax.set_xlabel('SHAP value (impact on model output)', fontsize=16)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=14)

for label in ax.get_yticklabels():
    label.set_fontsize(16)
    label.set_fontname('Arial')

cbar_ax = plt.gcf().axes[-1]
cbar_ax.tick_params(labelsize=16)
cbar_ax.set_ylabel('Feature value', fontsize=16)

plt.title('SHAP Feature Importance (Top 15)', fontsize=16, pad=15)
plt.tight_layout()
plt.savefig('SHAP_Summary_Plot.png', dpi=600, bbox_inches='tight')
print('>>> 图像已保存: SHAP_Summary_Plot.png')
plt.close()

# ---------- 7-2. SHAP Bar Plot (特征重要性排序) ----------
print("\n>>> 生成SHAP Bar Plot...")

plt.clf()
fig, ax = plt.subplots(figsize=(10, 12))

shap.summary_plot(
    shap_values, 
    X_tr_imp, 
    feature_names=feature_names,
    plot_type="bar",
    show=False,
    max_display=15,
)

ax = plt.gca()
ax.set_xlim(0, 0.25)
ax.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20])
ax.set_xlabel('mean(|SHAP value|) (average impact on model output magnitude)', fontsize=12)
ax.tick_params(axis='x', labelsize=11)
ax.tick_params(axis='y', labelsize=11)

for label in ax.get_yticklabels():
    label.set_fontsize(11)
    label.set_fontname('Arial')

plt.title('Mean |SHAP value| (average impact on model output magnitude)', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('SHAP_Bar_Plot.png', dpi=600, bbox_inches='tight')
print('>>> 图像已保存: SHAP_Bar_Plot.png')
plt.close()


# ============================================================
# 8. 适用域分析（Applicability Domain, Williams Plot）
# ============================================================
print('\n>>> 开始适用域分析 (Applicability Domain)...')

# 8-1 基于训练集计算 (X^T X)^{-1}
X_mean = X_tr_imp.mean(axis=0)
X_train_c = X_tr_imp - X_mean
XtX_inv = np.linalg.pinv(X_train_c.T @ X_train_c)  

def calc_leverage(X):
    Xc = X - X_mean
    return np.einsum('ij,jk,ik->i', Xc, XtX_inv, Xc)

h_train = calc_leverage(X_tr_imp)
h_test  = calc_leverage(X_te_imp)

n_train = X_tr_imp.shape[0]   # 训练集样本数 n
m_desc  = X_tr_imp.shape[1]   # 描述符个数 m
h_star  = 3 * (m_desc + 1) / n_train   # 临界杠杆阈值 h*

# 8-2 标准化残差
res_train = y_train.values - y_tr_pred
res_test  = y_test.values  - y_te_pred

dof = n_train - m_desc - 1
s_res = np.sqrt(np.sum(res_train ** 2) / dof) if dof > 0 else np.std(res_train, ddof=1)
if s_res == 0:
    s_res = 1e-12

std_res_train = res_train / s_res
std_res_test  = res_test  / s_res

# 8-3 适用域判定与离群类型标注
def classify(h, sr):
    out_resp = np.abs(sr) > 3        # 响应离群
    out_lev  = h > h_star            # 杠杆离群
    types = np.where(out_resp & out_lev, 'Both',
            np.where(out_resp, 'Response outlier',
            np.where(out_lev, 'Leverage outlier', 'Within AD')))
    return np.where(types == 'Within AD', 'Within AD', 'Outside AD'), types

ad_flag_train, type_train = classify(h_train, std_res_train)
ad_flag_test,  type_test  = classify(h_test,  std_res_test)

print(f'\n>>> 适用域参数: n(train)={n_train}, m(descriptors)={m_desc}, h*={h_star:.4f}')
print(f'>>> 训练集: 域外 {np.sum(ad_flag_train=="Outside AD")}/{n_train} 个样本')
print(f'>>> 测试集: 域外 {np.sum(ad_flag_test=="Outside AD")}/{len(ad_flag_test)} 个样本')

# 8-4 分别统计标准化残差超限(-3,3)与杠杆值超限(h > h*)的数据点个数
n_train_std_out = int((np.abs(std_res_train) > 3).sum())
n_test_std_out  = int((np.abs(std_res_test) > 3).sum())
n_train_lev_out = int((h_train > h_star).sum())
n_test_lev_out  = int((h_test > h_star).sum())

print('\n===== AD 超限数据点统计 =====')
print(f'标准化残差超过 (-3, 3) 范围:      训练集 {n_train_std_out}/{len(std_res_train)} 个,  测试集 {n_test_std_out}/{len(std_res_test)} 个')
print(f'Leverage h 超过 h* ({h_star:.4f}):  训练集 {n_train_lev_out}/{len(h_train)} 个,  测试集 {n_test_lev_out}/{len(h_test)} 个')

# 8-5 Williams Plot 可视化
fig, ax = plt.subplots(figsize=(9, 7))

colors = {'Within AD': 'steelblue', 'Response outlier': 'red',
          'Leverage outlier': 'orange', 'Both': 'purple'}
markers = {'Train': 'o', 'Test': 's'}

plotted = set()
for set_name, type_, h, sr in [('Train', type_train, h_train, std_res_train),
                               ('Test',  type_test,  h_test,  std_res_test)]:
    for t in colors:
        mask = type_ == t
        if mask.sum() == 0:
            continue
        label = f'{set_name} - {t}' if (set_name, t) not in plotted else None
        plotted.add((set_name, t))
        ax.scatter(h[mask], sr[mask], s=55, alpha=0.75, edgecolors='black',
                   color=colors[t], marker=markers[set_name], label=label)

x_max = max(h_train.max(), h_test.max(), h_star) * 1.15
ax.axvline(h_star, color='red', ls='--', lw=2, label=f'h* = {h_star:.2f}')
ax.axhline(3,  color='green', ls='--', lw=1.5)
ax.axhline(-3, color='green', ls='--', lw=1.5, label='±3 Std Residual')
ax.axhline(0,  color='gray',  ls='-',  lw=0.8)

ax.set_xlim(0, x_max)
ax.set_ylim(min(-4, std_res_train.min(), std_res_test.min()) - 0.5,
            max(4,  std_res_train.max(), std_res_test.max()) + 0.5)
ax.set_xlabel('Leverage ($h_i$)', fontsize=12)
ax.set_ylabel('Standardized Residuals', fontsize=12)
ax.set_title('Williams Plot — Applicability Domain', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.savefig('Williams_Plot_AD.png', dpi=300, bbox_inches='tight')
print('>>> 图像已保存: Williams_Plot_AD.png')
plt.close()

# 8-6 保存适用域分析数据
ad_df = pd.DataFrame({
    'Set': ['Train'] * len(h_train) + ['Test'] * len(h_test),
    'Sample_ID': np.concatenate([X_train.index.values, X_test.index.values]),
    'Experimental_lgKs': np.concatenate([y_train.values, y_test.values]),
    'Predicted_lgKs': np.concatenate([y_tr_pred, y_te_pred]),
    'Residual': np.concatenate([res_train, res_test]),
    'Std_Residual': np.concatenate([std_res_train, std_res_test]),
    'Leverage': np.concatenate([h_train, h_test]),
    'h_star': h_star,
    'AD_Flag': np.concatenate([ad_flag_train, ad_flag_test]),
    'Outlier_Type': np.concatenate([type_train, type_test])
})
ad_df.to_csv('applicability_domain.csv', index=False)
print('>>> 数据已保存: applicability_domain.csv')

print('\n>>> 所有分析任务完成！')