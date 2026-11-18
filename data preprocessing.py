import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import spearmanr

# 1. 读数据
csv_path = 'calculated descriptors.csv'
data = pd.read_csv(csv_path)
# 一次性剔除 SMILES 并保留数值描述符
exclude_cols = {'host_smiles', 'guest_smiles', 'lgKs'}
num_cols = [c for c in data.columns
            if c.startswith(('host_', 'guest_', 'com_'))
            and c not in exclude_cols]

if not num_cols:
    raise ValueError('未找到任何描述符列，请检查列名前缀！')

# 去掉全 NaN 列
data = data[num_cols + ['lgKs']].dropna(axis=1, how='all')
num_cols = [c for c in data.columns if c != 'lgKs']  
print(f'可用数值描述符列数: {len(num_cols)}')

# 2. X / y 分离
X_df = data[num_cols].copy()
y = data['lgKs'].astype(float)

# 确保无字符串
if X_df.dtypes.eq('object').any():
    obj_cols = X_df.columns[X_df.dtypes.eq('object')].tolist()
    raise TypeError(f'以下列仍为字符串，已剔除 → {obj_cols}')

# 3. 划分训练/测试
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42)

# ========== 低方差过滤 ==========
X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan)  
X_test_df  = X_test_df.replace([np.inf, -np.inf], np.nan)

impute_inf = SimpleImputer(strategy='median')
X_train_df = pd.DataFrame(
    impute_inf.fit_transform(X_train_df),
    columns=X_train_df.columns,
    index=X_train_df.index
)
X_test_df = pd.DataFrame(
    impute_inf.transform(X_test_df),
    columns=X_test_df.columns,
    index=X_test_df.index
)

vt = VarianceThreshold(threshold=0.05)
vt.fit(X_train_df)                                   
X_train_vt = vt.transform(X_train_df)                  
X_test_vt  = vt.transform(X_test_df)                   
vt_cols = X_train_df.columns[vt.get_support()]        
print(f'VarianceThreshold 后特征数: {len(vt_cols)}')

# ========== Spearman 冗余聚类 ==========             
def spearman_cluster(X, names, rho_max=0.95):
    mask = np.std(X, axis=0) > 0
    X = X[:, mask]
    names = names[mask]
    if X.shape[1] < 2:
        return names          
    """返回要保留的列名列表"""
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

# 按最终列名重新切片
X_train_clu = pd.DataFrame(X_train_vt, columns=vt_cols)[keep_names]
X_test_clu  = pd.DataFrame(X_test_vt,  columns=vt_cols)[keep_names]
#4. 标准化
pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])
X_train_imp = pipe.fit_transform(X_train_clu)   
X_test_imp  = pipe.transform(X_test_clu)       

print(f'总样本 {len(y)}  训练 {len(y_train)}  测试 {len(y_test)}  特征 {X_train_imp.shape[1]}')