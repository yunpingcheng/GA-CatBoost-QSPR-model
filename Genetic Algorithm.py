import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import spearmanr
from catboost import CatBoostRegressor

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

# 5. 加载CatBoost模型超参数
BEST_PARAM_JSON = 'best_params.json' 

def _load_best_params(path):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            if 'params' in obj and isinstance(obj['params'], dict):
                return obj['params'], path
            if any(k in obj for k in ('depth', 'learning_rate', 'iterations')):
                return obj, path
        except Exception as e:
            print(f'[警告] 读取 {path} 失败: {e}')
    raise RuntimeError(f'未找到超参数文件: {path}')

GA_PARAMS, GA_PARAM_SOURCE = _load_best_params(BEST_PARAM_JSON)

GA_PARAMS.setdefault('bootstrap_type',     'Bernoulli')
GA_PARAMS.setdefault('sampling_frequency', 'PerTree')
GA_PARAMS.setdefault('loss_function',      'RMSE')
GA_PARAMS.setdefault('task_type',          'CPU')
GA_PARAMS.setdefault('verbose',            False)
GA_PARAMS.setdefault('random_seed',        42)
GA_PARAMS.setdefault('allow_writing_files', False)

print('[参数来源]', GA_PARAM_SOURCE)
print('[固定超参数]')
for k, v in GA_PARAMS.items():
    print(f'    {k}: {v}')

# 6. 遗传算法参数
pc = 0.02          # 变异概率
t = 60             # 代数
n = 30             # 种群规模
min_feat, max_feat, step_feat = 5, 25, 5

# 7. 适应度函数（5 折 KFold R2）
def fitness_function(ind, X_df_raw, y, pipe):
    sel = np.where(ind == 1)[0]
    if len(sel) == 0 or len(y) < 5:
        return 0.0
    X_scaled = pipe.fit_transform(X_df_raw)[:, sel]
    y_arr = np.asarray(y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_i, va_i in kf.split(X_scaled):
        params = {**GA_PARAMS, 'thread_count': 1}
        model = CatBoostRegressor(**params)
        model.fit(X_scaled[tr_i], y_arr[tr_i])
        scores.append(r2_score(y_arr[va_i], model.predict(X_scaled[va_i])))

    return float(np.mean(scores))

# 8. 遗传算子
def selection(pop, fit):
    fit = np.maximum(fit, 1e-10)
    cum = np.cumsum(fit / fit.sum())
    new_pop = np.empty_like(pop)
    for i in range(n):
        idx = np.searchsorted(cum, np.random.rand())
        new_pop[i] = pop[idx]
    return new_pop

def crossover(pop, d):
    np.random.shuffle(pop)
    father = pop[:n//2]
    mother = pop[n//2:]
    for i in range(len(father)):
        cp = np.random.randint(1, pop.shape[1]-1)
        child1 = np.r_[father[i][:cp], mother[i][cp:]]
        child2 = np.r_[mother[i][:cp], father[i][cp:]]
        father[i] = adjust(child1, d)
        mother[i] = adjust(child2, d)
    return np.vstack([father, mother])

def mutation(pop, pc, d):
    for i in range(n):
        if np.random.rand() < pc:
            ind = pop[i].copy()
            ones = np.where(ind == 1)[0]
            zeros = np.where(ind == 0)[0]
            if len(ones) and len(zeros):
                ind[np.random.choice(zeros)] = 1
                ind[np.random.choice(ones)] = 0
            pop[i] = ind
    return pop

def adjust(ind, d):
    curr = int(ind.sum())
    if curr == d:
        return ind
    if curr < d:
        zeros = np.where(ind == 0)[0]
        need = min(d - curr, len(zeros))
        ind[np.random.choice(zeros, need, replace=False)] = 1
    else:
        ones = np.where(ind == 1)[0]
        need = min(curr - d, len(ones))
        ind[np.random.choice(ones, need, replace=False)] = 0
    return ind

# 9. GA 主流程
def GA(X_raw_df, y, d_target, pipe):
    p = X_raw_df.shape[1]
    pop = np.zeros((n, p), int)
    for i in range(n):
        ind = np.zeros(p, int)
        ind[np.random.choice(p, d_target, replace=False)] = 1
        pop[i] = ind

    best_hist = []                 
    best_so_far = -np.inf          
    best_ind_so_far = None         

    for gen in range(t):
        fit = np.array(Parallel(n_jobs=-1)(
            delayed(fitness_function)(ind, X_raw_df, y, pipe) for ind in pop))
        gen_best = float(fit.max())

        if gen_best > best_so_far + 1e-4:
            best_so_far = gen_best
            best_ind_so_far = pop[np.argmax(fit)].copy()

        best_hist.append(best_so_far)

        pop = selection(pop, fit)
        pop = crossover(pop, d_target)
        pop = mutation(pop, pc, d_target)
        if (gen+1) % 10 == 0:
            print(f'  Gen {gen+1}/{t}  gen_best={gen_best:.4f}  '
                  f'global={best_so_far:.4f}')

    return best_ind_so_far, best_so_far, best_hist

# 10. 搜索最佳特征数
def find_best_d(X_raw_df, y, pipe):
    cand = list(range(min_feat, max_feat+1, step_feat))
    perf = []
    best_ind_global = None         
    best_fit_global = -np.inf     

    for d in cand:
        print(f'\n>>> 评估特征数 = {d}')
        best_ind, best_fit, _ = GA(X_raw_df, y, d, pipe)
        perf.append(best_fit)
        print(f'    最佳适应度 = {best_fit:.4f}')

        if best_fit > best_fit_global:
            best_fit_global = best_fit
            best_ind_global = best_ind

    best_d = cand[np.argmax(perf)]
    plt.figure(figsize=(6,4))
    plt.plot(cand, perf, 'bo-')
    plt.axvline(best_d, color='r', ls='--')
    plt.xlabel('特征数'); plt.ylabel('适应度(5-Fold R²)')
    plt.title('特征数 vs 适应度')
    plt.savefig('feature_d_curve.png', dpi=300)
    plt.show()
    return best_d, best_ind_global, best_fit_global

# 11. 运行
if __name__ == '__main__':
    print('\n===== 1. 搜索最佳特征数（同时拿到最优个体） =====')
    best_d, best_ind, best_fit = find_best_d(X_train_clu, y_train, pipe)
    print(f'\n最佳特征数 = {best_d}')
    print(f'最佳适应度 = {best_fit:.4f}')

    print('\n===== 2. 直接用 find_best_d 的最优个体 =====')
    selected_idx = np.where(best_ind == 1)[0]
    selected_name = X_train_clu.columns[selected_idx]
    print(f'选中特征 {len(selected_name)} 个：')
    for name in selected_name:
        print('  ', name)
    pd.DataFrame({'Index': selected_idx, 'Name': selected_name}) \
        .to_csv('selected_features.csv', index=False)