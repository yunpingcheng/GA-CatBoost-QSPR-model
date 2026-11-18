import numpy as np
import pandas as pd
import rdkit as rk
from rdkit import Chem 
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import os

# 读取CSV文件
df = pd.read_csv('SMILES.csv')
# 2. 基本清洗：去掉 SMILES 为 NaN、空字符串、或纯白的行
df = df.dropna(subset=[df.columns[0], df.columns[1]])          
df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()        
df.iloc[:, 1] = df.iloc[:, 1].astype(str).str.strip()
df = df[df.iloc[:, 0].str.lower() != 'nan']                  
df = df[df.iloc[:, 1].str.lower() != 'nan']
df = df[df.iloc[:, 0] != '']
df = df[df.iloc[:, 1] != '']
print(f'清洗后有效行数：{len(df)}')

# 获取所有可用的描述符列表
all_des_list = [x[0] for x in Descriptors._descList]
print(f"Number of all available descriptors: {len(all_des_list)}")

# 创建描述符计算器
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(all_des_list)

# 存储结果的列表
results = []

# 定义函数：安全计算分子描述符
def safe_calc_descriptors(smiles, descriptor_names, calculator, mol_type="molecule"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            descriptors = calculator.CalcDescriptors(mol)
            return pd.Series(descriptors, index=descriptor_names), True
        except Exception as e:
            print(f"警告: 无法计算{mol_type}分子 {smiles} 的描述符: {str(e)}")
            # 返回NaN值
            return pd.Series([np.nan] * len(descriptor_names), index=descriptor_names), False
    else:
        print(f"警告: 无法解析{mol_type} SMILES: {smiles}")
        # 返回NaN值
        return pd.Series([np.nan] * len(descriptor_names), index=descriptor_names), False
# 遍历每一行数据
for index, row in df.iterrows():
    host_smiles = row.iloc[0]  # 第一列：主体分子SMILES
    guest_smiles = row.iloc[1]  # 第二列：客体分子SMILES
    lgks_value = row.iloc[2]  # 第三列：lgKs值
    
    # 安全计算主体分子描述符
    host_series, host_success = safe_calc_descriptors(host_smiles, all_des_list, calculator, "主体")
    
    # 安全计算客体分子描述符
    guest_series, guest_success = safe_calc_descriptors(guest_smiles, all_des_list, calculator, "客体")
    
    # 计算组合描述符：sqrt(host_descriptor * guest_descriptor)
    combined_descriptors = {}
    for desc_name in all_des_list:
        host_val = host_series[desc_name]
        guest_val = guest_series[desc_name]
        
        # 只有当两个值都不是NaN时才计算组合描述符
        if not pd.isna(host_val) and not pd.isna(guest_val):
            # 计算乘积
            product = host_val * guest_val
            
            # 处理可能的负值（对于平方根计算）
            if product >= 0:
                combined_val = np.sqrt(product)
            else:
                # 如果乘积为负，使用绝对值然后标记为负
                combined_val = -np.sqrt(abs(product))
        else:
            # 如果任一值为NaN，组合描述符也为NaN
            combined_val = np.nan
        
        combined_descriptors[f'com_{desc_name}'] = combined_val
    
    # 创建结果字典
    result_row = {
        'host_smiles': host_smiles,
        'guest_smiles': guest_smiles,
        'lgKs': lgks_value
    }
     # 添加主体描述符（前缀为host_）
    for desc_name, value in host_series.items():
        result_row[f'host_{desc_name}'] = value
    
    # 添加客体描述符（前缀为guest_）
    for desc_name, value in guest_series.items():
        result_row[f'guest_{desc_name}'] = value
    
    # 添加组合描述符（前缀为com_）
    result_row.update(combined_descriptors)
    
    results.append(result_row)
    
    # 在终端输出当前处理进度
    status = "成功" if host_success and guest_success else "部分成功" if host_success or guest_success else "失败"
    print(f"处理行 {index+1}/{len(df)}: 主体={host_smiles}, 客体={guest_smiles}, 状态={status}")

# 创建最终的数据框
results_df = pd.DataFrame(results)
# 输出结果信息
print(f"\n处理完成!")
print(f"总处理行数: {len(results)}")
print(f"每个分子的描述符数量: {len(all_des_list)}")
print(f"输出总列数: {len(results_df.columns)}")
print(f"列分类:")
print(f"  - 基本信息: 3 列 (host_smiles, guest_smiles, lgKs)")
print(f"  - 主体描述符: {len(all_des_list)} 列")
print(f"  - 客体描述符: {len(all_des_list)} 列")
print(f"  - 组合描述符: {len(all_des_list)} 列")

# 1. 按前缀把列分成三组
host_cols   = [c for c in results_df.columns if c.startswith('host_')]
guest_cols  = [c for c in results_df.columns if c.startswith('guest_')]
com_cols    = [c for c in results_df.columns if c.startswith('com_')]

# 2. 统计每组 NaN 的数量
host_nan = results_df[host_cols].isna().sum().sum()
guest_nan = results_df[guest_cols].isna().sum().sum()
com_nan = results_df[com_cols].isna().sum().sum()

print('NaN 统计')
print(f'主体描述符 NaN 总数: {host_nan}')
print(f'客体描述符 NaN 总数: {guest_nan}')
print(f'组合描述符 NaN 总数: {com_nan}')

def is_row_successful(row, all_des_list):
    """检查一行数据是否成功计算了所有描述符"""
    # 检查第一个主体描述符和第一个客体描述符是否为NaN
    host_first_desc = f'host_{all_des_list[0]}'
    guest_first_desc = f'guest_{all_des_list[0]}'
    
    return not pd.isna(row[host_first_desc]) and not pd.isna(row[guest_first_desc])

success_count = sum(1 for row in results if is_row_successful(row, all_des_list))
print(f"\n成功计算完整描述符的行数: {success_count}/{len(results)}")

# 显示前几行数据的基本信息
print(f"\n结果的前几行:")
print(results_df.head())

# 保存结果到CSV文件
save_path = 'calculated descriptors.csv'
if os.path.dirname(save_path): 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
results_df.to_csv(save_path, index=False)
print(f"\n结果已保存到 '{save_path}'")