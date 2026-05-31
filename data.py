# -*- coding: utf-8 -*-
"""
第二章：数据预处理与特征构造
功能：读取原始Excel，清洗、构造特征、标准化、划分数据集
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==================== 1. 读取原始数据 ====================
df = pd.read_excel(r"C:\数据\远程数据格式-100435参考扭矩2518Nm.xlsx")
print(f"原始数据行数: {len(df)}，目标列非零: {(df['nox瞬时排放强度'] != 0).sum()}")

# ==================== 2. 时间排序 ====================
df['采集时间'] = pd.to_datetime(df['采集时间'])
df = df.sort_values('采集时间').reset_index(drop=True)

# ==================== 3. 剔除熄火/停车（车速>0 或 转速>0） ====================
df = df[(df['车速'] > 0) | (df['发动机转速'] > 0)].copy()
print(f"剔除熄火后行数: {len(df)}，非零: {(df['nox瞬时排放强度'] != 0).sum()}")

# ==================== 4. 删除关键特征缺失值 ====================
key_cols = ['车速', '发动机转速', '发动机燃料流量', 'scr上游nox传感器输出值', 'nox瞬时排放强度']
df = df.dropna(subset=key_cols).copy()
print(f"删除缺失后行数: {len(df)}，非零: {(df['nox瞬时排放强度'] != 0).sum()}")

# ==================== 5. 车速3σ异常值剔除（仅车速） ====================
mean = df['车速'].mean()
std = df['车速'].std()
df = df[(df['车速'] >= mean - 3*std) & (df['车速'] <= mean + 3*std)]
print(f"剔除车速异常值后行数: {len(df)}，非零: {(df['nox瞬时排放强度'] != 0).sum()}")

# ==================== 6. 特征构造 ====================
# 6.1 车速单位转换
df['车速_mps'] = df['车速'] / 3.6

# 6.2 加速度（时间差法）
df['time_diff'] = df['采集时间'].diff().dt.total_seconds()
df['time_diff'] = df['time_diff'].replace(0, np.nan)
df['加速度'] = df['车速_mps'].diff() / df['time_diff']
df['加速度'] = df['加速度'].fillna(0)
df['加速度'] = df['加速度'].clip(-5, 5)

# 6.3 车辆比功率 VSP
df['VSP'] = df['车速_mps'] * (1.1*df['加速度'] + 0.132) + 0.000302 * df['车速_mps']**3

# 6.4 SCR效率（原始直接除法，无除零保护，与旧版完全一致）
df['scr_效率'] = (df['scr上游nox传感器输出值'] - df['scr下游nox传感器输出值']) / df['scr上游nox传感器输出值']
df['scr_效率'] = df['scr_效率'].clip(0, 1)

# ==================== 7. 填充其余缺失值 ====================
df = df.fillna(0)

# ==================== 8. 选择建模特征 ====================
feature_cols = ['车速', '发动机转速', '发动机净输出扭矩', '发动机燃料流量',
                '进气量', '发动机冷却液温度', 'scr入口温度', 'scr出口温度',
                '尿素消耗量', 'VSP', '加速度', 'scr_效率']
available_features = [col for col in feature_cols if col in df.columns]

X = df[available_features].copy()
y = df['nox瞬时排放强度'].copy()

# ==================== 9. 移除常数列（如油门踏板） ====================
for col in X.columns:
    if X[col].nunique() <= 1:
        X = X.drop(columns=[col])
        print(f"移除常数列: {col}")

# ==================== 10. 生成描述性统计表（表2-1） ====================
desc_df = X.describe().T[['mean', 'std', 'min', '50%', 'max']]
desc_df.columns = ['均值', '标准差', '最小值', '中位数', '最大值']
desc_df = desc_df.round(2)
print("\n表2-1 特征描述性统计：")
print(desc_df)

# ==================== 11. 标准化与数据集划分 ====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==================== 12. 保存处理后的数据 ====================
df_to_save = df[available_features + ['nox瞬时排放强度']].copy()
df_to_save.to_csv('processed_data_correct.csv', index=False)
print(f"\n最终保存数据行数: {len(df_to_save)}，非零: {(df_to_save['nox瞬时排放强度'] != 0).sum()}")