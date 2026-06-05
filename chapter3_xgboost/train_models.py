# -*- coding: utf-8 -*-
"""
第三章：基于XGBoost的NOx排放预测模型
包括数据加载、预处理、模型训练（XGBoost、随机森林、SVM）、评估及可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import seaborn as sns

# ========== 全局设置 ==========
# 设置中文字体（避免图表乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # Windows
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']        # Mac
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12          # 全局基础字体大小

# ========== 1. 加载数据 ==========
df = pd.read_csv('processed_data_correct.csv')
print("数据形状:", df.shape)
print("列名:", df.columns.tolist())

target_col = 'nox瞬时排放强度'
feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols].copy()
y = df[target_col].copy()

print(f"特征矩阵形状: {X.shape}")
print(f"目标变量非零个数: {(y != 0).sum()} / {len(y)}")

# ========== 2. 标准化和划分数据集 ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ========== 3. 训练 XGBoost ==========
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=7,
    learning_rate=0.01,
    subsample=1.0,
    colsample_bytree=0.6,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# 确定绘图坐标轴上限（基于99.9%分位数，取整为20）
combined = np.concatenate([y_test, y_pred_xgb])
upper_limit_99 = np.percentile(combined, 99.9)   # 约19.98
upper_limit = 20.0                               # 取整便于阅读
print(f"绘图坐标轴上限设为: {upper_limit} g/s (基于99.9%分位数: {upper_limit_99:.2f})")

# ========== 4. 训练随机森林 ==========
rf_model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# ========== 5. 训练 SVM ==========
svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# ========== 6. 评估指标计算与对比 ==========
def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {model_name} ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    return rmse, mae, r2

rmse_xgb, mae_xgb, r2_xgb = evaluate(y_test, y_pred_xgb, "XGBoost")
rmse_rf, mae_rf, r2_rf = evaluate(y_test, y_pred_rf, "随机森林")
rmse_svm, mae_svm, r2_svm = evaluate(y_test, y_pred_svm, "SVM")

# 保存对比结果到CSV
comparison_df = pd.DataFrame({
    '模型': ['XGBoost', '随机森林', 'SVM'],
    'RMSE': [rmse_xgb, rmse_rf, rmse_svm],
    'MAE': [mae_xgb, mae_rf, mae_svm],
    'R²': [r2_xgb, r2_rf, r2_svm]
})
comparison_df.to_csv('model_comparison.csv', index=False, encoding='utf-8-sig')
print("\n模型对比结果已保存到 model_comparison.csv")

# ========== 7. 绘图1：XGBoost散点图（预测值与真实值） ==========
# 设定坐标轴上限（可根据需要修改为12或15）
upper_limit = 12.0   # 或使用自适应：upper_limit = max(np.percentile(y_test, 95), np.percentile(y_pred_xgb, 95))
plt.figure(figsize=(6, 6))
# 不进行mask截断，保留所有点（但绘图时xlim/ylim会限制显示范围）
plt.scatter(y_test, y_pred_xgb, s=15, alpha=0.3, color='blue')
plt.plot([0, upper_limit], [0, upper_limit], 'r--', linewidth=2)
plt.xlim(0, upper_limit)
plt.ylim(0, upper_limit)
plt.xlabel("真实值 (g/s)", fontsize=16)
plt.ylabel("预测值 (g/s)", fontsize=16)
plt.tight_layout()
plt.savefig('scatter_xgb.png', dpi=300)
plt.show()

# ========== 8. 绘图2：XGBoost内置特征重要性条形图 ==========
importance = xgb_model.feature_importances_
sorted_idx = np.argsort(importance)[::-1]
sorted_importance = importance[sorted_idx]
sorted_features = [feature_cols[i] for i in sorted_idx]

plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(sorted_importance)), sorted_importance,
                color='steelblue', edgecolor='navy', linewidth=0.8, alpha=0.85)
plt.yticks(range(len(sorted_importance)), sorted_features, fontsize=11)
plt.xlabel("特征重要性", fontsize=14, labelpad=8)
plt.grid(axis='x', linestyle='--', alpha=0.5)
for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
    plt.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=10, color='black')
plt.gca().invert_yaxis()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('feature_importance_xgb.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 9. 绘图3：带边际直方图的六边形分箱图（可选，论文未使用，保留作为参考） ==========
g = sns.jointplot(
    x=y_test, y=y_pred_xgb,
    kind='hex', gridsize=50, cmap='Blues',
    marginal_kws=dict(bins=30, fill=True)
)
g.ax_joint.set_xlim(0, upper_limit)
g.ax_joint.set_ylim(0, upper_limit)
g.ax_joint.plot([0, upper_limit], [0, upper_limit], 'r--', linewidth=1.5, label='理想线 (y=x)')
g.ax_joint.legend()
g.set_axis_labels('真实值 (g/s)', '预测值 (g/s)', fontsize=12)
g.fig.suptitle('XGBoost 预测值与真实值对比 (六边形分箱+边际直方图)', y=1.02, fontsize=16)
plt.savefig('scatter_hexbin_marginal.png', dpi=300, bbox_inches='tight')
plt.show()
