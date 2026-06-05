# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ========== 设置中文字体（放大全局字号） ==========
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12

# ========== 辅助函数：绘制统一的SHAP条形图 ==========
def plot_shap_bar(mean_abs_shap, feature_names, model_name, save_path, color='steelblue'):
    """绘制统一的SHAP特征重要性条形图（降序，带数值标签）"""
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_values = mean_abs_shap[sorted_idx]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(sorted_idx)), sorted_values,
                    color=color, edgecolor='navy', linewidth=0.8, alpha=0.85)
    plt.yticks(range(len(sorted_idx)), sorted_features, fontsize=11)
    plt.xlabel('平均 |SHAP 值|', fontsize=14, labelpad=8)
    plt.title(f'{model_name} SHAP 特征重要性', fontsize=16, pad=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    for i, (bar, val) in enumerate(zip(bars, sorted_values)):
        plt.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=10, color='black')
    plt.gca().invert_yaxis()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")

# ========== 辅助函数：手动绘制依赖图（字体调大一号） ==========
def plot_dependence_manual(shap_values, feature_vals, feature_name, interact_vals,
                           interact_name, save_path, title_suffix=""):
    """
    手动绘制 SHAP 依赖图，字体调大
    shap_values: 该特征的 SHAP 值（一维数组）
    feature_vals: 该特征原始值（一维数组）
    interact_vals: 交互特征的值（用于颜色映射）
    """
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(feature_vals, shap_values, c=interact_vals, cmap='coolwarm',
                     alpha=0.7, edgecolors='none')
    cbar = plt.colorbar(sc)
    cbar.set_label(interact_name, fontsize=14)   # 调大
    plt.xlabel(feature_name, fontsize=14)        # 调大
    plt.ylabel('SHAP 值', fontsize=14)           # 调大
    plt.title(f'{feature_name} 的 SHAP 依赖图{title_suffix}', fontsize=16)  # 调大
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"已保存: {save_path}")

# ========== 1. 加载数据 ==========
data_path = 'processed_data_correct.csv'
df = pd.read_csv(data_path, encoding='utf-8-sig')
print("数据形状:", df.shape)
print("列名:", df.columns.tolist())

target_col = 'nox瞬时排放强度'
feature_cols = [col for col in df.columns if col != target_col]

# 剔除常数列（若有）
constant_cols = [col for col in feature_cols if df[col].nunique() <= 1]
if constant_cols:
    print(f"剔除常数列: {constant_cols}")
    feature_cols = [col for col in feature_cols if col not in constant_cols]

X = df[feature_cols].copy()
y = df[target_col].copy()

# ========== 2. 划分训练集和测试集（先划分再标准化） ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集样本数: {X_train_scaled.shape[0]}, 测试集样本数: {X_test_scaled.shape[0]}")
print(f"测试集中非零目标值个数: {(y_test > 0).sum()}")

# ========== 3. 训练 XGBoost 模型（与第三章参数一致） ==========
print("\n训练 XGBoost...")
xgb_params = {
    'n_estimators': 100,
    'max_depth': 7,
    'learning_rate': 0.01,
    'subsample': 1.0,
    'colsample_bytree': 0.6,
    'random_state': 42,
    'objective': 'reg:squarederror'
}
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_model.predict(X_test_scaled)
print(f"XGBoost 测试集 R²: {r2_score(y_test, y_pred_xgb):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.4f}")

# ========== 4. 训练随机森林和 SVM（用于对比） ==========
print("\n训练 Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
rf_model.fit(X_train_scaled, y_train)

print("训练 SVM...")
svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svm_model.fit(X_train_scaled, y_train)

# ========== 5. XGBoost SHAP ==========
print("\n计算 XGBoost SHAP 值...")
xgb_explainer = shap.TreeExplainer(xgb_model)
xgb_shap_values = xgb_explainer.shap_values(X_test_scaled)
print("XGBoost SHAP 计算完成。")

# 5.1 摘要图（保留英文，不加修改）
plt.figure(figsize=(10, 7))
shap.summary_plot(xgb_shap_values, X_test_scaled, feature_names=feature_cols,
                  show=False, cmap='coolwarm')
plt.title('XGBoost SHAP 摘要图', fontsize=16, pad=15)
plt.tight_layout()
plt.savefig('shap_summary_new.png', dpi=300, bbox_inches='tight')
plt.close()
print("已保存: shap_summary_new.png")

# 5.2 条形图（统一函数）
mean_abs_shap_xgb = np.abs(xgb_shap_values).mean(axis=0)
plot_shap_bar(mean_abs_shap_xgb, feature_cols, 'XGBoost', 'shap_bar_new.png')

# 5.3 依赖图（手动绘制，字体调大）
top_idx_xgb = np.argmax(mean_abs_shap_xgb)
top_feature_xgb = feature_cols[top_idx_xgb]
print(f"XGBoost 最重要的特征: {top_feature_xgb}")
plot_dependence_manual(
    shap_values=xgb_shap_values[:, top_idx_xgb],
    feature_vals=X_test[top_feature_xgb].values,
    feature_name=top_feature_xgb,
    interact_vals=X_test['发动机燃料流量'].values,
    interact_name='发动机燃料流量 (L/h)',
    save_path=f'shap_dependence_{top_feature_xgb}.png'
)

# 5.4 瀑布图1（高效率样本，删除底部文字）
mask_high = (X_test['scr_效率'] > 0.9)
candidate_indices = X_test[mask_high].index
if len(candidate_indices) > 0:
    sample_idx = candidate_indices[0]
    sample_position = X_test.index.get_loc(sample_idx)
    print(f"\n瀑布图样本1: 索引 {sample_idx}, 真实值 {y_test.iloc[sample_position]:.4f}")
    try:
        plt.clf()
        plt.figure(figsize=(10, 6))
        exp = shap.Explanation(values=xgb_shap_values[sample_position],
                               base_values=xgb_explainer.expected_value,
                               data=X_test_scaled[sample_position],
                               feature_names=feature_cols)
        shap.waterfall_plot(exp, show=False, max_display=10)
        plt.xlabel('SHAP 值 (对预测的贡献)', fontsize=14)
        plt.tight_layout()
        plt.savefig('shap_waterfall_sample.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已保存: shap_waterfall_sample.png")
    except Exception as e:
        print(f"瀑布图1绘制失败: {e}")
else:
    print("未找到SCR效率>0.9的样本，跳过瀑布图1。")

# 5.5 瀑布图2（高速巡航样本，删除底部文字）
mask_cruise = (X_test['车速'] > 60) & (X_test['scr_效率'] > 0.9) & (X_test['发动机燃料流量'] < 10)
candidate_indices2 = X_test[mask_cruise].index
if len(candidate_indices2) > 0:
    sample_idx2 = candidate_indices2[0]
    sample_position2 = X_test.index.get_loc(sample_idx2)
    print(f"瀑布图样本2: 索引 {sample_idx2}, 真实值 {y_test.iloc[sample_position2]:.4f}")
    try:
        plt.clf()
        plt.figure(figsize=(10, 6))
        exp2 = shap.Explanation(values=xgb_shap_values[sample_position2],
                                base_values=xgb_explainer.expected_value,
                                data=X_test_scaled[sample_position2],
                                feature_names=feature_cols)
        shap.waterfall_plot(exp2, show=False, max_display=10)
        plt.xlabel('SHAP 值 (对预测的贡献)', fontsize=14)
        plt.tight_layout()
        plt.savefig('shap_waterfall_sample2.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已保存: shap_waterfall_sample2.png")
    except Exception as e:
        print(f"瀑布图2绘制失败: {e}")
else:
    print("未找到高速巡航样本，跳过瀑布图2。")

# ========== 6. 随机森林 SHAP ==========
print("\n计算 Random Forest SHAP 值...")
start = time.time()
rf_explainer = shap.TreeExplainer(rf_model)
rf_shap_values = rf_explainer.shap_values(X_test_scaled)
print(f"RF SHAP 计算完成，耗时 {time.time()-start:.2f} 秒")

# 6.1 摘要图（保留英文）
plt.figure(figsize=(10, 7))
shap.summary_plot(rf_shap_values, X_test_scaled, feature_names=feature_cols,
                  show=False, cmap='coolwarm')
plt.title('随机森林 SHAP 摘要图', fontsize=16, pad=15)
plt.tight_layout()
plt.savefig('shap_summary_rf.png', dpi=300, bbox_inches='tight')
plt.close()
print("已保存: shap_summary_rf.png")

# 6.2 条形图（统一函数）
mean_abs_shap_rf = np.abs(rf_shap_values).mean(axis=0)
plot_shap_bar(mean_abs_shap_rf, feature_cols, '随机森林', 'shap_bar_rf.png')

# 6.3 依赖图（手动绘制，字体调大）
top_idx_rf = np.argmax(mean_abs_shap_rf)
top_feature_rf = feature_cols[top_idx_rf]
print(f"RF 最重要的特征: {top_feature_rf}")
plot_dependence_manual(
    shap_values=rf_shap_values[:, top_idx_rf],
    feature_vals=X_test[top_feature_rf].values,
    feature_name=top_feature_rf,
    interact_vals=X_test['发动机燃料流量'].values,
    interact_name='发动机燃料流量 (L/h)',
    save_path=f'shap_dependence_rf_{top_feature_rf}.png'
)

# ========== 7. SVM SHAP（抽样加速） ==========
print("\n注意: SVM 的 KernelExplainer 计算较慢，将随机抽取 300 个测试样本进行分析...")
sample_size = min(300, X_test_scaled.shape[0])
np.random.seed(42)
sample_indices = np.random.choice(X_test_scaled.shape[0], sample_size, replace=False)
X_test_sampled = X_test_scaled[sample_indices]
X_test_orig_sampled = X_test.iloc[sample_indices]

background = shap.sample(X_train_scaled, 100)
svm_explainer = shap.KernelExplainer(svm_model.predict, background)
svm_shap_values = svm_explainer.shap_values(X_test_sampled, nsamples=50)
print("SVM SHAP 计算完成")

# 7.1 摘要图（保留英文）
plt.figure(figsize=(10, 7))
shap.summary_plot(svm_shap_values, X_test_sampled, feature_names=feature_cols,
                  show=False, cmap='coolwarm')
plt.title('SVM SHAP 摘要图（抽样300样本）', fontsize=16, pad=15)
plt.tight_layout()
plt.savefig('shap_summary_svm_sampled.png', dpi=300, bbox_inches='tight')
plt.close()
print("已保存: shap_summary_svm_sampled.png")

# 7.2 条形图（颜色统一为 steelblue）
mean_abs_shap_svm = np.abs(svm_shap_values).mean(axis=0)
plot_shap_bar(mean_abs_shap_svm, feature_cols, 'SVM', 'shap_bar_svm_sampled.png', color='steelblue')

# 7.3 依赖图（手动绘制，字体调大）
top_idx_svm = np.argmax(mean_abs_shap_svm)
top_feature_svm = feature_cols[top_idx_svm]
print(f"SVM 最重要的特征: {top_feature_svm}")
try:
    plot_dependence_manual(
        shap_values=svm_shap_values[:, top_idx_svm],
        feature_vals=X_test_orig_sampled[top_feature_svm].values,
        feature_name=top_feature_svm,
        interact_vals=X_test_orig_sampled['发动机燃料流量'].values,
        interact_name='发动机燃料流量 (L/h)',
        save_path=f'shap_dependence_svm_{top_feature_svm}.png'
    )
except Exception as e:
    print(f"SVM 依赖图绘制失败: {e}")

# ========== 8. 保存三个模型 SHAP 值对比表 ==========
importance_df = pd.DataFrame({
    '特征': feature_cols,
    'XGBoost': mean_abs_shap_xgb,
    'RandomForest': mean_abs_shap_rf,
    'SVM': mean_abs_shap_svm
})

# 保留4位小数（不影响原始数组，只影响表格保存和打印）
importance_df[['XGBoost', 'RandomForest', 'SVM']] = importance_df[['XGBoost', 'RandomForest', 'SVM']].round(4)

importance_df.to_csv('shap_importance_comparison.csv', index=False, encoding='utf-8-sig')
print("\n三个模型的SHAP特征重要性对比已保存至 shap_importance_comparison.csv")
print(importance_df.sort_values('XGBoost', ascending=False).to_string(index=False, float_format='{:.4f}'.format))
