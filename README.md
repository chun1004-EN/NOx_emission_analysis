# NOx_emission_analysis
可解释机器学习预测NOx排放的代码
# 基于可解释机器学习的NOx排放预测与归因分析
​本仓库提供论文《基于可解释机器学习的NOx排放预测与归因分析》的完整代码。
- `chapter2_preprocess/data_preprocessing.py` – 数据清洗、特征构造、标准化、数据集划分
-  `chapter3_xgboost/train_models.py` – XGBoost、随机森林、SVM 模型训练与评估，生成散点图、特征重要性图及性能对比表
- `chapter4_shap/shap_analysis.py` – XGBoost、随机森林、SVM 的 SHAP 归因分析，生成摘要图、条形图、依赖图、瀑布图及跨模型对比表
运行环境：Python 3.8+，依赖库见 `requirements.txt`
