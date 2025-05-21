#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
网络攻击多分类识别模型
基于机器学习和深度学习方法的网络入侵检测系统
用于识别DDoS、密码攻击、后门、扫描、XSS、注入、MITM、勒索软件等网络攻击类型
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings
import joblib
from tqdm import tqdm
import time
import datetime
import logging

os.makedirs('log', exist_ok=True)
os.makedirs('output/ml', exist_ok=True)
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/机器学习模型训练日志.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('机器学习训练日志')



class XGBLoggerCallback(xgb.callback.TrainingCallback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def after_iteration(self, model, epoch, evals_log):
        eval_results = []
        # 从evals_log中提取评估结果（格式：{'train': {'logloss': [值]}, 'validation': {'logloss': [值]}}
        for data_name in evals_log:
            for metric_name, scores in evals_log[data_name].items():
                current_score = scores[-1]  # 获取当前轮次的分数
                eval_results.append(f"{data_name}-{metric_name}={current_score:.4f}")
        log_msg = f"XGBoost Iteration {epoch + 1}: " + ", ".join(eval_results)
        self.logger.info(log_msg)
        return False  # 不提前停止训练

class LGBLoggerCallback:
    def __init__(self, logger):
        self.logger = logger

    def __call__(self, env):
        iteration = env.iteration
        eval_results = env.evaluation_result_list
        log_msg = f"LightGBM Iteration {iteration}: " + ", ".join([f"{k}={v:.4f}" for k, v in eval_results])
        self.logger.info(log_msg)
# 忽略警告，使输出更加整洁
warnings.filterwarnings("ignore")

# 设置随机种子以确保结果可重现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
 # 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
class NetworkIntrusionDetection:
    """网络入侵检测类，用于处理网络流量数据和构建攻击检测模型"""
    
    def __init__(self, train_path, test_path, output_path="output/ml/output.csv"):
        """
        初始化网络入侵检测系统
        
        参数:
        train_path: 训练数据路径
        test_path: 测试数据路径
        output_path: 输出预测结果的路径
        """
        self.train_path = train_path
        self.test_path = test_path
        self.output_path = output_path
        self.models_dir = "saved_models"
        
        # 创建模型保存目录
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        # 初始化变量
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.feature_importances = None
        self.selected_features = None
        self.history = None
        # 尝试加载预训练模型和预处理器
        try:
            # 加载标准化器和特征列表
            if os.path.exists(os.path.join(self.models_dir, 'scaler.pkl')):
                self.scaler = joblib.load(os.path.join(self.models_dir, 'scaler.pkl'))
                logger.info("成功加载标准化器")
            
            if os.path.exists(os.path.join(self.models_dir, 'selected_features.pkl')):
                self.selected_features = joblib.load(os.path.join(self.models_dir, 'selected_features.pkl'))
                logger.info("成功加载特征列表")
            
            if os.path.exists(os.path.join(self.models_dir, 'label_encoder.pkl')):
                self.label_encoder = joblib.load(os.path.join(self.models_dir, 'label_encoder.pkl'))
                logger.info("成功加载标签编码器")
            
            # 尝试加载最佳模型
            if os.path.exists(os.path.join(self.models_dir, 'best_model.pkl')):
                self.best_model = joblib.load(os.path.join(self.models_dir, 'best_model.pkl'))
                logger.info("成功加载预训练模型")
        except Exception as e:
            logger.error(f"加载预训练模型时出现警告: {e}")
            logger.error("将在需要时重新训练模型。")
    
    def load_data(self,train_data_nrows=None, test_data_nrows=None):
        """加载训练和测试数据"""
        logger.info("正在加载数据...")
        
        # 加载训练集
        self.train_data = pd.read_csv(self.train_path,nrows=train_data_nrows)
        logger.info(f"读取训练数据集行数：{train_data_nrows}")
        # 加载测试集
        self.test_data = pd.read_csv(self.test_path,nrows=test_data_nrows)
        logger.info(f"读取测试数据集行数：{train_data_nrows}")
        
        logger.info(f"机器学习训练集大小: {self.train_data.shape}")
        logger.info(f"机器学习测试集大小: {self.test_data.shape}")
        
        # 检查是否存在标签列
        if 'Label' not in self.train_data.columns:
            raise ValueError("训练数据中缺少'Label'列")
            
        # 显示数据类型分布
        logger.info("\n训练集攻击类型分布:")
        logger.info(self.train_data['Label'].value_counts())
        
        return self
    
    def explore_data(self):
        """数据探索分析"""
        logger.info("\n进行数据探索分析...")
        
        # 显示统计信息
        logger.info("\n数据统计摘要:")
        logger.info(self.train_data.describe())
        
        # 检查缺失值
        missing_values = self.train_data.isnull().sum()
        missing_percent = (missing_values / len(self.train_data)) * 100
        logger.info("\n缺失值统计:")
        missing_data = pd.DataFrame({
            '缺失值数量': missing_values,
            '缺失百分比': missing_percent
        })
        logger.info(missing_data[missing_data['缺失值数量'] > 0])
        
        # 检查数据类型
        logger.info("\n数据类型:")
        logger.info(self.train_data.dtypes.value_counts())
        
        return self
    
    def preprocess_data(self):
        """数据预处理：清洗数据，处理缺失值和异常值"""
        logger.info("\n进行数据预处理...")
        
        # 1. 处理缺失值
        # 对数值列使用中位数填充
        numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.train_data[col].isnull().sum() > 0:
                median_val = self.train_data[col].median()
                self.train_data[col].fillna(median_val, inplace=True)
                if col in self.test_data.columns:
                    self.test_data[col].fillna(median_val, inplace=True)
        
        # 对分类列使用众数填充
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Label' and self.train_data[col].isnull().sum() > 0:
                mode_val = self.train_data[col].mode()[0]
                self.train_data[col].fillna(mode_val, inplace=True)
                if col in self.test_data.columns:
                    self.test_data[col].fillna(mode_val, inplace=True)
        
        # 2. 检测并处理异常值 (使用IQR方法)
        for col in numeric_cols:
            if col != 'Timestamp':  # 跳过时间戳
                Q1 = self.train_data[col].quantile(0.25)
                Q3 = self.train_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 使用上下限替换异常值
                self.train_data[col] = self.train_data[col].clip(lower_bound, upper_bound)
                if col in self.test_data.columns:
                    self.test_data[col] = self.test_data[col].clip(lower_bound, upper_bound)
        
        # 3. 处理分类特征
        # 使用标签编码器转换攻击类型标签
        if 'Label' in self.train_data.columns:
            self.train_data['Label_encoded'] = self.label_encoder.fit_transform(self.train_data['Label'])
            logger.info("\n攻击类型编码映射:")
            for i, label in enumerate(self.label_encoder.classes_):
                logger.info(f"{label}: {i}")
        
        # 处理其他分类特征
        for col in categorical_cols:
            if col != 'Label' and col in self.train_data.columns:
                self.train_data[col] = pd.factorize(self.train_data[col])[0]
                if col in self.test_data.columns:
                    self.test_data[col] = pd.factorize(self.test_data[col])[0]
        
        # 4. 特征变换 - 将时间戳转换为可用特征
        if 'Timestamp' in self.train_data.columns:
            self.train_data['Timestamp'] = pd.to_datetime(self.train_data['Timestamp'])
            self.train_data['Hour'] = self.train_data['Timestamp'].dt.hour
            self.train_data['Day'] = self.train_data['Timestamp'].dt.day
            self.train_data['DayOfWeek'] = self.train_data['Timestamp'].dt.dayofweek
            
            # 对测试集做相同处理
            if 'Timestamp' in self.test_data.columns:
                self.test_data['Timestamp'] = pd.to_datetime(self.test_data['Timestamp'])
                self.test_data['Hour'] = self.test_data['Timestamp'].dt.hour
                self.test_data['Day'] = self.test_data['Timestamp'].dt.day
                self.test_data['DayOfWeek'] = self.test_data['Timestamp'].dt.dayofweek
        
        return self
    
    def feature_engineering(self, n_features=40):
        """特征工程：特征选择和构建新特征"""
        logger.info("\n进行特征工程...")
        
        # 1. 分离特征和目标变量
        if 'Label' in self.train_data.columns:
            X = self.train_data.drop(['Label', 'Label_encoded', 'Timestamp'], axis=1, errors='ignore')
            y = self.train_data['Label_encoded']
            
            # 检查每个类的样本数量
            class_counts = np.bincount(y)
            min_samples = np.min(class_counts[class_counts > 0])
            logger.info(f"最小类别的样本数: {min_samples}")
            
            # 2. 划分训练集和验证集
            if min_samples >= 2:
                # 只有当所有类至少有2个样本时才使用stratify
                logger.info("使用分层抽样进行训练集和验证集划分")
                self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
                )
            else:
                # 对于样本数量不足的情况，不使用stratify
                logger.info("检测到样本稀少的类别，使用随机抽样进行训练集和验证集划分")
                self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                    X, y, test_size=0.2, random_state=RANDOM_SEED
                )
                
                # 输出训练集和验证集中各类别的分布
                logger.info("训练集类别分布:")
                logger.info(np.bincount(self.y_train))
                logger.info("验证集类别分布:")
                logger.info(np.bincount(self.y_val))
            
            # 3. 特征选择
            logger.info("执行特征选择...")
            # 使用卡方检验进行特征选择
            selector = SelectKBest(score_func=f_classif, k=n_features)
            selector.fit(self.X_train, self.y_train)
            
            # 获取选中的特征列名
            features_scores = pd.DataFrame({
                'Feature': X.columns,
                'Score': selector.scores_
            }).sort_values(by='Score', ascending=False)
            
            self.selected_features = features_scores.head(n_features)['Feature'].tolist()
            logger.info(f"选择了前{n_features}个最重要的特征")
            
            # 应用特征选择
            self.X_train = self.X_train[self.selected_features]
            self.X_val = self.X_val[self.selected_features]
            
            # 4. 特征标准化
            logger.info("执行特征标准化...")
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_val = self.scaler.transform(self.X_val)
            
            # 5. 处理数据不平衡 - 使用SMOTE进行过采样
            logger.info("使用SMOTE处理类别不平衡...")
            # 检查每个类别是否至少有一个样本
            class_counts_train = np.bincount(self.y_train)
            if np.all(class_counts_train[np.unique(self.y_train)] >= 1):
                try:
                    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=min(5, min_samples-1) if min_samples > 1 else 1)
                    self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
                    logger.info(f"SMOTE过采样后的训练集大小: {self.X_train.shape}")
                except ValueError as e:
                    logger.info(f"SMOTE过采样失败: {e}")
                    logger.info("跳过SMOTE过采样步骤，继续使用原始数据")
            else:
                logger.error("由于类别样本不足，跳过SMOTE过采样步骤")
            
            # 保存标准化器和特征选择器，用于后续测试数据处理
            joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))
            joblib.dump(self.selected_features, os.path.join(self.models_dir, 'selected_features.pkl'))
            
        return self
    
    def train_models(self,model_type='all'):
        """训练和评估不同的模型"""
        logger.info("\n开始模型训练与评估...")
        
        # 定义评估指标
        def evaluate_model(model, X_train, y_train, X_val, y_val):
            # 获取所有唯一类别
            all_classes = np.sort(np.unique(np.concatenate([y_train, y_val])))
            logger.info(f"训练数据中的类别: {np.sort(np.unique(y_train))}")
            logger.info(f"验证数据中的类别: {np.sort(np.unique(y_val))}")
            logger.info(f"合并后的所有类别: {all_classes}")
            
            # 特殊处理XGBoost - 确保数据标签连续
            if isinstance(model, XGBClassifier):
                # 创建一个映射，将现有类别映射到连续的索引
                class_map = {cls: i for i, cls in enumerate(all_classes)}
                logger.info(f"类别映射: {class_map}")
                
                # 应用映射到训练和验证数据
                y_train_mapped = np.array([class_map[cls] for cls in y_train])
                y_val_mapped = np.array([class_map[cls] for cls in y_val])
                
                # 训练模型
                #model.fit(X_train, y_train_mapped)
                model.fit(X_train, y_train_mapped, eval_set=[(X_val, y_val_mapped)])
                # 预测验证集
                y_pred_mapped = model.predict(X_val)
                
                # 将预测结果映射回原始类别
                reverse_map = {i: cls for cls, i in class_map.items()}
                y_pred = np.array([reverse_map[cls] for cls in y_pred_mapped])
            else:
                # # 对于其他模型，正常训练和预测
                # model.fit(X_train, y_train)
                # y_pred = model.predict(X_val)
                # 对于其他模型，正常训练和预测
                if isinstance(model, lgb.LGBMClassifier):
                    # 使用LightGBM回调记录训练日志
                    model.fit(X_train, y_train, callbacks=[LGBLoggerCallback(logger)])
                elif isinstance(model, RandomForestClassifier):
                    # 记录RandomForest训练开始和结束日志
                    logger.info("开始训练RandomForest模型...")
                    model.fit(X_train, y_train)
                    logger.info("RandomForest模型训练完成。")
                else:
                    model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            
            
            # 计算准确率
            accuracy = accuracy_score(y_val, y_pred)
            
            # 生成分类报告
            report = classification_report(y_val, y_pred, output_dict=True)
            
            return model, accuracy, report
        
        # 根据model_type创建不同的模型进行调用
        callbacks = [XGBLoggerCallback(logger=logger)]
        if model_type == 'all':
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1),
                'XGBoost': XGBClassifier(n_estimators=100,callbacks=callbacks, random_state=RANDOM_SEED, n_jobs=-1),
                'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
            }
        elif model_type == 'LightGBM':
            models = {
                'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
            }
        elif model_type == 'XGBoost':
            models = {
                'XGBoost': XGBClassifier(n_estimators=100,callbacks=callbacks, random_state=RANDOM_SEED, n_jobs=-1)
            }
        elif model_type == 'RandomForest':
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
            }
        else:
            raise ValueError("无效的模型类型")

        # 存储结果
        results = {}
        best_accuracy = 0
        
        # 训练和评估每个模型
        for name, model in tqdm(models.items(), desc="训练模型"):
            logger.info(f"\n训练 {name} 模型...")
            trained_model, accuracy, report = evaluate_model(
                model, self.X_train, self.y_train, self.X_val, self.y_val
            )
            
            results[name] = {
                'model': trained_model,
                'accuracy': accuracy,
                'report': report
            }
            
            logger.info(f"{name} 模型在验证集上的准确率: {accuracy:.4f}")
            
            # 保存每个模型
            joblib.dump(trained_model, os.path.join(self.models_dir, f'{name}_model.pkl'))
            
            # 更新最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = trained_model
                joblib.dump(self.best_model, os.path.join(self.models_dir, 'best_model.pkl'))
                # 如果是树模型，获取特征重要性
                if hasattr(trained_model, 'feature_importances_'):
                    self.feature_importances = pd.DataFrame({
                        'Feature': self.selected_features,
                        'Importance': trained_model.feature_importances_
                    }).sort_values(by='Importance', ascending=False)
        
        # 打印所有模型的表现
        logger.info("\n所有模型在验证集上的表现:")
        for name, result in results.items():
            logger.info(f"{name}: 准确率 = {result['accuracy']:.4f}")
            
            # 输出详细分类报告
            logger.info(f"\n{name} 分类报告:")
            report_df = pd.DataFrame(result['report']).transpose()
            logger.info(report_df)
            
            # 绘制混淆矩阵
            y_pred = result['model'].predict(self.X_val)
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(self.y_val, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=self.label_encoder.classes_,
                        yticklabels=self.label_encoder.classes_)
            plt.title(f'{name} 混淆矩阵',fontsize=14)
            plt.ylabel('真实标签')
            plt.xlabel('预测标签')
            plt.tight_layout()
            plt.savefig(f'saved_models/{name}_confusion_matrix.png',
                        dpi=1024,#
                        bbox_inches='tight')

        # 如果有特征重要性，绘制特征重要性图
        if self.feature_importances is not None:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', 
                        data=self.feature_importances)
            plt.title('特征重要性',fontsize=14)
            plt.tight_layout()
            plt.savefig('saved_models/ml_feature_importance.png',
                        dpi=1024,  #
                        bbox_inches='tight'
                        )
            
        # 保存标签编码器，用于后续测试数据处理
        joblib.dump(self.label_encoder, os.path.join(self.models_dir, 'label_encoder.pkl'))
        logger.info("已保存标签编码器")
        
        return self
    def plot_training_history(self):
        """绘制训练历史"""
        if self.history is not None:
           
            # 准确率
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'],marker='o')
            plt.plot(self.history.history['val_accuracy'],marker='o')
            plt.title('模型准确率',fontsize=12)
            plt.ylabel('准确率',fontsize=10)
            plt.xlabel('Epoch',fontsize=10)
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.legend(['训练集', '验证集'], loc='upper left')
            
            # 损失
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('模型损失',fontsize=12)
            plt.ylabel('损失',fontsize=10)
            plt.xlabel('Epoch',fontsize=10)
            plt.legend(['训练集', '验证集'], loc='upper left')
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(self.models_dir, 'training_history.png'))
    def hyperparameter_tuning(self):
        """对最佳模型进行超参数调优"""
        logger.info("\n进行超参数调优...")
        
        # # 根据最佳模型类型选择不同的参数网格
        # if isinstance(self.best_model, RandomForestClassifier):
        #     logger.info("调优随机森林模型...")
        #     param_grid = {
        #         'n_estimators': [100, 200, 300],
        #         'max_depth': [10, 20, 30, None],
        #         'min_samples_split': [2, 5, 10],
        #         'min_samples_leaf': [1, 2, 4]
        #     }
        #     model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
        #
        # elif isinstance(self.best_model, GradientBoostingClassifier):
        #     logger.info("调优梯度提升模型...")
        #     param_grid = {
        #         'n_estimators': [100, 200, 300],
        #         'learning_rate': [0.01, 0.1, 0.2],
        #         'max_depth': [3, 5, 7]
        #     }
        #     model = GradientBoostingClassifier(random_state=RANDOM_SEED)
        #
        # elif isinstance(self.best_model, XGBClassifier):
        #     logger.info("调优XGBoost模型...")
        #     param_grid = {
        #         'n_estimators': [100, 200, 300],
        #         'learning_rate': [0.01, 0.1, 0.2],
        #         'max_depth': [3, 5, 7],
        #         'subsample': [0.8, 0.9, 1.0]
        #     }
        #     model = XGBClassifier(random_state=RANDOM_SEED, n_jobs=-1)
        #
        # elif isinstance(self.best_model, lgb.LGBMClassifier):
        #     logger.info("调优LightGBM模型...")
        #     param_grid = {
        #         'n_estimators': [100, 200, 300],
        #         'learning_rate': [0.01, 0.1, 0.2],
        #         'max_depth': [3, 5, 7, -1],
        #         'num_leaves': [31, 50, 100]
        #     }
        #     model = lgb.LGBMClassifier(random_state=RANDOM_SEED, n_jobs=-1)
        #
        # else:
        #     logger.info("无法识别的模型类型，跳过超参数调优")
        #     return self
        # 根据最佳模型类型选择不同的参数网格
        if isinstance(self.best_model, RandomForestClassifier):
            logger.info("调优随机森林模型...")
            param_grid = {
                'n_estimators': [300, 400, 500],  # 扩大上限适应大规模数据
                'max_depth': [10, 20, 30, None],  # 增加深度范围
                'min_samples_split': [2, 4, 6, 8, 10],  # 细化分割阈值
                'min_samples_leaf': [1, 2, 3, 4],  # 细化叶子节点样本数
                'class_weight': ['balanced', None]  # 处理类别不平衡
            }
            model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)

        elif isinstance(self.best_model, XGBClassifier):
            logging.info("调优XGBoost模型...")
            param_grid = {
                'n_estimators': [200, 300, 400],  # 适应大规模数据增加基学习器数量
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],  # 细化学习率粒度
                'max_depth': [4, 6, 8],  # 增加深度范围
                'subsample': [0.7, 0.8, 0.9, 1.0],  # 扩大子采样范围
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],  # 扩大列采样范围
                'reg_alpha': [0, 0.05, 0.1, 0.3],  # 细化L1正则化
                'reg_lambda': [0, 0.05, 0.1, 0.3]  # 细化L2正则化
            }
            model = XGBClassifier(random_state=RANDOM_SEED, n_jobs=-1)

        elif isinstance(self.best_model, lgb.LGBMClassifier):
            logging.info("调优LightGBM模型...")
            param_grid = {
                'n_estimators': [200, 300, 400],  # 适应大规模数据增加基学习器数量
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],  # 细化学习率粒度
                'max_depth': [4, 6, 8, -1],  # 增加深度范围（-1表示不限制）
                'num_leaves': [31, 50, 100, 200],  # 扩大叶子节点数量范围
                'feature_fraction': [0.7, 0.8, 0.9, 1.0],  # 扩大特征采样范围
                'reg_alpha': [0, 0.05, 0.1, 0.3],  # 细化L1正则化
                'reg_lambda': [0, 0.05, 0.1, 0.3]  # 细化L2正则化
            }
            model = lgb.LGBMClassifier(random_state=RANDOM_SEED, n_jobs=-1)

        else:
            logging.info("无法识别的模型类型，跳过超参数调优")
            return self
        # # 使用分层K折交叉验证
        # cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        #这种方法太耗资源，设备不足以支持
        # # 网格搜索
        # grid_search = GridSearchCV(
        #     estimator=model,
        #     param_grid=param_grid,
        #     cv=cv,
        #     scoring='accuracy',
        #     n_jobs=2,
        #     verbose=1
        # )
        #
        # # 执行网格搜索
        # grid_search.fit(self.X_train, self.y_train)

        # 改用随机搜索提高大规模数据调优效率
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=50,  # 随机搜索迭代次数（比网格搜索更高效）
            scoring='f1_macro',  # 多分类使用F1宏平均
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),#使用分层K折交叉验证
            n_jobs=10,  # 使用CPU核心
            verbose=1,
            random_state=RANDOM_SEED
        )
        search.fit(self.X_train, self.y_train)
        # 打印最佳参数
        logger.info(f"最佳参数: {search.best_params_}")
        logger.info(f"最佳交叉验证分数: {search.best_score_:.4f}")
        
        # 使用最佳参数更新模型
        self.best_model = search.best_estimator_
        
        # 在验证集上评估调优后的模型
        y_pred = self.best_model.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        
        logger.info(f"调优后模型在验证集上的准确率: {accuracy:.4f}")
        
        # 保存调优后的最佳模型
        joblib.dump(self.best_model, os.path.join(self.models_dir, 'best_model.pkl'))
        
        return self
    
    def predict(self,model_type='all'):
        """使用最佳模型预测测试集"""
        logger.info("\n使用最佳模型预测测试集...")

        # 检查模型是否已训练

        if model_type == 'all' and os.path.exists(os.path.join(self.models_dir, 'best_model.pkl')):
            try:
                self.best_model = joblib.load(os.path.join(self.models_dir, 'best_model.pkl'))
                logger.info("成功加载best_model")
            except Exception as e:
                logger.error(f"best_model未保存: {e}")
        elif model_type == 'RandomForest' and os.path.exists(os.path.join(self.models_dir, 'RandomForest_model.pkl')):
            try:
                self.best_model = joblib.load(os.path.join(self.models_dir, 'RandomForest_model.pkl'))
                logger.info("成功加载RandomForest_model")
            except Exception as e:
                logger.error(f"RandomForest_model未保存: {e}")
        elif model_type == 'XGBoost' and os.path.exists(os.path.join(self.models_dir, 'XGBoost_model.pkl')):
            try:
                self.best_model = joblib.load(os.path.join(self.models_dir, 'XGBoost_model.pkl'))
                logger.info("成功加载XGBoost_model")
            except Exception as e:
                logger.error(f"XGBoost_model未保存: {e}")
        elif model_type == 'LightGBM' and os.path.exists(os.path.join(self.models_dir, 'LightGBM_model.pkl')):
            try:
                self.best_model = joblib.load(os.path.join(self.models_dir, 'LightGBM_model.pkl'))
                logger.info("成功加载LightGBM_model")
            except Exception as e:
                logger.error(f"LightGBM_model未保存: {e}")
        else:
            logger.error("无机器学习模型可被加载")
        # 准备测试数据
        X_test = self.test_data.copy()
        
        # 删除不必要的列（如果存在）
        X_test = X_test.drop(['Timestamp'], axis=1, errors='ignore')
        
        # 确保测试集只包含选定的特征
        X_test = X_test[self.selected_features]
        
        # 应用相同的标准化
        X_test = self.scaler.transform(X_test)

        # 预测
        test_pred = self.best_model.predict(X_test)
        
        # 将数值标签转换回原始标签
        test_pred_labels = self.label_encoder.inverse_transform(test_pred)
        
        # 创建输出数据框
        output_df = pd.DataFrame({
            'index': range(len(test_pred_labels)),
            'Label': test_pred_labels
        })
        
        # 保存预测结果
        output_df.to_csv(self.output_path, index=False)
        logger.info(f"预测结果已保存至 {self.output_path}")
        
        return output_df
    
    def run_pipeline(self,train_data_nrows=None,test_data_nrows=None,model_type='all',use_tuning=False):
        """运行完整的数据处理和模型训练管道"""
        try:
            logger.info("\n开始运行网络入侵检测流水线...")
            pipeline = (self.load_data(train_data_nrows,test_data_nrows)
                 .explore_data()
                 .preprocess_data()
                 .feature_engineering()
                 .train_models(model_type))
            if use_tuning:
                pipeline = pipeline.hyperparameter_tuning()
            pipeline.predict(model_type)
            
            logger.info("\n完整的入侵检测流水线已成功运行!")
            return self
        except FileNotFoundError as e:
            logger.error(f"\n错误:找不到文件 - {e}")
            logger.error("请确保数据文件路径正确，并且文件存在。")
            return self
        except Exception as e:
            logger.error(f"\n错误:运行流水线时发生异常 - {e}")
            logger.error("请检查数据格式是否正确，或查看日志获取更多信息。")
            return self


# 主程序入口
if __name__ == "__main__":
    # 设置文件路径
    train_path = "dataset_train.csv"
    test_path = "dataset_test.csv"
    output_path = "output/ml/output.csv"
    nrows=None
    use_tuning=False
    # 创建并运行网络入侵检测流水线
    detector = NetworkIntrusionDetection(train_path, test_path, output_path)
    detector.run_pipeline(nrows,nrows,use_tuning)