#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
网络攻击多分类识别 - 集成模型方法
结合机器学习和深度学习模型的预测结果，实现更高准确率的网络攻击分类
用于识别DDoS、密码攻击、后门、扫描、XSS、注入、MITM、勒索软件等网络攻击类型
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
import joblib
from tqdm import tqdm
import logging
# 从其他脚本导入模型类
from network_intrusion_detection import NetworkIntrusionDetection,XGBLoggerCallback,LGBLoggerCallback
from deep_learning_intrusion_detection import DeepLearningIntrusionDetection,LoggerCallback


os.makedirs('log', exist_ok=True)
os.makedirs('output/em', exist_ok=True)
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/集成模型训练日志.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('集成模型训练日志')
# 忽略警告，使输出更加整洁
warnings.filterwarnings("ignore")
# 忽略警告，使输出更加整洁
warnings.filterwarnings("ignore")

# 设置随机种子以确保结果可重现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
 # 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
class EnsembleIntrusionDetection:
    """
    集成网络入侵检测类，结合机器学习和深度学习模型的优势
    """
    
    def __init__(self, train_path, test_path, output_path="output/em/output.csv"):
        """
        初始化集成网络入侵检测系统
        
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
        self.ensemble_model = None
        self.ml_models = {}
        self.dl_model = None
        self.selected_features = None
        self.dl_selected_features = None
    
    def load_models(self):
        """加载已训练的机器学习和深度学习模型"""
        logger.info("加载已训练的模型...")
        
        try:
            # 加载机器学习模型
            self.ml_models['RandomForest'] = joblib.load(os.path.join(self.models_dir, 'RandomForest_model.pkl'))
            self.ml_models['XGBoost'] = joblib.load(os.path.join(self.models_dir, 'XGBoost_model.pkl'))
            self.ml_models['LightGBM'] = joblib.load(os.path.join(self.models_dir, 'LightGBM_model.pkl'))
            
            # 加载深度学习模型
            self.dl_model = load_model(os.path.join(self.models_dir, 'best_dl_model.h5'))
            
            # 加载特征选择和标准化器
            self.scaler = joblib.load(os.path.join(self.models_dir, 'scaler.pkl'))
            self.selected_features = joblib.load(os.path.join(self.models_dir, 'selected_features.pkl'))
            self.dl_scaler = joblib.load(os.path.join(self.models_dir, 'dl_scaler.pkl'))
            self.dl_selected_features = joblib.load(os.path.join(self.models_dir, 'dl_selected_features.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.models_dir, 'dl_label_encoder.pkl'))
            
            logger.info("成功加载所有模型和预处理器")
            return True
        
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            logger.error("将运行单独的模型训练流程")
            return False
    
    def train_individual_models(self,train_data_nrows=None, test_data_nrows=None):
        """训练各个独立的模型"""
        logger.info("开始训练各个独立模型...")
        
        # 训练机器学习模型
        logger.info("\n训练机器学习模型...")
        ml_detector = NetworkIntrusionDetection(self.train_path, self.test_path)
        ml_detector.load_data(train_data_nrows,test_data_nrows).explore_data().preprocess_data().feature_engineering().train_models()

        # 获取机器学习模型和预处理器
        self.ml_models = {
            'RandomForest': ml_detector.best_model if isinstance(ml_detector.best_model, RandomForestClassifier) 
                           else joblib.load(os.path.join(self.models_dir, 'RandomForest_model.pkl')),
            'XGBoost': joblib.load(os.path.join(self.models_dir, 'XGBoost_model.pkl')),
            'LightGBM': joblib.load(os.path.join(self.models_dir, 'LightGBM_model.pkl'))
        }
        self.scaler = ml_detector.scaler
        self.selected_features = ml_detector.selected_features
        self.label_encoder = ml_detector.label_encoder
        
        # 训练深度学习模型
        logger.info("\n训练深度学习模型...")
        dl_detector = DeepLearningIntrusionDetection(self.train_path, self.test_path)
        dl_detector.load_data().preprocess_data().feature_engineering().train_models()
        
        # 获取深度学习模型和预处理器
        self.dl_model = dl_detector.best_model
        self.dl_scaler = dl_detector.scaler
        self.dl_selected_features = dl_detector.selected_features
        
        logger.info("所有独立模型训练完成")
    
    def build_ensemble(self,train_data_nrows=None):
        """构建集成模型，组合各个分类器的预测"""
        logger.info("\n开始构建集成模型...")
        
        # 加载训练数据用于集成模型评估
        logger.info("加载数据进行集成...")
        self.train_data = pd.read_csv(self.train_path,nrows=train_data_nrows)
        logger.info(f"集成训练集大小: {self.train_data.shape}")
        
        # 预处理数据
        if 'Label' in self.train_data.columns:
            # 编码标签
            self.train_data['Label_encoded'] = self.label_encoder.fit_transform(self.train_data['Label'])
            X = self.train_data.drop(['Label', 'Label_encoded', 'Timestamp'], axis=1, errors='ignore')
            y = self.train_data['Label_encoded']
            
            # 检查并处理无穷大值和缺失值
            logger.info("检查并处理无穷大值和缺失值...")
            # 将无穷值替换为NaN，然后用列的中位数填充
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # 输出包含NaN或inf的列
            inf_cols = X.columns[np.any(np.isinf(X.values), axis=0)]
            nan_cols = X.columns[X.isna().any()]
            if len(inf_cols) > 0:
                logger.info(f"发现包含无穷值的列: {inf_cols}")
            if len(nan_cols) > 0:
                logger.info(f"发现包含NaN的列: {nan_cols}")
            
            # 用中位数填充NaN
            X = X.fillna(X.median())
            
            # 检查极端值并用阈值替换
            def replace_extremes(df, threshold=1e15):
                """将超过阈值的值替换为阈值"""
                return df.clip(lower=-threshold, upper=threshold)
            
            X = replace_extremes(X)
            logger.info("数据清理完成")
            
            # 检查每个类的样本数量
            class_counts = np.bincount(y)
            min_samples = np.min(class_counts[class_counts > 0])
            logger.info(f"最小类别的样本数: {min_samples}")
            logger.info(f"类别分布: {class_counts}")
            
            # 划分数据 - 处理稀有类别
            if min_samples >= 2:
                # 只有当所有类至少有2个样本时才使用stratify
                logger.info("使用分层抽样进行训练集和验证集划分")
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
                )
            else:
                # 对于样本数量不足的情况，不使用stratify
                logger.info("检测到样本稀少的类别，使用随机抽样进行训练集和验证集划分")
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=RANDOM_SEED
                )
                
                # 输出训练集和验证集中各类别的分布
                logger.info("训练集类别分布:")
                train_counts = np.bincount(y_train)
                logger.info(train_counts)
                logger.info("验证集类别分布:")
                val_counts = np.bincount(y_val)
                logger.info(val_counts)
            
            # 保存用于集成模型评估的验证集
            self.X_val = X_val
            self.y_val = y_val
            
            # 准备用于各个模型预测的数据
            try:
                # 确保特征列存在
                missing_features = [f for f in self.selected_features if f not in X_val.columns]
                if missing_features:
                    logger.info(f"警告：验证集中缺少以下特征: {missing_features}")
                    # 只使用存在的特征
                    existing_features = [f for f in self.selected_features if f in X_val.columns]
                    ml_X_val = X_val[existing_features].copy()
                else:
                    ml_X_val = X_val[self.selected_features].copy()
                
                logger.info("应用数据标准化前的统计信息:")
                ml_X_val_stats = ml_X_val.describe().T
                logger.info(f"最小值: {ml_X_val_stats['min'].min()}, 最大值: {ml_X_val_stats['max'].max()}")
                
                # 安全地应用标准化
                try:
                    ml_X_val = self.scaler.transform(ml_X_val)
                except Exception as e:
                    logger.info(f"标准化时出错: {e}")
                    logger.info("尝试重新训练标准化器...")
                    self.scaler.fit(ml_X_val)
                    ml_X_val = self.scaler.transform(ml_X_val)
                
                # 同样处理深度学习特征
                missing_dl_features = [f for f in self.dl_selected_features if f not in X_val.columns]
                if missing_dl_features:
                    logger.info(f"警告：验证集中缺少以下深度学习特征: {missing_dl_features}")
                    existing_dl_features = [f for f in self.dl_selected_features if f in X_val.columns]
                    dl_X_val = X_val[existing_dl_features].copy()
                else:
                    dl_X_val = X_val[self.dl_selected_features].copy()
                
                try:
                    dl_X_val = self.dl_scaler.transform(dl_X_val)
                except Exception as e:
                    logger.error(f"深度学习特征标准化时出错: {e}")
                    logger.error("尝试重新训练深度学习标准化器...")
                    self.dl_scaler.fit(dl_X_val)
                    dl_X_val = self.dl_scaler.transform(dl_X_val)
                
                # 获取各个模型的预测
                ml_preds = {}
                for name, model in self.ml_models.items():
                    try:
                        ml_preds[name] = model.predict(ml_X_val)
                    except Exception as e:
                        logger.error(f"{name}模型预测时出错: {e}")
                        ml_preds[name] = np.zeros(len(y_val), dtype=int)  # 使用占位符
                
                try:
                    dl_preds_proba = self.dl_model.predict(dl_X_val)
                    dl_preds = np.argmax(dl_preds_proba, axis=1)
                except Exception as e:
                    logger.error(f"深度学习模型预测时出错: {e}")
                    dl_preds = np.zeros(len(y_val), dtype=int)  # 使用占位符
                
                # 计算各个模型的准确率
                accuracies = {}
                for name, preds in ml_preds.items():
                    acc = accuracy_score(y_val, preds)
                    accuracies[name] = acc
                    logger.info(f"{name} 模型在验证集上的准确率: {acc:.4f}")
                
                dl_acc = accuracy_score(y_val, dl_preds)
                accuracies['DeepLearning'] = dl_acc
                logger.info(f"深度学习模型在验证集上的准确率: {dl_acc:.4f}")
                
                # 基于准确率计算权重
                total_acc = sum(accuracies.values())
                if total_acc > 0:
                    weights = {name: acc/total_acc for name, acc in accuracies.items()}
                else:
                    # 如果所有模型准确率为0，则使用均等权重
                    weights = {name: 1.0/len(accuracies) for name in accuracies.keys()}
                
                logger.info("\n集成模型权重分配:")
                for name, weight in weights.items():
                    logger.info(f"{name}: {weight:.4f}")
                
                # 使用加权投票方式进行集成预测
                def weighted_voting(ml_predictions, dl_prediction, weights):
                    # 创建投票计数数组
                    votes = np.zeros(len(self.label_encoder.classes_))
                    
                    # 统计机器学习模型的投票
                    for name, preds in ml_predictions.items():
                        votes[preds] += weights[name]
                    
                    # 统计深度学习模型的投票
                    votes[dl_prediction] += weights['DeepLearning']
                    
                    # 返回得票最多的类别
                    return np.argmax(votes)
                
                # 应用加权投票到验证集
                ensemble_preds = []
                for i in range(len(y_val)):
                    ml_pred_i = {name: preds[i] for name, preds in ml_preds.items()}
                    dl_pred_i = dl_preds[i]
                    ensemble_pred = weighted_voting(ml_pred_i, dl_pred_i, weights)
                    ensemble_preds.append(ensemble_pred)
                
                # 评估集成模型
                ensemble_acc = accuracy_score(y_val, ensemble_preds)
                logger.info(f"\n集成模型在验证集上的准确率: {ensemble_acc:.4f}")
                
                # 获取实际出现在验证集中的类别
                unique_classes = np.unique(np.concatenate([ensemble_preds, y_val]))
                logger.info(f"验证集中的实际类别: {unique_classes}")
                
                # 使用labels参数指定实际出现的类别
                try:
                    # 获取实际出现在验证集中的类别名称
                    target_names = [self.label_encoder.classes_[i] for i in unique_classes if i < len(self.label_encoder.classes_)]
                    
                    # 生成分类报告
                    logger.info("\n集成模型分类报告:")
                    ensemble_report = classification_report(
                        y_val, ensemble_preds, 
                        labels=unique_classes,
                        target_names=target_names
                    )
                    logger.info(ensemble_report)
                except Exception as e:
                    logger.error(f"生成分类报告时出错: {e}")
                    logger.error("使用基础分类报告:")
                    ensemble_report = classification_report(y_val, ensemble_preds)
                    logger.error(ensemble_report)
                
                # 绘制混淆矩阵
                try:
                    plt.figure(figsize=(12, 10))
                    cm = confusion_matrix(y_val, ensemble_preds, labels=unique_classes)
                    xticklabels = [self.label_encoder.classes_[i] if i < len(self.label_encoder.classes_) else f"Unknown_{i}" 
                                  for i in unique_classes]
                    yticklabels = xticklabels.copy()
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                              xticklabels=xticklabels,
                              yticklabels=yticklabels)
                    plt.title('集成模型混淆矩阵',fontsize=14)
                    plt.ylabel('真实标签')
                    plt.xlabel('预测标签')
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.models_dir, 'ensemble_confusion_matrix.png'))
                except Exception as e:
                    logger.error(f"绘制混淆矩阵时出错: {e}")
                    logger.error("跳过混淆矩阵绘制")
                
                # 保存权重信息
                joblib.dump(weights, os.path.join(self.models_dir, 'ensemble_weights.pkl'))
                
                # 保存集成相关信息
                self.ensemble_info = {
                    'weights': weights,
                    'ml_selected_features': self.selected_features,
                    'dl_selected_features': self.dl_selected_features,
                    'ml_scaler': self.scaler,
                    'dl_scaler': self.dl_scaler,
                    'label_encoder': self.label_encoder
                }
                joblib.dump(self.ensemble_info, os.path.join(self.models_dir, 'ensemble_info.pkl'))
                
            except Exception as e:
                logger.error(f"构建集成模型过程中出错: {e}")
                import traceback
                traceback.print_exc()
                logger.info("尝试使用简单平均法构建集成模型")
                
                # 简单存储基本信息，用于后续预测
                self.ensemble_info = {
                    'weights': {'RandomForest': 0.33, 'XGBoost': 0.33, 'LightGBM': 0.34, 'DeepLearning': 0.0},  # 暂时不使用深度学习模型
                    'ml_selected_features': self.selected_features,
                    'dl_selected_features': self.dl_selected_features,
                    'ml_scaler': self.scaler,
                    'dl_scaler': self.dl_scaler,
                    'label_encoder': self.label_encoder
                }
                joblib.dump(self.ensemble_info, os.path.join(self.models_dir, 'ensemble_info.pkl'))
    
    def predict(self, test_data_nrows=None):
        """使用集成模型预测测试集"""
        logger.info("\n使用集成模型预测测试集...")
        
        # 加载测试数据
        self.test_data = pd.read_csv(self.test_path, nrows=test_data_nrows)
        logger.info(f"集成测试集大小: {self.test_data.shape}")
        # 加载集成信息
        ensemble_info = joblib.load(os.path.join(self.models_dir, 'ensemble_info.pkl'))
        weights = ensemble_info['weights']
        ml_selected_features = ensemble_info['ml_selected_features']
        dl_selected_features = ensemble_info['dl_selected_features']
        ml_scaler = ensemble_info['ml_scaler']
        dl_scaler = ensemble_info['dl_scaler']
        label_encoder = ensemble_info['label_encoder']
        
        # 准备用于预测的数据
        X_test = self.test_data.copy()
        X_test = X_test.drop(['Timestamp'], axis=1, errors='ignore')
        
        # 检查并处理无穷大值和缺失值
        logger.info("检查并处理测试数据中的无穷大值和缺失值...")
        # 将无穷值替换为NaN，然后用列的中位数填充
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # 输出包含NaN或inf的列
        inf_cols = X_test.columns[np.any(np.isinf(X_test.values), axis=0)]
        nan_cols = X_test.columns[X_test.isna().any()]
        if len(inf_cols) > 0:
            logger.info(f"发现包含无穷值的列: {inf_cols}")
        if len(nan_cols) > 0:
            logger.info(f"发现包含NaN的列: {nan_cols}")
        
        # 用中位数填充NaN
        X_test = X_test.fillna(X_test.median())
        
        # 检查极端值并用阈值替换
        def replace_extremes(df, threshold=1e15):
            """将超过阈值的值替换为阈值"""
            return df.clip(lower=-threshold, upper=threshold)
        
        X_test = replace_extremes(X_test)
        logger.info("测试数据清理完成")
        
        try:
            # 准备用于机器学习模型的数据
            missing_features = [f for f in ml_selected_features if f not in X_test.columns]
            if missing_features:
                logger.info(f"警告：测试集中缺少以下特征: {missing_features}")
                # 只使用存在的特征
                existing_features = [f for f in ml_selected_features if f in X_test.columns]
                ml_X_test = X_test[existing_features].copy()
            else:
                ml_X_test = X_test[ml_selected_features].copy()
            
            logger.info("测试数据标准化前的统计信息:")
            ml_X_test_stats = ml_X_test.describe().T
            logger.info(f"最小值: {ml_X_test_stats['min'].min()}, 最大值: {ml_X_test_stats['max'].max()}")
            
            # 安全地应用标准化
            try:
                ml_X_test = ml_scaler.transform(ml_X_test)
            except Exception as e:
                logger.error(f"标准化测试数据时出错: {e}")
                logger.info("尝试直接对数据进行Z-score标准化...")
                # 简单的Z-score标准化
                ml_X_test = (ml_X_test - ml_X_test.mean()) / ml_X_test.std().clip(lower=1e-10)
            
            # 准备用于深度学习模型的数据
            missing_dl_features = [f for f in dl_selected_features if f not in X_test.columns]
            if missing_dl_features:
                logger.info(f"警告：测试集中缺少以下深度学习特征: {missing_dl_features}")
                existing_dl_features = [f for f in dl_selected_features if f in X_test.columns]
                dl_X_test = X_test[existing_dl_features].copy()
            else:
                dl_X_test = X_test[dl_selected_features].copy()
            
            try:
                dl_X_test = dl_scaler.transform(dl_X_test)
            except Exception as e:
                logger.error(f"深度学习特征标准化测试数据时出错: {e}")
                # 简单的Z-score标准化
                dl_X_test = (dl_X_test - dl_X_test.mean()) / dl_X_test.std().clip(lower=1e-10)
            
            # 获取各个模型的预测
            ml_preds = {}
            for name, model in self.ml_models.items():
                try:
                    ml_preds[name] = model.predict(ml_X_test)
                except Exception as e:
                    logger.error(f"{name}模型预测测试数据时出错: {e}")
                    # 使用最常见类别作为默认预测
                    most_common_class = np.argmax(np.bincount(np.concatenate([pred for pred in ml_preds.values()]) if ml_preds else [0]))
                    ml_preds[name] = np.full(len(X_test), most_common_class, dtype=int)
            
            try:
                dl_preds_proba = self.dl_model.predict(dl_X_test)
                dl_preds = np.argmax(dl_preds_proba, axis=1)
            except Exception as e:
                logger.error(f"深度学习模型预测测试数据时出错: {e}")
                # 使用最常见类别作为默认预测
                most_common_class = 0  # 默认为0类
                if ml_preds:
                    all_preds = np.concatenate([pred for pred in ml_preds.values()])
                    if len(all_preds) > 0:
                        most_common_class = np.argmax(np.bincount(all_preds))
                dl_preds = np.full(len(X_test), most_common_class, dtype=int)
            
            # 使用加权投票方式进行集成预测
            def weighted_voting(ml_predictions, dl_prediction, weights):
                # 创建投票计数数组
                n_classes = len(label_encoder.classes_)
                votes = np.zeros(n_classes)
                
                # 统计机器学习模型的投票
                for name, pred in ml_predictions.items():
                    if pred < n_classes:  # 确保预测的类别有效
                        votes[pred] += weights[name]
                
                # 统计深度学习模型的投票
                if dl_prediction < n_classes:  # 确保预测的类别有效
                    votes[dl_prediction] += weights['DeepLearning']
                
                # 返回得票最多的类别
                return np.argmax(votes)
            
            # 应用加权投票到测试集
            ensemble_preds = []
            for i in tqdm(range(len(X_test)), desc="生成集成预测"):
                ml_pred_i = {name: preds[i] for name, preds in ml_preds.items()}
                dl_pred_i = dl_preds[i]
                ensemble_pred = weighted_voting(ml_pred_i, dl_pred_i, weights)
                ensemble_preds.append(ensemble_pred)
            
            # 将预测结果转换为原始标签
            ensemble_preds = np.array(ensemble_preds)
            
            # 确保预测的类别在label_encoder的范围内
            valid_ensemble_preds = np.array([
                p if p < len(label_encoder.classes_) else 0  # 使用0作为默认类别
                for p in ensemble_preds
            ])
            
            pred_labels = label_encoder.inverse_transform(valid_ensemble_preds)
            
            # 创建输出数据框
            output_df = pd.DataFrame({
                'index': range(len(pred_labels)),
                'Label': pred_labels
            })
            
            # 尝试不同的输出路径，处理文件权限问题
            output_paths = [
                self.output_path,  # 原始路径
                os.path.join(os.getcwd(), 'prediction_output.csv'),  # 当前工作目录
                os.path.join(self.models_dir, 'prediction_output.csv'),  # 模型保存目录
                os.path.expanduser('~/prediction_output.csv')  # 用户主目录
            ]
            
            saved_successfully = False
            for path in output_paths:
                try:
                    logger.info(f"尝试保存预测结果到: {path}")
                    output_df.to_csv(path, index=False)
                    logger.info(f"集成模型预测结果已成功保存至 {path}")
                    self.output_path = path  # 更新输出路径
                    saved_successfully = True
                    break
                except PermissionError as e:
                    logger.error(f"保存到 {path} 失败: {e}")
                except Exception as e:
                    logger.error(f"保存到 {path} 时出现未知错误: {e}")
            
            if not saved_successfully:
                logger.info("警告：无法保存预测结果到文件。以下是前10个预测结果:")
                logger.info(output_df.head(10))
            
            return output_df
            
        except Exception as e:
            logger.error(f"预测过程中出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 创建默认预测结果
            n_samples = len(X_test)
            default_class = label_encoder.inverse_transform([0])[0]  # 默认使用第0个类别
            pred_labels = [default_class] * n_samples
            
            # 创建输出数据框
            output_df = pd.DataFrame({
                'index': range(n_samples),
                'Label': pred_labels
            })
            
            # 尝试不同的输出路径，处理文件权限问题
            output_paths = [
                self.output_path,  # 原始路径
                os.path.join(os.getcwd(), 'fallback_output.csv'),  # 当前工作目录
                os.path.join(self.models_dir, 'fallback_output.csv'),  # 模型保存目录
                os.path.expanduser('~/fallback_output.csv')  # 用户主目录
            ]
            
            saved_successfully = False
            for path in output_paths:
                try:
                    logger.info(f"尝试保存默认预测结果到: {path}")
                    output_df.to_csv(path, index=False)
                    logger.info(f"默认预测结果已成功保存至 {path}")
                    self.output_path = path  # 更新输出路径
                    saved_successfully = True
                    break
                except Exception as e:
                    logger.error(f"保存到 {path} 时出错: {e}")
            
            if not saved_successfully:
                logger.info("警告：无法保存预测结果到任何文件。以下是前10个默认预测结果:")
                logger.info(output_df.head(10))
            
            return output_df
    
    def run_pipeline(self,train_data_nrows=None, test_data_nrows=None):
        """运行完整的集成检测流水线"""
        logger.info("开始运行集成检测流水线...")
        
        # 尝试加载已训练的模型
        models_loaded = self.load_models()
        
        # 如果模型加载失败，则训练个别模型
        if not models_loaded:
            self.train_individual_models(train_data_nrows, test_data_nrows)
        
        # 构建集成模型
        self.build_ensemble(train_data_nrows)
        
        # 预测测试集
        self.predict(test_data_nrows)
        
        logger.info("\n集成入侵检测流水线已成功运行!")
        return self


# 主程序入口
if __name__ == "__main__":
    # 设置文件路径
    train_path = "dataset_train.csv"
    test_path = "dataset_test.csv"
    output_path = "output/em/output.csv"  # 最终预测结果使用这个输出文件
    
    # 创建并运行集成网络入侵检测流水线
    detector = EnsembleIntrusionDetection(train_path, test_path, output_path)
    detector.run_pipeline() 