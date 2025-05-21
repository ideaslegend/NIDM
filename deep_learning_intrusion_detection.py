#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于深度学习的网络攻击多分类识别模型
用于识别DDoS、密码攻击、后门、扫描、XSS、注入、MITM、勒索软件等网络攻击类型
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import warnings
import joblib
from tqdm import tqdm
import logging

os.makedirs('log', exist_ok=True)
os.makedirs('output/dl', exist_ok=True)
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/深度学习模型训练日志.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('深度学习训练日志')

class LoggerCallback(tf.keras.callbacks.Callback):
    """自定义日志回调类，用于记录每个epoch的训练信息"""
    def __init__(self, log):
        super().__init__()
        self.logger = log

    def on_epoch_end(self, epoch, logs=None):
        """在每个epoch结束时记录日志"""
        if logs is not None:
            log_msg = f"Epoch {epoch + 1}: "
            log_msg += ", ".join([f"{key}={value:.4f}" for key, value in logs.items()])
            self.logger.info(log_msg)

# 忽略警告，使输出更加整洁
warnings.filterwarnings("ignore")

# 设置随机种子以确保结果可重现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
 # 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class DeepLearningIntrusionDetection:
    """基于深度学习的网络入侵检测类"""
    
    def __init__(self, train_path, test_path, output_path="output/dl/output"):
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
        self.n_classes = None
        self.selected_features = None
        self.history = None
    
    def load_data(self,train_data_nrows=None, test_data_nrows=None):
        """加载训练和测试数据"""
        logger.info("正在加载数据...")
        
         # 加载训练集
        self.train_data = pd.read_csv(self.train_path,nrows=train_data_nrows)
       
        # 加载测试集
        self.test_data = pd.read_csv(self.test_path,nrows=test_data_nrows)
  
        logger.info(f"深度学习训练集大小: {self.train_data.shape}")
        logger.info(f"深度学习测试集大小: {self.test_data.shape}")
        
        # 检查是否存在标签列
        if 'Label' not in self.train_data.columns:
            raise ValueError("训练数据中缺少'Label'列")
            
        # 显示数据类型分布
        logger.info("\n训练集攻击类型分布:")
        logger.info(self.train_data['Label'].value_counts())
        
        return self
    
    def preprocess_data(self):
        """数据预处理：清洗数据，处理缺失值和异常值"""
        logger.info("\n进行数据预处理...")
        
        # 1. 处理缺失值
        numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.train_data[col].isnull().sum() > 0:
                median_val = self.train_data[col].median()
                self.train_data[col].fillna(median_val, inplace=True)
                if col in self.test_data.columns:
                    self.test_data[col].fillna(median_val, inplace=True)
        
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Label' and self.train_data[col].isnull().sum() > 0:
                mode_val = self.train_data[col].mode()[0]
                self.train_data[col].fillna(mode_val, inplace=True)
                if col in self.test_data.columns:
                    self.test_data[col].fillna(mode_val, inplace=True)
        
        # 2. 处理异常值 (使用IQR方法)
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
            self.n_classes = len(self.label_encoder.classes_)
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
        """特征工程：特征选择和创建新特征"""
        logger.info("\n进行特征工程...")
        
        # 1. 分离特征和目标变量
        if 'Label' in self.train_data.columns:
            X = self.train_data.drop(['Label', 'Label_encoded', 'Timestamp'], axis=1, errors='ignore')
            y = self.train_data['Label_encoded']
            
            # 检查每个类的样本数量
            class_counts = np.bincount(y)
            min_samples = np.min(class_counts[class_counts > 0])
            logger.info(f"最小类别的样本数: {min_samples}")
            logger.info(f"类别分布: {class_counts}")
            
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
            # 使用F统计量进行特征选择
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
                    logger.info(f"过采样后的类别分布: {np.bincount(self.y_train)}")
                except ValueError as e:
                    logger.error(f"SMOTE过采样失败: {e}")
                    logger.error("跳过SMOTE过采样步骤，继续使用原始数据")
            else:
                logger.error("由于类别样本不足，跳过SMOTE过采样步骤")
            
            # 6. 将标签转换为one-hot编码，以便用于深度学习模型
            # 识别所有可能的类别
            all_classes = np.unique(np.concatenate([self.y_train, self.y_val]))
            max_class_id = max(all_classes)
            logger.info(f"数据中出现的最大类别ID: {max_class_id}")
            
            # 确保n_classes大于等于最大类别ID+1
            self.n_classes = max(self.n_classes, max_class_id + 1)
            logger.info(f"使用的类别数量: {self.n_classes}")
            
            # 转换为one-hot编码
            self.y_train_categorical = to_categorical(self.y_train, num_classes=self.n_classes)
            self.y_val_categorical = to_categorical(self.y_val, num_classes=self.n_classes)
            
            # 保存标准化器和特征选择器，用于后续测试数据处理
            joblib.dump(self.scaler, os.path.join(self.models_dir, 'dl_scaler.pkl'))
            joblib.dump(self.selected_features, os.path.join(self.models_dir, 'dl_selected_features.pkl'))
            joblib.dump(self.label_encoder, os.path.join(self.models_dir, 'dl_label_encoder.pkl'))
            
        return self
    
    def build_mlp_model(self, input_dim, hp=None):
        """构建支持超参数调优的多层感知器模型"""
        logger.info("构建MLP模型...")
        
        if hp is None:
            # 默认参数配置
            logger.info("使用MLP默认参数配置...")
            units1 = 256
            units2 = 128
            units3 = 64
            dropout_rate = 0.3
        else:
            # 超参数搜索空间    
            logger.info("使用MLP超参数搜索空间...")
            units1 = hp.Int('mlp_units1', min_value=128, max_value=512, step=128)
            units2 = hp.Int('mlp_units2', min_value=64, max_value=256, step=64)
            units3 = hp.Int('mlp_units3', min_value=32, max_value=128, step=32)
            dropout_rate = hp.Float('mlp_dropout', min_value=0.2, max_value=0.4, step=0.1)
        
        model = Sequential([
            Dense(units1, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(units2, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(units3, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(self.n_classes, activation='softmax')
        ])
        
        # 动态学习率
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log') if hp else 1e-3
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        return model
    
    def build_deep_model(self, input_dim, hp=None):
        """构建支持超参数调优的深层神经网络模型"""
        logger.info("构建深层神经网络模型...")
        
        inputs = Input(shape=(input_dim,))
        
        if hp is None:
            # 默认参数配置
            logger.info("使用深度学习默认参数配置...")
            units_list = [512, 256, 128, 64]
            dropout_rate = 0.3
        else:
            # 超参数搜索空间
            logger.info("使用深度学习超参数搜索空间...")
            num_layers = hp.Int('deep_layers', min_value=3, max_value=5)
            units_list = [hp.Int(f'deep_units_{i}', min_value=64, max_value=512, step=64) for i in range(num_layers)]
            dropout_rate = hp.Float('deep_dropout', min_value=0.2, max_value=0.4, step=0.1)
        
        x = inputs
        for i, units in enumerate(units_list):
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        
        outputs = Dense(self.n_classes, activation='softmax')(x)
        
        # 动态学习率
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log') if hp else 1e-3
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        return model



    def tune_hyperparameters(self, model_type='mlp', max_trials=10, epochs=10):
        """使用Keras Tuner进行超参数调优"""
        logger.info("\n开始超参数调优...")
        input_dim = self.X_train.shape[1]
        
        # 定义调优器
        # 根据模型类型选择构建函数
        if model_type == 'mlp':
            build_func = lambda hp: self.build_mlp_model(input_dim, hp)
        elif model_type == 'deep':
            build_func = lambda hp: self.build_deep_model(input_dim, hp)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}，请选择'mlp'或'deep'")
        
        tuner = kt.RandomSearch(
            build_func,
            objective='val_accuracy',
            max_trials=max_trials,
            directory=self.models_dir,
            project_name=f'dl_hyperparameter_tuning_{model_type}'
        )
        
        # 搜索超参数
        tuner.search(
            self.X_train, self.y_train_categorical,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val_categorical),
            callbacks=[EarlyStopping(monitor='val_loss', patience=3)]
        )
        
        # 获取最佳超参数
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.best_hyperparameters = best_hps.values
        logger.info("最佳超参数:", self.best_hyperparameters)

        
        return best_hps

    # 训练轮次epochs = 15轮 batch_size = 1024
    def train_models(self, epochs=10, batch_size=1024, use_tuning=True):
        """训练深度学习模型（支持超参数调优）"""

        logger.info("\n开始训练深度学习模型...")
        
        # 获取输入维度
        input_dim = self.X_train.shape[1]
        
        # 执行超参数调优
        mlp_best_hps = None
        deep_best_hps = None
        if use_tuning:
            mlp_best_hps = self.tune_hyperparameters(model_type='mlp')
            deep_best_hps = self.tune_hyperparameters(model_type='deep')
        
        # 创建回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, 'best_dl_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5),
            LoggerCallback(logger)
        ]
        
        # 构建并训练MLP模型
        mlp_model = self.build_mlp_model(input_dim, mlp_best_hps) if use_tuning else self.build_mlp_model(input_dim)
        mlp_history = mlp_model.fit(
            self.X_train, self.y_train_categorical,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val_categorical),
            callbacks=callbacks,
            verbose=1
        )
        
        # 构建并训练深层神经网络模型
        deep_model = self.build_deep_model(input_dim, deep_best_hps) if use_tuning else self.build_deep_model(input_dim)
        deep_history = deep_model.fit(
            self.X_train, self.y_train_categorical,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val_categorical),
            callbacks=callbacks,
            verbose=1
        )
        
        # 评估模型
        mlp_val_loss, mlp_val_acc = mlp_model.evaluate(self.X_val, self.y_val_categorical)
        deep_val_loss, deep_val_acc = deep_model.evaluate(self.X_val, self.y_val_categorical)
        
        logger.info(f"MLP模型验证集准确率: {mlp_val_acc:.4f}")
        logger.info(f"深层神经网络模型验证集准确率: {deep_val_acc:.4f}")
        
        # 选择最佳模型
        if deep_val_acc > mlp_val_acc:
            self.best_model = deep_model
            self.history = deep_history
            logger.info("深层神经网络模型表现更好，已选为最佳模型")
        else:
            self.best_model = mlp_model
            self.history = mlp_history
            logger.info("MLP模型表现更好,已选为最佳模型")
        
        # 绘制训练历史
        self.plot_training_history()
        
        # 评估最佳模型
        self.evaluate_model()
        
        # 保存最佳模型
        self.best_model.save(os.path.join(self.models_dir, 'best_dl_model.h5'))
        
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
    
    def evaluate_model(self):
        """评估最佳模型"""
        if self.best_model is not None:
            # 预测验证集
            y_pred_proba = self.best_model.predict(self.X_val)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # 计算准确率
            accuracy = accuracy_score(self.y_val, y_pred)
            logger.info(f"\n最佳模型在验证集上的准确率: {accuracy:.4f}")
            
            # 获取实际出现在验证集中的类别
            unique_classes = np.unique(np.concatenate([y_pred, self.y_val]))
            logger.info(f"验证集中的实际类别: {unique_classes}")
            logger.info(f"验证集标签统计: {np.bincount(self.y_val)}")
            logger.info(f"预测标签统计: {np.bincount(y_pred)}")
            
            # 使用labels参数指定实际出现的类别
            try:
                # 获取实际出现在验证集中的类别名称
                target_names = [self.label_encoder.classes_[i] for i in unique_classes if i < len(self.label_encoder.classes_)]
                
                # 生成分类报告
                logger.info("\n分类报告:")
                report = classification_report(
                    self.y_val, y_pred, 
                    labels=unique_classes,
                    target_names=target_names
                )
                logger.info(report)
            except Exception as e:
                logger.error(f"生成分类报告时出错: {e}")
                logger.error("使用基础分类报告:")
                report = classification_report(self.y_val, y_pred)
                logger.error(report)
            
            # 绘制混淆矩阵
            try:
                plt.figure(figsize=(12, 10))
                cm = confusion_matrix(self.y_val, y_pred, labels=unique_classes)
                xticklabels = [self.label_encoder.classes_[i] if i < len(self.label_encoder.classes_) else f"Unknown_{i}" 
                              for i in unique_classes]
                yticklabels = xticklabels.copy()
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=xticklabels,
                            yticklabels=yticklabels)
                plt.title('深度学习模型混淆矩阵',fontsize=14)
                plt.ylabel('真实标签')
                plt.xlabel('预测标签')
                plt.xticks(rotation=45)
                plt.yticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(self.models_dir, 'dl_confusion_matrix.png'))
            except Exception as e:
                logger.error(f"绘制混淆矩阵时出错: {e}")
                logger.error("跳过混淆矩阵绘制")
    
    def predict(self):
        """使用最佳模型预测测试集"""
        logger.info("\n使用深度学习模型预测测试集...")
        
        # 准备测试数据
        X_test = self.test_data.copy()
        
        # 删除不必要的列（如果存在）
        X_test = X_test.drop(['Timestamp'], axis=1, errors='ignore')
        
        # 确保测试集只包含选定的特征
        X_test = X_test[self.selected_features]
        
        # 应用相同的标准化
        X_test = self.scaler.transform(X_test)
        
        # 预测
        test_pred_proba = self.best_model.predict(X_test)
        test_pred = np.argmax(test_pred_proba, axis=1)
        
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
    
    def run_pipeline(self,train_data_nrows=None, test_data_nrows=None):
        """运行完整的数据处理和模型训练管道"""
        (self.load_data(train_data_nrows, test_data_nrows)
             .preprocess_data()
             .feature_engineering()
             .train_models()
             .predict())
        
        logger.info("\n完整的深度学习入侵检测流水线已成功运行!")
        return self


# 主程序入口
if __name__ == "__main__":
    # 设置文件路径
    train_path = "dataset_train.csv"
    test_path = "dataset_test.csv"
    output_path = "output/dl/output.csv"
    
    # 创建并运行深度学习网络入侵检测流水线
    detector = DeepLearningIntrusionDetection(train_path, test_path, output_path)
    detector.run_pipeline()