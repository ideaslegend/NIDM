#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
网络攻击多分类识别系统 - 主程序
基于机器学习和深度学习方法的网络入侵检测系统
用于识别DDoS、密码攻击、后门、扫描、XSS、注入、MITM、勒索软件等网络攻击类型
"""

import os
import argparse
import pandas as pd
from network_intrusion_detection import NetworkIntrusionDetection,XGBLoggerCallback,LGBLoggerCallback
from deep_learning_intrusion_detection import DeepLearningIntrusionDetection,LoggerCallback
from ensemble_intrusion_detection import EnsembleIntrusionDetection
import time
import datetime
import logging
os.makedirs('log', exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs('output/dl', exist_ok=True)
os.makedirs('output/ml', exist_ok=True)
os.makedirs('output/em', exist_ok=True)
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/训练日志.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='网络攻击多分类识别系统')
    
    parser.add_argument(
        '--train_path', 
        type=str, 
        default='dataset_train.csv',
        help='训练数据集路径 (默认: dataset_train.csv)'
    )
    
    parser.add_argument(
        '--test_path', 
        type=str, 
        default='dataset_test.csv',
        help='测试数据集路径 (默认: dataset_test.csv)'
    )
    
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='output/em/output.csv',
        help='预测结果输出路径 (默认: output/em/output.csv)'
    )
    
    parser.add_argument(
        '--model_type', 
        type=str, 
        choices=['ml', 'dl', 'ensemble', 'all'], 
        default='ensemble',
        help='选择模型类型: ml (机器学习), dl (深度学习), ensemble (集成), all (全部) (默认: ensemble)'
    )

    parser.add_argument(
        '--ml_model_type',
        type=str,
        choices=['all', 'RandomForest', 'LightGBM', 'XGBoost'],
        default='all',
        help='选择模型类型: all (训练模型，使用最佳模型预测结果集), RandomForest模型, LightGBM模型, XGBoost模型 (默认: all,使用三种模型中最佳模型进行结果预测)'
    )
    
    parser.add_argument(
        '--predict_only', 
        action='store_true',
        help='仅使用已训练的模型进行预测，不进行训练'
    )
    
    parser.add_argument(
        '--n_features', 
        type=int, 
        default=40,
        help='特征选择时保留的特征数量 (默认: 40)'
    )
    parser.add_argument(
        '--train_size', 
        type=int, 
        default=None,
        help='选取训练数据集数量 (默认: 40)'
    )
    parser.add_argument(
        '--test_size', 
        type=int, 
        default=None,
        help='选取测试数据集数量 (默认: 40)'
    )
    parser.add_argument(
        '--use_ml_tuning', 
        type=bool, 
        default=False,
        help='是否进行机器学习模型参数调优 (默认: False)'
    )
    
    return parser.parse_args()

def run_ml_model(args):
    """运行机器学习模型流水线"""
    logger.info("执行机器学习模型流水线...")
    
    start_time = time.time()
    
    # 创建并执行机器学习模型流水线
    ml_detector = NetworkIntrusionDetection(
        train_path=args.train_path,
        test_path=args.test_path,
        output_path=args.output_path if args.model_type == 'ml' else 'output/ml/output.csv.csv'
    )
    
    if args.predict_only:
        # 仅执行预测
        ml_detector.load_data(args.train_size,args.test_size)
        ml_detector.preprocess_data()
        # 加载预训练模型和预处理器
        ml_detector.best_model = joblib.load(os.path.join(ml_detector.models_dir, 'best_model.pkl'))
        ml_detector.scaler = joblib.load(os.path.join(ml_detector.models_dir, 'scaler.pkl'))
        ml_detector.selected_features = joblib.load(os.path.join(ml_detector.models_dir, 'selected_features.pkl'))
        ml_detector.label_encoder = joblib.load(os.path.join(ml_detector.models_dir, 'label_encoder.pkl'))
        ml_detector.predict()
    else:
        # 执行完整流水线
        ml_detector.run_pipeline(args.train_size,args.test_size,args.ml_model_type,args.use_ml_tuning)
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"机器学习模型流水线已完成，耗时: {str(datetime.timedelta(seconds=duration))}")
    
    return ml_detector

def run_dl_model(args):
    """运行深度学习模型流水线"""
    logger.info("执行深度学习模型流水线...")
    
    start_time = time.time()
    
    # 创建并执行深度学习模型流水线
    dl_detector = DeepLearningIntrusionDetection(
        train_path=args.train_path,
        test_path=args.test_path,
        output_path=args.output_path if args.model_type == 'dl' else 'output/dl/output.csv'
    )
    
    if args.predict_only:
        # 仅执行预测
        dl_detector.load_data(args.train_size,args.test_size)
        dl_detector.preprocess_data()
        # 加载预训练模型和预处理器
        import tensorflow as tf
        import joblib
        dl_detector.best_model = tf.keras.models.load_model(os.path.join(dl_detector.models_dir, 'best_dl_model.h5'))
        dl_detector.scaler = joblib.load(os.path.join(dl_detector.models_dir, 'dl_scaler.pkl'))
        dl_detector.selected_features = joblib.load(os.path.join(dl_detector.models_dir, 'dl_selected_features.pkl'))
        dl_detector.label_encoder = joblib.load(os.path.join(dl_detector.models_dir, 'dl_label_encoder.pkl'))
        dl_detector.predict()
    else:
        # 执行完整流水线
        dl_detector.run_pipeline(args.train_size,args.test_size)
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"深度学习模型流水线已完成，耗时: {str(datetime.timedelta(seconds=duration))}")
    
    return dl_detector

def run_ensemble_model(args):
    """运行集成模型流水线"""
    logger.info("执行集成模型流水线...")
    
    start_time = time.time()
    
    # 创建并执行集成模型流水线
    ensemble_detector = EnsembleIntrusionDetection(
        train_path=args.train_path,
        test_path=args.test_path,
        output_path=args.output_path
    )
    
    # 执行流水线
    ensemble_detector.run_pipeline(args.train_size,args.test_size)
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"集成模型流水线已完成，耗时: {str(datetime.timedelta(seconds=duration))}")
    
    return ensemble_detector

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 显示执行信息
    logger.info("网络攻击多分类识别系统启动")
    logger.info(f"训练数据路径: {args.train_path}")
    logger.info(f"测试数据路径: {args.test_path}")
    logger.info(f"输出结果路径(默认): {args.output_path}")
    logger.info(f"模型类型: {args.model_type}")
    logger.info(f"仅预测模式: {args.predict_only}")
    
    # 创建模型保存目录
    os.makedirs('saved_models', exist_ok=True)
    
    # 检查数据文件是否存在
    if not os.path.exists(args.train_path) and not args.predict_only:
        logger.error(f"训练数据文件不存在: {args.train_path}")
        return
    
    if not os.path.exists(args.test_path):
        logger.error(f"测试数据文件不存在: {args.test_path}")
        return
    
    try:
        # 根据选择的模型类型执行相应的流水线
        start_time = time.time()
        
        if args.model_type == 'ml' or args.model_type == 'all':
            args.output_path='output/ml/output.csv'
            run_ml_model(args)
        
        if args.model_type == 'dl' or args.model_type == 'all':
            args.output_path = 'output/dl/output.csv'
            run_dl_model(args)
        
        if args.model_type == 'ensemble' or args.model_type == 'all':
            args.output_path = 'output/em/output.csv'
            run_ensemble_model(args)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        logger.info(f"所有处理已完成，总耗时: {str(datetime.timedelta(seconds=total_duration))}")
        logger.info(f"预测结果已保存至: {args.output_path}")
        
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}", exc_info=True)

if __name__ == "__main__":
    import joblib
    main()