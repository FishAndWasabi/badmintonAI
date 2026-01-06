"""Machine learning based shot classifier"""

import numpy as np
from typing import Tuple, Optional
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class MLShotClassifier:
    """基于机器学习的球种分类器"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化分类器
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # 初始化一个简单的模型（需要训练）
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.is_trained = False
    
    def extract_features(self, ball_velocity: float,
                        ball_direction: Tuple[float, float, float],
                        ball_position: Tuple[float, float, float],
                        hitter: Optional[int] = None,
                        player_keypoints: Optional[np.ndarray] = None) -> np.ndarray:
        """
        提取特征
        
        Args:
            ball_velocity: 球速
            ball_direction: 方向向量
            ball_position: 位置
            hitter: 击球方
            player_keypoints: 关键点
            
        Returns:
            特征向量
        """
        dx, dy, dz = ball_direction
        
        # 基础特征
        features = [
            ball_velocity,
            dx, dy, dz,
            ball_position[0], ball_position[1], ball_position[2],
            np.sqrt(dx**2 + dy**2),  # 水平速度
            np.arctan2(dz, np.sqrt(dx**2 + dy**2)),  # 角度
        ]
        
        # 添加击球方信息
        if hitter is not None:
            features.append(hitter)
        else:
            features.append(-1)
        
        # 添加玩家姿态特征（如果可用）
        if player_keypoints is not None:
            # 手腕位置
            if len(player_keypoints) > 10:
                wrist_x = player_keypoints[9, 0] if player_keypoints[9, 2] > 0.5 else 0
                wrist_y = player_keypoints[9, 1] if player_keypoints[9, 2] > 0.5 else 0
                features.extend([wrist_x, wrist_y])
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
        
        return np.array(features)
    
    def classify(self, ball_velocity: float,
                ball_direction: Tuple[float, float, float],
                ball_position: Tuple[float, float, float],
                hitter: Optional[int] = None,
                player_keypoints: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """
        分类球种
        
        Args:
            ball_velocity: 球速
            ball_direction: 方向向量
            ball_position: 位置
            hitter: 击球方
            player_keypoints: 关键点
            
        Returns:
            (球种类别, 置信度)
        """
        if self.model is None or not self.is_trained:
            # 如果模型未训练，返回默认值
            return ('unknown', 0.0)
        
        # 提取特征
        features = self.extract_features(
            ball_velocity, ball_direction, ball_position,
            hitter, player_keypoints
        )
        
        # 标准化
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # 预测
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        return (prediction, confidence)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签
        """
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data.get('is_trained', True)
