"""Machine learning based shot classifier"""

import numpy as np
from typing import Tuple, Optional
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.utils.registry import SHOT_CLASSIFIERS


@SHOT_CLASSIFIERS.register_module(name='classifier')
class MLShotClassifier:
    """Machine learning based shot type classifier"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize classifier
        
        Args:
            model_path: Model file path
        """
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize a simple model (needs training)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.is_trained = False
    
    def extract_features(self, ball_velocity: float,
                        ball_direction: Tuple[float, float, float],
                        ball_position: Tuple[float, float, float],
                        hitter: Optional[int] = None,
                        player_keypoints: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract features
        
        Args:
            ball_velocity: Ball velocity
            ball_direction: Direction vector
            ball_position: Position
            hitter: Hitter
            player_keypoints: Keypoints
            
        Returns:
            Feature vector
        """
        dx, dy, dz = ball_direction
        
        # Basic features
        features = [
            ball_velocity,
            dx, dy, dz,
            ball_position[0], ball_position[1], ball_position[2],
            np.sqrt(dx**2 + dy**2),  # Horizontal velocity
            np.arctan2(dz, np.sqrt(dx**2 + dy**2)),  # Angle
        ]
        
        # Add hitter information
        if hitter is not None:
            features.append(hitter)
        else:
            features.append(-1)
        
        # Add player pose features (if available)
        if player_keypoints is not None:
            # Wrist position
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
        Classify shot type
        
        Args:
            ball_velocity: Ball velocity
            ball_direction: Direction vector
            ball_position: Position
            hitter: Hitter
            player_keypoints: Keypoints
            
        Returns:
            (Shot type category, confidence)
        """
        if self.model is None or not self.is_trained:
            # If model not trained, return default value
            return ('unknown', 0.0)
        
        # Extract features
        features = self.extract_features(
            ball_velocity, ball_direction, ball_position,
            hitter, player_keypoints
        )
        
        # Normalize
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        return (prediction, confidence)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train model
        
        Args:
            X: Feature matrix
            y: Labels
        """
        # Normalize
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def save_model(self, path: str):
        """
        Save model
        
        Args:
            path: Save path
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
        Load model
        
        Args:
            path: Model path
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data.get('is_trained', True)
