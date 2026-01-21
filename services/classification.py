"""
Baby Cry Classification Service
Handles model training, prediction, and continuous improvement for baby cry classification.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib

from services.audio import extract_features, convert_audio_format

# Model storage directory
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Default cry types
DEFAULT_CRY_TYPES = [
    "hungry",
    "tired",
    "discomfort",
    "pain",
    "attention",
    "diaper_change",
    "overstimulated",
    "colic"
]


class BabyCryClassifier:
    """Baby cry classification model with training and prediction capabilities."""
    
    def __init__(self, model_type: str = "random_forest", cry_types: Optional[List[str]] = None):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of model ('random_forest' or 'gradient_boosting')
            cry_types: List of cry type labels
        """
        self.model_type = model_type
        self.cry_types = cry_types or DEFAULT_CRY_TYPES
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.training_metrics = {}
        self.version = "1.0.0"
        
    def _create_model(self):
        """Create the ML model based on model_type."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _extract_feature_vector(self, features: Dict) -> np.ndarray:
        """
        Extract feature vector from audio features dictionary.
        
        Args:
            features: Dictionary containing extracted audio features
        
        Returns:
            Feature vector as numpy array
        """
        feature_list = []
        
        # MFCC features (mean values)
        if "mfcc" in features and "mfcc_mean" in features["mfcc"]:
            feature_list.extend(features["mfcc"]["mfcc_mean"])
        
        # Pitch and frequency features
        if "pitch_frequency" in features:
            pf = features["pitch_frequency"]
            feature_list.extend([
                pf.get("pitch_mean", 0),
                pf.get("pitch_std", 0),
                pf.get("dominant_frequency", 0),
                pf.get("spectral_centroid_mean", 0),
                pf.get("zero_crossing_rate_mean", 0)
            ])
        
        # Duration features
        if "duration" in features:
            d = features["duration"]
            feature_list.extend([
                d.get("total_duration_seconds", 0),
                d.get("actual_audio_duration_seconds", 0),
                d.get("silence_percentage", 0)
            ])
        
        # Spectrogram statistics
        if "spectrogram" in features:
            sp = features["spectrogram"]
            feature_list.extend([
                sp.get("magnitude_mean", 0),
                sp.get("magnitude_max", 0),
                sp.get("magnitude_min", 0)
            ])
        
        return np.array(feature_list)
    
    def train(
        self,
        training_data: List[Dict],
        test_size: float = 0.2,
        validation_size: float = 0.1,
        retrain: bool = False
    ) -> Dict:
        """
        Train the classification model.
        
        Args:
            training_data: List of dictionaries with 'features' and 'label' keys
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            retrain: Whether to retrain existing model (for continuous improvement)
        
        Returns:
            Dictionary containing training metrics and results
        """
        if not training_data:
            raise ValueError("Training data cannot be empty")
        
        # Extract features and labels
        X = []
        y = []
        
        for item in training_data:
            if "features" not in item or "label" not in item:
                continue
            
            feature_vector = self._extract_feature_vector(item["features"])
            if len(feature_vector) == 0:
                continue
            
            X.append(feature_vector)
            y.append(item["label"])
        
        if len(X) == 0:
            raise ValueError("No valid training samples found")
        
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels based on data (avoid mismatches when dataset has fewer classes)
        labels_in_data = sorted(set(y))
        if self.cry_types:
            labels_in_data = [label for label in self.cry_types if label in labels_in_data]
        self.label_encoder.fit(labels_in_data)
        y_encoded = self.label_encoder.transform(y)
        
        # Store feature names for reference
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        if not retrain or self.model is None:
            self._create_model()
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
        
        # Evaluate on test set
        y_test_pred = self.model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring='accuracy'
        )
        
        # Store metrics
        self.training_metrics = {
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
            "validation_accuracy": float(val_accuracy),
            "validation_precision": float(val_precision),
            "validation_recall": float(val_recall),
            "validation_f1": float(val_f1),
            "test_accuracy": float(test_accuracy),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1": float(test_f1),
            "cross_val_mean": float(cv_scores.mean()),
            "cross_val_std": float(cv_scores.std()),
            "num_features": int(X.shape[1]),
            "num_classes": len(self.label_encoder.classes_)
        }
        
        return {
            "status": "success",
            "metrics": self.training_metrics,
            "classification_report": classification_report(
                y_test, y_test_pred, 
                target_names=self.label_encoder.classes_,
                output_dict=True,
                zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist()
        }
    
    def predict(self, features: Dict) -> Dict:
        """
        Predict cry type from extracted features.
        
        Args:
            features: Dictionary containing extracted audio features
        
        Returns:
            Dictionary with prediction, confidence scores, and metadata
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Extract feature vector
        feature_vector = self._extract_feature_vector(features)
        
        if len(feature_vector) == 0:
            raise ValueError("No features extracted from audio")
        
        # Reshape for single prediction
        X = feature_vector.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction_encoded = self.model.predict(X_scaled)[0]
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get prediction probabilities (confidence scores)
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Create confidence scores dictionary
        confidence_scores = {}
        for i, cry_type in enumerate(self.label_encoder.classes_):
            confidence_scores[cry_type] = float(probabilities[i])
        
        # Get top prediction confidence
        max_confidence = float(np.max(probabilities))
        
        return {
            "predicted_cry_type": prediction,
            "confidence_score": max_confidence,
            "confidence_scores": confidence_scores,
            "all_predictions": [
                {
                    "cry_type": cry_type,
                    "confidence": confidence_scores[cry_type]
                }
                for cry_type in sorted(
                    confidence_scores.keys(),
                    key=lambda x: confidence_scores[x],
                    reverse=True
                )
            ]
        }
    
    def save(self, model_name: str, version: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            model_name: Name of the model
            version: Version string (optional)
        
        Returns:
            Path to saved model file
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        version = version or self.version
        model_path = MODELS_DIR / f"{model_name}_v{version}.pkl"
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "cry_types": self.cry_types,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "version": version,
            "saved_at": datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        return str(model_path)
    
    @staticmethod
    def load(model_path: str) -> 'BabyCryClassifier':
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the model file
        
        Returns:
            Loaded BabyCryClassifier instance
        """
        model_data = joblib.load(model_path)
        
        classifier = BabyCryClassifier(
            model_type=model_data["model_type"],
            cry_types=model_data["cry_types"]
        )
        classifier.model = model_data["model"]
        classifier.scaler = model_data["scaler"]
        classifier.label_encoder = model_data["label_encoder"]
        classifier.feature_names = model_data["feature_names"]
        classifier.training_metrics = model_data.get("training_metrics", {})
        classifier.version = model_data.get("version", "1.0.0")
        
        return classifier
    
    def improve_with_new_data(
        self,
        new_training_data: List[Dict],
        test_size: float = 0.2
    ) -> Dict:
        """
        Continuously improve model with new training data.
        This implements incremental learning by retraining on combined data.
        
        Args:
            new_training_data: New training samples with 'features' and 'label' keys
            test_size: Proportion of new data for testing
        
        Returns:
            Dictionary containing improvement metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        # For continuous improvement, we would ideally:
        # 1. Load existing training data (if available)
        # 2. Combine with new data
        # 3. Retrain model
        
        # For now, we'll retrain on the new data
        # In production, you'd want to persist and load previous training data
        result = self.train(
            training_data=new_training_data,
            test_size=test_size,
            retrain=True
        )
        
        # Compare with previous metrics if available
        improvement = {}
        if self.training_metrics:
            old_accuracy = self.training_metrics.get("test_accuracy", 0)
            new_accuracy = result["metrics"]["test_accuracy"]
            improvement["accuracy_change"] = float(new_accuracy - old_accuracy)
            improvement["improved"] = new_accuracy > old_accuracy
        
        result["improvement"] = improvement
        return result


# Global model instance (can be loaded from disk)
_current_model: Optional[BabyCryClassifier] = None
_current_model_path: Optional[str] = None


def get_model(model_path: Optional[str] = None) -> BabyCryClassifier:
    """
    Get the current model instance, loading from disk if needed.
    
    Args:
        model_path: Optional path to model file
    
    Returns:
        BabyCryClassifier instance
    """
    global _current_model, _current_model_path
    
    if model_path and model_path != _current_model_path:
        _current_model = BabyCryClassifier.load(model_path)
        _current_model_path = model_path
    elif _current_model is None:
        # Try to load default model if exists
        default_model = MODELS_DIR / "baby_cry_classifier_v1.0.0.pkl"
        if default_model.exists():
            _current_model = BabyCryClassifier.load(str(default_model))
            _current_model_path = str(default_model)
        else:
            raise ValueError("No model available. Please train a model first.")
    
    return _current_model


def set_model(classifier: BabyCryClassifier, model_path: Optional[str] = None):
    """
    Set the current model instance.
    
    Args:
        classifier: BabyCryClassifier instance
        model_path: Optional path to model file
    """
    global _current_model, _current_model_path
    _current_model = classifier
    _current_model_path = model_path


def get_model_metadata() -> Dict:
    """
    Return metadata for the currently loaded model.
    """
    global _current_model, _current_model_path
    try:
        model = get_model()
    except ValueError:
        return {
            "available": False,
            "model_path": None,
            "model_type": None,
            "version": None,
            "num_classes": None
        }

    num_classes = None
    try:
        num_classes = len(model.label_encoder.classes_)
    except Exception:
        num_classes = None

    return {
        "available": True,
        "model_path": _current_model_path,
        "model_type": model.model_type,
        "version": model.version,
        "num_classes": num_classes
    }

