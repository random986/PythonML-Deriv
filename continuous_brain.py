import os
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

class GatekeeperBrain:
    def __init__(self, name: str, filepath: str = None):
        self.name = name
        self.filepath = filepath or f"{name.lower()}_brain.joblib"
        
        # SGD primary probability engine
        # Using log_loss for properly calibrated probabilities [0.0, 1.0]
        # Using constant learning rate so the model never stops learning
        self.sgd = SGDClassifier(
            loss='log_loss',
            learning_rate='constant',
            eta0=0.01,
            warm_start=True
        )
        
        self.is_fitted = False
        self.update_count = 0
        
        # Warm-up phase: during the first N updates, the model gathers data.
        self.WARMUP_UPDATES = 50
        
        self.load()

    def get_verdict(self, feature_vector: list[float], global_context: float):
        if not self.is_fitted:
            # Not fitted at all — return neutral 0.50 (coin-flip)
            return "EVALUATE", 0.50
            
        # Merge local features with global shadow-prophet context
        combined_X = np.array([np.append(feature_vector, global_context)])
        
        # Calculate probability from SGD
        proba = self.sgd.predict_proba(combined_X)
        conf = float(proba[0][1])  # probability of class 1 (win)
        
        # HARD CLAMP to [0.0, 1.0] — no more 105% values
        conf = max(0.0, min(1.0, conf))
        
        # During warm-up, return neutral-ish so we gather data without bias
        if self.update_count < self.WARMUP_UPDATES:
            return "EVALUATE", max(conf, 0.51)
        
        # Post warm-up: return the raw learned probability. No boosting, no blocking.
        # The trading_bot uses this to compare across markets.
        return "EVALUATE", conf

    def update(self, feature_vector: list[float], global_context: float, won: bool):
        combined_X = np.array([np.append(feature_vector, global_context)])
        y = np.array([1 if won else 0])
        
        self.sgd.partial_fit(combined_X, y, classes=np.array([0, 1]))
        self.is_fitted = True
        self.update_count += 1
        
        if self.update_count == self.WARMUP_UPDATES:
            print(f"[BRAIN] {self.name} warm-up complete ({self.WARMUP_UPDATES} trades). Full ML active.")

    def save(self):
        try:
            joblib.dump({
                "sgd": self.sgd,
                "is_fitted": self.is_fitted,
                "update_count": self.update_count
            }, self.filepath)
        except Exception as e:
            print(f"[BRAIN] Failed to save {self.name}: {e}")

    def load(self):
        if os.path.exists(self.filepath):
            try:
                data = joblib.load(self.filepath)
                self.sgd = data["sgd"]
                self.is_fitted = data["is_fitted"]
                self.update_count = data.get("update_count", 0)
                print(f"[BRAIN] {self.name} loaded — {self.update_count} prior updates.")
            except Exception as e:
                print(f"[BRAIN] Failed to load {self.name}, starting fresh: {e}")
        else:
            print(f"[BRAIN] {self.name} no existing memory found, starting fresh.")
