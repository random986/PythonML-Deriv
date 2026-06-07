import os
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

class GatekeeperBrain:
    def __init__(self, name: str, filepath: str = None):
        self.name = name
        self.filepath = filepath or f"{name.lower()}_brain.joblib"
        
        # SGD primary probability engine
        self.sgd = SGDClassifier(
            loss='modified_huber',
            learning_rate='invscaling',
            eta0=0.01,
            power_t=0.25,
            warm_start=True
        )
        # Hard boundary classifier (PA-equivalent via SGD hinge)
        self.pa = SGDClassifier(
            loss='hinge',
            learning_rate='optimal',
            warm_start=True
        )
        
        self.is_fitted = False
        self.update_count = 0
        
        # Warm-up phase: during the first N updates, use a low threshold
        # so the model can actually see real trade outcomes and calibrate.
        self.WARMUP_UPDATES = 50
        
        self.load()

    def get_verdict(self, feature_vector: list[float], global_context: float):
        if not self.is_fitted:
            # Not fitted at all — always execute so we can gather training data
            return "SAFE_EXECUTE", 0.60
            
        # Merge local features with global shadow-prophet context
        combined_X = np.array([np.append(feature_vector, global_context)])
        
        # Calculate Conviction: probability from SGD
        proba = self.sgd.predict_proba(combined_X)
        conf = float(proba[0][1])  # probability of class 1 (win)
        
        # During warm-up (< WARMUP_UPDATES), ALWAYS execute — never block.
        # The model needs real trade outcomes to calibrate. Blocking defeats the purpose.
        if self.update_count < self.WARMUP_UPDATES:
            return "SAFE_EXECUTE", max(conf, 0.55)
        
        # The user requested Probabilistic Entry Scaling instead of BLOCKing.
        # We simply return the raw confidence (Quality Score).
        # The trading_bot will decide when this score is high enough to enter.
        is_safe = self.pa.predict(combined_X)[0] == 1
        
        # We can slightly boost the score if the PA algorithm agrees it's safe
        quality_score = conf
        if is_safe:
            quality_score += 0.05
            
        return "EVALUATE", quality_score

    def update(self, feature_vector: list[float], global_context: float, won: bool):
        combined_X = np.array([np.append(feature_vector, global_context)])
        y = np.array([1 if won else 0])
        
        self.sgd.partial_fit(combined_X, y, classes=np.array([0, 1]))
        self.pa.partial_fit(combined_X, y, classes=np.array([0, 1]))
        self.is_fitted = True
        self.update_count += 1
        
        if self.update_count == self.WARMUP_UPDATES:
            print(f"[BRAIN] {self.name} warm-up complete ({self.WARMUP_UPDATES} trades). Strict gating now active.")

    def save(self):
        try:
            joblib.dump({
                "sgd": self.sgd,
                "pa": self.pa,
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
                self.pa = data["pa"]
                self.is_fitted = data["is_fitted"]
                self.update_count = data.get("update_count", 0)
                print(f"[BRAIN] {self.name} loaded — {self.update_count} prior updates.")
            except Exception as e:
                print(f"[BRAIN] Failed to load {self.name}, starting fresh: {e}")
        else:
            print(f"[BRAIN] {self.name} no existing memory found, starting fresh.")
