import os
import joblib
import numpy as np
from collections import deque
from sklearn.ensemble import RandomForestClassifier

class GatekeeperBrain:
    """
    The ML brain that answers ONE question per tick:
    "Given the current market conditions AND my recovery state,
     is this market a good place to enter right now?"

    It does NOT predict over/under. It predicts ENTRY QUALITY.

    Label = 1  → entering here led to recovery (the side that needed to win, won)
    Label = 0  → entering here deepened the loss (the side that needed to win, lost)

    The feature vector includes:
      - Market digit patterns (26 features — what the market is doing)
      - Recovery context (3 features — what the bot's current state is)
    This means the ML itself learns: "JD75 when I'm at step 3 in under-recovery
    and the over_ratio is 0.65 is a BAD entry" — without any hardcoded rule.
    """

    def __init__(self, name: str, filepath: str = None):
        self.name = name
        self.filepath = filepath or f"{name.lower()}_brain.joblib"

        # Sliding window: ~200 trades of pure market memory
        self.MAX_MEMORY = 200
        self.memory = deque(maxlen=self.MAX_MEMORY)

        self.is_fitted = False
        self.update_count = 0
        self.WARMUP_UPDATES = 50

        self.model = self._create_model()
        self.load()

    def _create_model(self):
        """100 decision trees. Each tree learns a different slice of the
        (market_state + recovery_context) space. The forest votes democratically."""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=3,   # Prevents over-fitting on tiny samples
            class_weight='balanced',
            n_jobs=-1
        )

    def _build_input(self, market_features: list, recovery_context: list) -> np.ndarray:
        """Combine market features + recovery context into one input vector."""
        return np.array(market_features + recovery_context, dtype=float)

    def get_entry_score(self, market_features: list, recovery_context: list) -> float:
        """
        Returns a float [0.0, 1.0] representing:
        How confident the ML is that entering this market NOW is a good decision,
        given the current recovery state.

        Higher = the forest says this looks like conditions that led to recovery.
        Lower  = the forest says this looks like conditions that deepened losses.

        During warm-up, returns 0.5 (neutral — gather data without bias).
        """
        if not self.is_fitted or len(self.memory) < self.WARMUP_UPDATES:
            return 0.5

        X = self._build_input(market_features, recovery_context).reshape(1, -1)
        proba = self.model.predict_proba(X)
        # Class 1 = "this entry led to recovery"
        score = float(proba[0][1])
        return max(0.0, min(1.0, score))

    def update(self,
               market_features: list,
               recovery_context: list,
               recovery_succeeded: bool,
               weight: float = 1.0):
        """
        Teach the ML from the outcome of a real trade.

        recovery_succeeded = True  → the side that needed to win, won
        recovery_succeeded = False → it lost, deepening the martingale step

        weight = the stake at the time of the trade. Larger stakes (deeper recovery
        steps) naturally carry higher importance — no artificial multiplier needed.
        """
        combined = self._build_input(market_features, recovery_context)
        expected_dim = len(combined)
        self.memory.append({
            "features": combined,
            "label": 1 if recovery_succeeded else 0,
            "weight": weight
        })
        self.update_count += 1

        # Purge any stale entries with mismatched feature dimensions
        # (can happen after feature vector changes between restarts)
        valid = [m for m in self.memory if len(m["features"]) == expected_dim]
        if len(valid) < len(self.memory):
            self.memory.clear()
            self.memory.extend(valid)

        if len(self.memory) < 2:
            return

        X = np.array([m["features"] for m in self.memory])
        y = np.array([m["label"] for m in self.memory])
        w = np.array([m["weight"] for m in self.memory])

        # Only train periodically to save CPU and event loop blocking!
        if len(np.unique(y)) > 1:
            if not self.is_fitted or self.update_count % 20 == 0:
                self.model = self._create_model()
                self.model.fit(X, y, sample_weight=w)
                self.is_fitted = True
        else:
            # Not enough diversity yet — keep waiting
            self.is_fitted = False

        if self.update_count == self.WARMUP_UPDATES:
            print(f"[BRAIN] {self.name} warm-up complete. ML is now thinking from {self.MAX_MEMORY}-trade memory.")

    def save(self):
        try:
            joblib.dump({
                "memory": self.memory,
                "update_count": self.update_count
            }, self.filepath)
        except Exception as e:
            print(f"[BRAIN] Failed to save {self.name}: {e}")

    def load(self):
        if os.path.exists(self.filepath):
            try:
                data = joblib.load(self.filepath)
                if "memory" in data:
                    loaded_mem = data["memory"]
                    if len(loaded_mem) > 0:
                        first_features = loaded_mem[0]["features"]
                        if len(first_features) != 32:
                            print(f"[BRAIN] {self.name} feature dimension mismatch ({len(first_features)} vs 32). Resetting memory.")
                            return
                    self.memory = loaded_mem
                    self.update_count = data.get("update_count", len(self.memory))
                    if len(self.memory) > 0:
                        X = np.array([m["features"] for m in self.memory])
                        y = np.array([m["label"] for m in self.memory])
                        w = np.array([m["weight"] for m in self.memory])
                        if len(np.unique(y)) > 1:
                            self.model.fit(X, y, sample_weight=w)
                            self.is_fitted = True
                    print(f"[BRAIN] {self.name} loaded — {len(self.memory)} trades in memory.")
                else:
                    print(f"[BRAIN] {self.name} old format detected, starting fresh.")
            except Exception as e:
                print(f"[BRAIN] Load error for {self.name}: {e}")
