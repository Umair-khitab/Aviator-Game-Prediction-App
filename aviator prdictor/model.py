"""
Aviator Game Prediction Model (Upgraded)
=========================================
- VotingRegressor ensemble: RandomForest (0.4) + XGBoost (0.6)
- Sequence-based features from last 10 rounds
- Calibrated crash classifier (LogisticRegression + isotonic)
- Saves models to /models/ as .pkl files
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
    print("[Model] WARNING: xgboost not installed, falling back to RF-only mode")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def build_sequence_features(series: np.ndarray) -> pd.DataFrame:
    """
    Build sequence-based features from a 1-D array of multipliers.
    Uses rolling windows — first 10 rows will be NaN and are dropped.
    """
    df = pd.DataFrame({"target": series})

    df["last_1"] = df["target"].shift(1)
    df["last_3_avg"] = df["target"].shift(1).rolling(3).mean()
    df["last_5_avg"] = df["target"].shift(1).rolling(5).mean()
    df["last_10_avg"] = df["target"].shift(1).rolling(10).mean()
    df["last_10_std"] = df["target"].shift(1).rolling(10).std()

    # streak_low: count of rounds < 2.0 in the previous 5
    df["streak_low"] = (
        (df["target"].shift(1) < 2.0)
        .rolling(5)
        .sum()
        .fillna(0)
        .astype(int)
    )

    # max of the last 5 rounds
    df["max_recent"] = df["target"].shift(1).rolling(5).max()

    # trend: last_1 minus the value 5 rounds ago
    df["trend"] = df["target"].shift(1) - df["target"].shift(5)

    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "last_1",
    "last_3_avg",
    "last_5_avg",
    "last_10_avg",
    "last_10_std",
    "streak_low",
    "max_recent",
    "trend",
]


class AviatorPredictor:
    """ML-based Aviator game multiplier predictor (upgraded)."""

    def __init__(self):
        self.model = None
        self.crash_clf = None
        self.accuracy_score = 0.0
        self.mae = 0.0
        self.rmse = 0.0
        self.crash_auc = 0.0
        self.multipliers = []
        
        # Check if we can load cached models to save 15 mins of training
        if os.path.exists(os.path.join(MODELS_DIR, "model.pkl")) and \
           os.path.exists(os.path.join(MODELS_DIR, "crash_classifier.pkl")):
            try:
                print("[Model] Loading cached models...")
                self.model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
                self.crash_clf = joblib.load(os.path.join(MODELS_DIR, "crash_classifier.pkl"))
                
                # Load raw multipliers for stats
                mult_path = os.path.join(BASE_DIR, "multipliers.csv")
                if os.path.exists(mult_path):
                    mult_df = pd.read_csv(mult_path)
                    self.multipliers = mult_df["Multiplier"].values.tolist()
                print("[Model] Cached models loaded!")
                return
            except Exception as e:
                print(f"[Model] Cache load failed: {e}. Retraining...")

        self._load_and_train()

    def _load_and_train(self):
        """Load dataset, engineer features, train ensemble + crash classifier."""

        # ---------- Load data ----------
        dataset_path = os.path.join(BASE_DIR, "aviator_dataset_clean.csv")
        raw = pd.read_csv(dataset_path)

        # Load raw multipliers for stats
        mult_path = os.path.join(BASE_DIR, "multipliers.csv")
        mult_df = pd.read_csv(mult_path)
        self.multipliers = mult_df["Multiplier"].values.tolist()

        # ---------- Feature engineering ----------
        targets = raw["target"].values
        df = build_sequence_features(targets)

        X = df[FEATURE_COLS].values
        y = df["target"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ---------- Regression ensemble ----------
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )

        if XGBRegressor is not None:
            xgb = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                n_jobs=1,
                verbosity=0,
            )
            self.model = VotingRegressor(
                estimators=[("rf", rf), ("xgb", xgb)],
                weights=[0.4, 0.6],
            )
        else:
            # Fallback if xgboost unavailable
            self.model = rf

        print("[Model] Training VotingRegressor ensemble...")
        self.model.fit(X_train, y_train)

        # ---------- Evaluate regression ----------
        y_pred = self.model.predict(X_test)
        self.mae = round(mean_absolute_error(y_test, y_pred), 4)
        self.rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        self.accuracy_score = round(r2_score(y_test, y_pred), 4)
        print(f"[Model] Regression — MAE: {self.mae}, RMSE: {self.rmse}, R²: {self.accuracy_score}")

        # ---------- Crash classifier ----------
        y_crash = (y_train < 2.0).astype(int)
        y_crash_test = (y_test < 2.0).astype(int)

        base_clf = LogisticRegression(max_iter=1000, random_state=42)
        self.crash_clf = CalibratedClassifierCV(
            base_clf, method="isotonic", cv=5
        )
        print("[Model] Training crash classifier...")
        self.crash_clf.fit(X_train, y_crash)

        crash_proba = self.crash_clf.predict_proba(X_test)[:, 1]
        self.crash_auc = round(roc_auc_score(y_crash_test, crash_proba), 4)
        print(f"[Model] Crash classifier AUC: {self.crash_auc}")

        # ---------- Save models ----------
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.model, os.path.join(MODELS_DIR, "model.pkl"))
        joblib.dump(self.crash_clf, os.path.join(MODELS_DIR, "crash_classifier.pkl"))
        print(f"[Model] Models saved to {MODELS_DIR}")

    # ------------------------------------------------------------------ #
    #  Prediction
    # ------------------------------------------------------------------ #
    def predict_next(self, last_10_rounds: list[float]) -> dict:
        """
        Predict the next multiplier given the last 10 round results.
        Returns prediction, calibrated confidence, risk level,
        suggested cashout, crash probability.
        """
        # Pad / truncate to exactly 10
        if not last_10_rounds or len(last_10_rounds) < 2:
            last_10_rounds = [1.5, 2.0, 1.8, 3.2, 1.2, 2.5, 1.1, 4.0, 2.8, 1.9]
        while len(last_10_rounds) < 10:
            last_10_rounds = [last_10_rounds[0]] + last_10_rounds
        last_10_rounds = last_10_rounds[-10:]

        arr = np.array(last_10_rounds, dtype=float)

        # Build the 8 features
        last_1 = arr[-1]
        last_3_avg = float(np.mean(arr[-3:]))
        last_5_avg = float(np.mean(arr[-5:]))
        last_10_avg = float(np.mean(arr))
        last_10_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        streak_low = int(np.sum(arr[-5:] < 2.0))
        max_recent = float(np.max(arr[-5:]))
        trend = float(arr[-1] - arr[-5])

        features = np.array(
            [[last_1, last_3_avg, last_5_avg, last_10_avg,
              last_10_std, streak_low, max_recent, trend]]
        )

        # Regression prediction
        prediction = float(self.model.predict(features)[0])
        prediction = max(1.0, round(prediction, 2))

        # Crash probability (calibrated)
        crash_prob = 0.5
        if self.crash_clf is not None:
            crash_prob = float(self.crash_clf.predict_proba(features)[0, 1])
        crash_prob = round(crash_prob, 4)

        # Confidence: inverse of crash probability, scaled
        confidence = round(max(0.10, min(0.95, 1.0 - crash_prob)), 2)

        # Risk level
        if prediction < 1.5:
            risk, risk_color = "LOW", "#00e676"
        elif prediction < 3.0:
            risk, risk_color = "MEDIUM", "#ffab00"
        elif prediction < 6.0:
            risk, risk_color = "HIGH", "#ff6d00"
        else:
            risk, risk_color = "EXTREME", "#ff1744"

        # Suggested cashout (conservative: 70 % of prediction)
        suggested_cashout = max(1.1, round(prediction * 0.7, 2))

        return {
            "prediction": prediction,
            "confidence": confidence,
            "risk": risk,
            "risk_color": risk_color,
            "suggested_cashout": suggested_cashout,
            "crash_probability": crash_prob,
            "input_mean": round(last_10_avg, 2),
            "input_var": round(float(np.var(arr)), 2),
            "streak_low": streak_low,
        }

    # ------------------------------------------------------------------ #
    #  Statistics (unchanged logic)
    # ------------------------------------------------------------------ #
    def get_statistics(self) -> dict:
        """Return summary statistics from the full multiplier dataset."""
        arr = np.array(self.multipliers)
        total = len(arr)

        avg = round(float(np.mean(arr)), 2)
        median = round(float(np.median(arr)), 2)
        max_val = round(float(np.max(arr)), 2)
        min_val = round(float(np.min(arr)), 2)

        buckets = {
            "1.0-1.5x": int(np.sum((arr >= 1.0) & (arr < 1.5))),
            "1.5-2.0x": int(np.sum((arr >= 1.5) & (arr < 2.0))),
            "2.0-3.0x": int(np.sum((arr >= 2.0) & (arr < 3.0))),
            "3.0-5.0x": int(np.sum((arr >= 3.0) & (arr < 5.0))),
            "5.0-10.0x": int(np.sum((arr >= 5.0) & (arr < 10.0))),
            "10.0x+": int(np.sum(arr >= 10.0)),
        }

        bucket_pcts = {k: round(v / total * 100, 1) for k, v in buckets.items()}
        crash_rate = round(float(np.sum(arr < 2.0) / total * 100), 1)

        recent = arr[-50:].tolist()
        recent_avg = round(float(np.mean(recent)), 2)

        streaks = []
        current_streak = 0
        for m in arr[-200:]:
            if m < 2.0:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        max_low_streak = max(streaks) if streaks else 0

        return {
            "total_rounds": total,
            "average": avg,
            "median": median,
            "max": max_val,
            "min": min_val,
            "crash_rate_below_2x": crash_rate,
            "distribution": buckets,
            "distribution_pcts": bucket_pcts,
            "recent_50_avg": recent_avg,
            "recent_50": [round(r, 2) for r in recent],
            "max_consecutive_low_streak": max_low_streak,
            "model_mae": self.mae,
            "model_r2": self.accuracy_score,
        }


# Allow running standalone for training / testing
if __name__ == "__main__":
    predictor = AviatorPredictor()
    print("\n--- Test prediction ---")
    result = predictor.predict_next([1.5, 2.3, 4.1, 1.8, 3.2, 1.1, 7.5, 2.0, 3.5, 1.9])
    for k, v in result.items():
        print(f"  {k}: {v}")
