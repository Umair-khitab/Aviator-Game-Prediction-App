"""
Aviator Game Prediction — Flask Web Server (Upgraded)
=====================================================
- Enhanced /predict endpoint with sequence features
- SQLite prediction history (/history, /accuracy)
- Streak analysis (/streak)
- Kelly Criterion bet sizing
- CORS + SocketIO support
"""

import os
import json
import sqlite3
import math
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from model import AviatorPredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "predictions.db")

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Initialized inside main
predictor = None


# ── SQLite helpers ──────────────────────────────────────────────
def init_db():
    """Create predictions table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_rounds TEXT NOT NULL,
            predicted REAL NOT NULL,
            actual REAL,
            confidence REAL NOT NULL,
            crash_probability REAL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


def log_prediction(input_rounds, predicted, confidence, crash_prob):
    """Insert a prediction record."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO predictions (timestamp, input_rounds, predicted, actual, confidence, crash_probability)
        VALUES (?, ?, ?, NULL, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            json.dumps(input_rounds),
            predicted,
            confidence,
            crash_prob,
        ),
    )
    conn.commit()
    conn.close()


# ── Kelly Criterion ─────────────────────────────────────────────
def kelly(confidence: float, odds: float, bankroll: float) -> dict:
    """
    Calculate quarter-Kelly bet size.
    confidence: win probability (0-1)
    odds: decimal odds (e.g. prediction multiplier)
    bankroll: total bankroll amount
    Returns dict with edge, fraction, and recommended bet.
    """
    if odds <= 1.0 or confidence <= 0:
        return {"edge": 0, "fraction": 0, "recommended_bet": 0}

    # Kelly fraction: f* = (bp − q) / b
    b = odds - 1  # net odds
    p = confidence
    q = 1 - p
    edge = b * p - q
    fraction = max(0, edge / b) if b > 0 else 0

    # Quarter-Kelly for safety
    quarter_kelly = fraction / 4
    recommended_bet = round(bankroll * quarter_kelly, 2)

    return {
        "edge": round(edge * 100, 2),            # percentage
        "fraction": round(quarter_kelly * 100, 2), # percentage
        "recommended_bet": recommended_bet,
    }


# ── Routes ──────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict the next multiplier.
    Accepts JSON body: { "last_10_rounds": [float x 10] }
    Also still accepts legacy format: { "multipliers": [...] }
    """
    try:
        data = request.get_json()

        # Support both new and legacy input formats
        raw_rounds = data.get("last_10_rounds") or data.get("multipliers", [])

        parsed = []
        for m in raw_rounds:
            try:
                val = float(m)
                if val >= 1.0:
                    parsed.append(val)
            except (ValueError, TypeError):
                continue

        result = predictor.predict_next(parsed)

        # Kelly calculation (default 1000 bankroll, frontend can override)
        bankroll = float(data.get("bankroll", 1000))
        kelly_data = kelly(result["confidence"], result["prediction"], bankroll)

        response = {
            "multiplier": result["prediction"],
            "confidence": result["confidence"],
            "risk_level": result["risk"],
            "risk_color": result["risk_color"],
            "suggested_cashout": result["suggested_cashout"],
            "crash_probability": result["crash_probability"],
            "kelly_fraction": kelly_data["fraction"],
            "kelly_edge": kelly_data["edge"],
            "kelly_bet": kelly_data["recommended_bet"],
            "streak_low": result["streak_low"],
            # Legacy fields (don't break old frontend)
            "prediction": result["prediction"],
            "risk": result["risk"],
            "input_mean": result["input_mean"],
            "input_var": result["input_var"],
        }

        # Log to SQLite
        log_prediction(parsed, result["prediction"], result["confidence"], result["crash_probability"])

        return jsonify({"status": "ok", "data": response})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    """Return dataset statistics."""
    try:
        result = predictor.get_statistics()
        return jsonify({"status": "ok", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def history():
    """Return the last 50 prediction records."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT 50"
        ).fetchall()
        conn.close()

        records = []
        for r in rows:
            records.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "input_rounds": json.loads(r["input_rounds"]),
                "predicted": r["predicted"],
                "actual": r["actual"],
                "confidence": r["confidence"],
                "crash_probability": r["crash_probability"],
            })

        return jsonify({"status": "ok", "data": records})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/accuracy", methods=["GET"])
def accuracy():
    """Return running accuracy statistics."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        total = conn.execute("SELECT COUNT(*) as cnt FROM predictions").fetchone()["cnt"]

        # For hit_rate: predictions where actual is not null and
        # predicted is within 20% of actual
        hits_row = conn.execute(
            """
            SELECT COUNT(*) as cnt FROM predictions
            WHERE actual IS NOT NULL
              AND ABS(predicted - actual) / actual <= 0.2
            """
        ).fetchone()

        actuals = conn.execute(
            """
            SELECT predicted, actual FROM predictions
            WHERE actual IS NOT NULL
            """
        ).fetchall()

        conn.close()

        mae = 0.0
        hit_rate = 0.0
        actual_count = len(actuals)

        if actual_count > 0:
            errors = [abs(r["predicted"] - r["actual"]) for r in actuals]
            mae = round(sum(errors) / len(errors), 4)
            hit_rate = round(hits_row["cnt"] / actual_count * 100, 1)

        return jsonify({
            "status": "ok",
            "data": {
                "total_predictions": total,
                "hit_rate": hit_rate,
                "mae": mae,
                "actual_count": actual_count,
                "model_status": "active",
            },
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/streak", methods=["POST"])
def streak():
    """
    Streak analysis.
    Accepts JSON body: { "last_20_rounds": [float x 20] }
    Returns streak type, length, risk flag, and message.
    """
    try:
        data = request.get_json()
        rounds = data.get("last_20_rounds", [])

        parsed = []
        for m in rounds:
            try:
                val = float(m)
                if val >= 1.0:
                    parsed.append(val)
            except (ValueError, TypeError):
                continue

        if len(parsed) < 3:
            return jsonify({
                "status": "ok",
                "data": {
                    "current_streak_type": "neutral",
                    "streak_length": 0,
                    "risk_flag": False,
                    "message": "Not enough data for streak analysis",
                },
            })

        # Analyze the most recent streak
        streak_type = "neutral"
        streak_length = 0
        risk_flag = False
        message = ""

        # Count consecutive low (< 2.0) from the end
        low_streak = 0
        for v in reversed(parsed):
            if v < 2.0:
                low_streak += 1
            else:
                break

        # Count consecutive high (> 5.0) from the end
        high_streak = 0
        for v in reversed(parsed):
            if v > 5.0:
                high_streak += 1
            else:
                break

        # Count crashes in last 5
        recent_5 = parsed[-5:] if len(parsed) >= 5 else parsed
        crashes_in_5 = sum(1 for v in recent_5 if v < 2.0)

        if low_streak >= 3 or crashes_in_5 >= 4:
            streak_type = "crash"
            streak_length = max(low_streak, crashes_in_5)
            risk_flag = True
            message = f"⚠️ {crashes_in_5} crashes detected — HIGH RISK"
        elif high_streak >= 3:
            streak_type = "hot"
            streak_length = high_streak
            risk_flag = False
            message = "🚀 Hot streak — momentum signal"
        else:
            # Check last 3 for moderate trends
            last_3 = parsed[-3:]
            if all(v > 5.0 for v in last_3):
                streak_type = "hot"
                streak_length = 3
                message = "🚀 Hot streak — momentum signal"
            elif all(v < 2.0 for v in last_3):
                streak_type = "crash"
                streak_length = 3
                risk_flag = True
                message = f"⚠️ {3} crashes in a row — HIGH RISK"
            else:
                avg = sum(last_3) / len(last_3)
                streak_type = "neutral"
                streak_length = 0
                message = f"📊 Normal variance — avg last 3: {round(avg, 2)}x"

        return jsonify({
            "status": "ok",
            "data": {
                "current_streak_type": streak_type,
                "streak_length": streak_length,
                "risk_flag": risk_flag,
                "message": message,
            },
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("[Server] Initializing predictor...")
    # Initialize predictor at the last moment to avoid double-loading
    predictor = AviatorPredictor()
    print("[Server] Model ready! Starting server...")
    socketio.run(app, host="0.0.0.0", debug=False, port=5001, allow_unsafe_werkzeug=True)
