from flask import Flask, render_template, jsonify
from client import USE_DP
import subprocess
import json
import os

from centralized_model import run_centralized

app = Flask(__name__)


# 🏠 HOME ROUTE
@app.route("/")
def home():
    return render_template("index.html")


# 🚀 START TRAINING
@app.route("/start")
def start_training():
    try:
        # Run federated server in background
        subprocess.Popen(["python", "server.py"])

        return "Training Started Successfully 🚀"
    except Exception as e:
        return f"Error: {str(e)}"


# 📊 GET RESULTS
@app.route("/results")
def get_results():
    try:
        # 🔥 Centralized results
        centralized = run_centralized()

        # 🔥 Federated results (read from JSON file)
        if os.path.exists("federated_results.json"):
            with open("federated_results.json", "r") as f:
                federated = json.load(f)
        else:
            federated = {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "training_time": 0
            }

        return jsonify({
            "centralized": centralized,
            "federated": federated,
            "privacy": "ON" if USE_DP else "OFF"
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)