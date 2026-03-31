from flask import Flask, jsonify, render_template
import subprocess
import json
import os
from threading import Thread

from centralized_model import run_centralized
from client import USE_DP

app = Flask(__name__)

# ✅ GLOBAL STATE
training_status = {"status": "idle"}
current_accuracy = 0
current_round = 0


# 🏠 HOME ROUTE
@app.route("/")
def home():
    return render_template("index.html")


# 🚀 START TRAINING (NON-BLOCKING)
@app.route("/start")
def start_training():
    def run_training():
        global training_status

        try:
            training_status["status"] = "running"

            # Run federated server
            subprocess.run(["python", "server.py"])

            training_status["status"] = "done"

        except Exception as e:
            print("ERROR:", e)
            training_status["status"] = "error"

    Thread(target=run_training).start()

    return jsonify({"message": "Training Started 🚀"})


# 📊 GET RESULTS
@app.route("/results")
def get_results():
    try:
        # 🔥 Centralized
        centralized = run_centralized()

        # 🔥 Federated (from JSON)
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

        # ✅ RETURN CLEAN DATA FOR DASHBOARD
        return jsonify({
            "accuracy": federated.get("accuracy", 0),
            "precision": federated.get("precision", 0),
            "recall": federated.get("recall", 0),
            "f1": federated.get("f1_score", 0),
            "time": federated.get("training_time", 0),

            "c_accuracy": centralized.get("accuracy", 0),
            "c_precision": centralized.get("precision", 0),
            "c_recall": centralized.get("recall", 0),
            "c_f1": centralized.get("f1_score", 0),
            "c_time": centralized.get("training_time", 0),

            "status": training_status["status"],
            "privacy": USE_DP
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)