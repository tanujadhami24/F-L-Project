from flask import Flask, render_template
import subprocess
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


# 🔥 NEW ROUTE (THIS FIXES YOUR BUTTON)
@app.route("/start")
def start_training():
    try:
        # Run server in background
        subprocess.Popen(["python", "server.py"])

        return "Training Started Successfully 🚀"
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)