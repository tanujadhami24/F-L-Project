from flask import Flask, render_template
import subprocess
import time

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start')
def start():
    subprocess.Popen(["python", "server.py"])
    time.sleep(2)
    subprocess.Popen(["python", "client.py", "0"])
    subprocess.Popen(["python", "client.py", "1"])

    return "Training Started Successfully!"

if __name__ == "__main__":
    app.run(debug=True)