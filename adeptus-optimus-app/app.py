import os
import subprocess
import json

from flask import Flask
from flask import request

app = Flask(__name__)


@app.route('/engine/', methods=['GET'])
def compare():
    params = request.args.get('params')
    print(params)
    if params is not None:
        print(json.loads(params))
    else:
        print("Empty props received")
    return {
        "matrix":
        [
            [0.75, 0.00, 0.75, 0.75, 0.00],
            [0.00, 0.00, 0.75, 0.75, 0.00],
            [0.75, 0.75, 0.75, 0.75, 0.75],
            [0.00, 0.00, 0.00, 0.75, 0.00],
            [0.00, 0.00, 0.75, 0.75, 0.00],
            [0.00, 0.00, 0.75, 0.75, 0.00],
            [0.75, 0.75, 0.75, 0.75, 0.75],
            [0.00, 0.00, 0.00, 0.75, 0.00],
            [0.00, 0.00, 0.75, 0.75, 0.00],
            [0.00, 0.00, 0.75, 0.75, 0.00]
        ]}


if __name__ == "__main__":
    os.environ["FLASK_APP"] = "app.py"
    subprocess.call(["python3", "-m", "flask", "run"])
