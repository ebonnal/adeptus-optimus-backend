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
        params = json.loads(params)
    else:
        print("Empty props received")
    if assertParamsValidity(params):
        return {
            "x": ['A', 'B', 'C', 'D', 'E'],
            "y": ["(1, 1)", "(2, 1)", "(3, 1)", "(4, 1)", "(5, 1)", "(6, 1)", "(7, 1)", "(8, 1)", "(9, 1)", "(10, 1)"],
            "z":
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
            ]}, 200
    else:
        return {"msg": "Weapons parameters invalid"}, 422

def assertParamsValidity(params):
    return "bla" not in params["nameA"]

if __name__ == "__main__":
    os.environ["FLASK_APP"] = "app.py"
    subprocess.call(["python3", "-m", "flask", "run"])
