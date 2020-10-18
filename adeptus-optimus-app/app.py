import os
import subprocess
import json

# TODO: log on GCS each duration and parameters and id/token
from flask import Flask
from flask import request

from engine import Weapon, compute_heatmap, require, RequirementFailError, Bonuses

# Flask
app = Flask(__name__)
N = 250


def parse_weapons(params):
    weapon_a = Weapon(
        hit=params["WSBSA"],
        a=params["AA"],
        s=params["SA"],
        ap=params["APA"],
        d=params["DA"],
        points=params["pointsA"]
    )

    weapon_b = Weapon(
        hit=params["WSBSB"],
        a=params["AB"],
        s=params["SB"],
        ap=params["APB"],
        d=params["DB"],
        points=params["pointsB"]
    )

    return weapon_a, weapon_b

@app.route('/engine/', methods=['GET'])
def compare():
    try:
        params = request.args.get('params')
        print(params)
        if params is not None:
            params = json.loads(params)
        else:
            print("Empty props received")
        try:
            return compute_heatmap(*parse_weapons(params), N=N), 200
        except RequirementFailError as e:
            return {"msg": f"Bad input: {e}"}, 422

    except Exception as e:
        return {"msg": f"{type(e)}: {str(e)}"}, 500

# Utils
def assertParamsValidity(params):
    return "bla" not in params["nameA"]

# Engine


if __name__ == "__main__":
    os.environ["FLASK_APP"] = "app.py"
    subprocess.call(["python3", "-m", "flask", "run"])
