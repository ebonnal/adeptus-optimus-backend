import os
import subprocess
import json
from time import time

# TODO: log on GCS each duration and parameters and id/token
from flask import Flask
from flask import request

from engineutils import Weapon, require, RequirementFailError, Bonuses
from enginecore import compute_heatmap


# Flask
app = Flask(__name__)


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
    start_time = time()
    try:
        params = request.args.get('params')
        print(params)
        if params is not None:
            params = json.loads(params)
        else:
            print("Empty props received")
        try:
            response = compute_heatmap(*parse_weapons(params)), 200
        except RequirementFailError as e:
            response = {"msg": f"Bad input: {e}"}, 422
    except Exception as e:
        print(e, e.__traceback__)
        response = {"msg": f"{type(e)}: {str(e)}"}, 500
    print(f"Request processing took {time() - start_time} seconds")
    return response

# v3.0 SAG/Bolt Request processing took 57.40342354774475 seconds



if __name__ == "__main__":
    os.environ["FLASK_APP"] = "app.py"
    subprocess.call(["python3", "-m", "flask", "run"])
