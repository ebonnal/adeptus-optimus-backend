import json
from time import time

from .utils import Weapon, RequirementFailError
from .core import compute_heatmap


# TODO use a list of weapon's params for each profile
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


def compare(request):
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
