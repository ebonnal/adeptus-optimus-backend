import json
import traceback
from time import time

from .utils import RequirementFailError
from .core import compute_heatmap, Profile, Weapon


def parse_profile(letter, params):
    present_indexes = set()
    params_keys = params.keys()
    for index in range(10):
        for key in params_keys:
            if f"{letter}{index}" in key:
                present_indexes.add(index)

    return Profile([
        Weapon(
            hit=params.get(f"WSBS{letter}{index}", "0"),
            a=params.get(f"A{letter}{index}", "0"),
            s=params.get(f"S{letter}{index}", "0"),
            ap=params.get(f"AP{letter}{index}", "0"),
            d=params.get(f"D{letter}{index}", "0"))
        for index in present_indexes], params[f"points{letter}"])

def parse_params(params):
    profile_a, profile_b = parse_profile("A", params), parse_profile("B", params)
    print(f"Parsed {len(profile_a.weapons)} weapons for profile A and {len(profile_b.weapons)} for profile B.")
    return profile_a, profile_b


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
            response = compute_heatmap(*parse_params(params)), 200
        except RequirementFailError as e:
            response = {"msg": f"Bad input: {e}"}, 422
    except Exception as e:
        traceback.print_exc()
        response = {"msg": f"{type(e)}: {e}"}, 500
    print(f"Request processing took {time() - start_time} seconds")
    return response
