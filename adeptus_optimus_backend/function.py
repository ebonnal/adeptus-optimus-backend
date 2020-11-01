import json
import traceback
from time import time

from .utils import RequirementFailError
from .core import compute_heatmap, Profile, Weapon


def parse_profile(letter, params):
    present_indexes = []
    params_keys = params.keys()
    for index in range(5):
        for key in params_keys:
            if f"{letter}{index}" in key:
                present_indexes.append(index)
                break

    return Profile([
        Weapon(
            hit=params.get(f"WSBS{letter}{index}", "0"),
            a=params.get(f"A{letter}{index}", "0"),
            s=params.get(f"S{letter}{index}", "0"),
            ap=params.get(f"AP{letter}{index}", "0"),
            d=params.get(f"D{letter}{index}", "0"),
            options=params[f"options{letter}{index}"])
        for index in present_indexes], params[f"points{letter}"])


def parse_params(params):
    profile_a, profile_b = parse_profile("A", params), parse_profile("B", params)
    print(f"Parsed {len(profile_a.weapons)} weapons for profile A and {len(profile_b.weapons)} for profile B.")
    return profile_a, profile_b


adeptus_optimus_web_app = "https://adeptus-optimus.web.app"


def compare(request):
    # see https://cloud.google.com/functions/docs/writing/http?hl=fr#functions_http_cors-python
    # For more information about CORS and CORS preflight requests, see
    # https://developer.mozilla.org/en-US/docs/Glossary/Preflight_request
    # for more information.

    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': adeptus_optimus_web_app,
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return '', 204, headers

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': adeptus_optimus_web_app
    }

    start_time = time()
    try:
        params = request.args.get('params')
        print(params)
        if params is not None:
            params = json.loads(params)
        else:
            print("Empty props received")
        try:
            response = compute_heatmap(*parse_params(params)), 200, headers
        except RequirementFailError as e:
            response = {"msg": f"Bad input: {e}"}, 422, headers
    except Exception as e:
        traceback.print_exc()
        response = {"msg": f"{type(e)}: {e}"}, 500, headers
    print(f"Request processing took {time() - start_time} seconds")
    return response
