import json
import traceback
from time import time

from .utils import RequirementError, is_dev_execution
from .engine import compute_heatmap, Profile, Weapon
from .linkgen import get_short_dynamic_link, get_long_dynamic_link


def parse_profile(letter, params):
    present_indexes = []
    params_keys = params.keys()
    for index in range(8):
        for key in params_keys:
            if f"{letter}{index}" in key:
                present_indexes.append(index)
                break

    return Profile([
        Weapon(
            hit=params[f"WSBS{letter}{index}"],
            a=params[f"A{letter}{index}"],
            s=params[f"S{letter}{index}"],
            ap=params[f"AP{letter}{index}"],
            d=params[f"D{letter}{index}"],
            options=params[f"options{letter}{index}"])
        for index in present_indexes], params.get(f"points{letter}", ""))


def parse_params(params):
    Weapon.at_least_one_blast_weapon = False
    profile_a, profile_b = parse_profile("A", params), parse_profile("B", params)
    if is_dev_execution():
        print(f"Parsed {len(profile_a.weapons)} weapons for profile A and {len(profile_b.weapons)} for profile B.")
    return profile_a, profile_b


def treat_request(request, allowed_origin):
    start_time = time()
    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': allowed_origin,
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '86400'  # firefox max
        }
        return '', 204, headers

    headers = {
        'Access-Control-Allow-Origin': allowed_origin
    }

    try:
        params = request.args.get('params')
        share_settings = request.args.get('share_settings')
        if is_dev_execution():
            print("received params=", params)
            print("received share_settings=", share_settings)
        if params is not None:
            params = json.loads(params)
            try:
                response = compute_heatmap(*parse_params(params)), 200, headers
            except RequirementError as e:
                response = {"msg": f"INVALID INPUT: {e}"}, 422, headers
        elif share_settings is not None:  # dynamic short link gen
            if is_dev_execution():
                dynamic_link = get_long_dynamic_link(share_settings)
            else:
                dynamic_link = get_short_dynamic_link(share_settings)
            response = {"link": dynamic_link}, 200, headers
        else:
            raise RuntimeError("Request json query string should contain key 'share_settings' or 'params'")
    except Exception as e:
        if is_dev_execution():
            traceback.print_exc()
        response = {"msg": f"{type(e)}: {e}"}, 500, headers
    if is_dev_execution():
        print(f"Request processing took {time() - start_time} seconds")
    print(f"user_ip:{request.remote_addr}")
    return response
