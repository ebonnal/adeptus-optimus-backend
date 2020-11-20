import json
import traceback
from time import time

from .utils import RequirementError, with_minimum_exec_time, is_dev_execution
from .core import compute_heatmap, Profile, Weapon
from .link import get_short_dynamic_link, get_long_dynamic_link


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
            link = f"https://adeptus-optimus.web.app?share_settings={share_settings}"
            if is_dev_execution():
                short_dynamic_link = get_long_dynamic_link(link)
            else:
                short_dynamic_link = get_short_dynamic_link(link)
            response = {"link": short_dynamic_link}, 200, headers
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


def ddos_tanking_treat_request(request, allowed_origins):
    # Ensures together with --max-instance that even in case
    # of DDOS we do not overpass 2 000 000 requests by month: 30*24*
    min_sec = 3
    percent_marge = 5
    month_in_seconds = (31 * 24 * 3600)
    assert ((1 + percent_marge / 100) * month_in_seconds < 2000000 * min_sec)
    return with_minimum_exec_time(min_sec, lambda: treat_request(request, allowed_origins))
