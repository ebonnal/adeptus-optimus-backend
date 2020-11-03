# Google Cloud Function entry point
from adeptus_optimus_backend import compare


def treat_request(request):
    adeptus_optimus_web_app = "https://adeptus-optimus.web.app"
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
    return compare(request, headers)


def run(request):
    # Ensures together with --max-instance that even in case
    # of DDOS we do not overpass 2 000 000 requests by month: 30*24*
    min_sec = 1.45
    percent_marge = 5
    month_in_seconds = (31 * 24 * 3600)
    assert ((1 + percent_marge / 100) * month_in_seconds < 2000000 * min_sec)
    return with_minimum_exec_time(min_sec, lambda: treat_request(request))
