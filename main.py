# Google Cloud Function entry point
from adeptus_optimus_backend import compare


def run(request):
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
