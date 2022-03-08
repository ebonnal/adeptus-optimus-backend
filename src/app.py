import os
import subprocess
import json
from time import time, sleep

from flask import Flask, request, g as app_ctx

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from adeptus_optimus_backend.function import treat_request
from adeptus_optimus_backend.utils import set_is_dev_execution, delay_from, min_exec_duration_seconds

set_is_dev_execution(False)

# Flask
app = Flask(__name__)

# Limiter per ip
default_limits = ["100 per day", "3 per 30 second"]
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=default_limits
)

@app.before_request
def before():
    # Store the start time for the request
    app_ctx.start_time = time()

def format_rate_limited_response(response):
    str_data = str(response.get_data(), "utf-8")
    if "Too Many Requests" in str_data:
        err_msg = str_data[str_data.index("<p>")+3:-5]
        response.set_data(json.dumps({"msg": f"Rate limit exceeded: {err_msg}"}))
    return response

@app.after_request
def after(response):
    try:
        delay_from(app_ctx.start_time, min_exec_duration_seconds)
    except AttributeError:
        sleep(min_exec_duration_seconds)
    format_rate_limited_response(response)
    return response


@app.route('/engine/', methods=['GET', 'OPTIONS'])
def engine():
    start_time = time()
    response = treat_request(request, "*")
    print(f"Run took {time() - start_time} seconds")
    return response


def create_app():
    return app

if __name__ == "__main__":
    os.environ["FLASK_APP"] = "app.py"
    subprocess.call(["python3", "-m", "flask", "run", "--port", "8080"])
