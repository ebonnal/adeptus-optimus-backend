import os
import subprocess
from time import time

from flask import Flask, request
from flask.json import dumps

from adeptus_optimus_backend import ddos_tanking_treat_request
from adeptus_optimus_backend.utils import set_is_dev_execution

set_is_dev_execution(False)

# Flask
app = Flask(__name__)


@app.route('/engine/', methods=['GET', 'OPTIONS'])
def run_dev():
    start_time = time()
    response = ddos_tanking_treat_request(request, "*")
    print(f"Run dev took {time() - start_time} seconds")
    with open("./.output.json", "w") as out:
        out.write(dumps(response[0], indent=None, separators=(",", ":")))
    return response


if __name__ == "__main__":
    os.environ["FLASK_APP"] = "app.py"
    subprocess.call(["python3", "-m", "flask", "run"])
