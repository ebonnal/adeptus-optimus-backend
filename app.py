import os
import subprocess
from time import time

from adeptus_optimus_backend import ddos_tanking_treat_request
from flask import Flask, request

# Flask
app = Flask(__name__)


@app.route('/engine/', methods=['GET', 'OPTIONS'])
def run_dev():
    start_time = time()
    response = ddos_tanking_treat_request(request, "*")
    print(f"Run dev took {time() - start_time} seconds")
    return response


if __name__ == "__main__":
    os.environ["FLASK_APP"] = "app.py"
    subprocess.call(["python3", "-m", "flask", "run"])
