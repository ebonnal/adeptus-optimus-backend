import os
import subprocess

from adeptus_optimus_backend.function import compare
from flask import Flask, request

# Flask
app = Flask(__name__)


@app.route('/engine/', methods=['GET'])
def compare_dev_route():
    return compare(request)


if __name__ == "__main__":
    os.environ["FLASK_APP"] = "app.py"
    subprocess.call(["python3", "-m", "flask", "run"])
