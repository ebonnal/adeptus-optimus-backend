import os
import subprocess

from flask import Flask
from flask import request

app = Flask(__name__)


@app.route('/engine/', methods=['GET'])
def login():
    statement = request.args.get('props')
    if statement is not None:
        data = query_runner.run(statement)
        fig = plotter.plot(data, statement, append=int(request.args.get('append')))
        return "<p>" + fig + "</p>"
    else:
        with open("index.html") as f:
            return f.read()


if __name__ == "__main__":
    os.environ["FLASK_APP"] = "app.py"
    subprocess.call(["python3", "-m", "flask", "run"])
