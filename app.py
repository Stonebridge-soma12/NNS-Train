import flask
from flask import Flask
import json
from train import Model
from dataset import get_dataset
from datetime import datetime

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/run', methods=['GET', 'POST'])
def run():
    # Load dataset
    body = flask.request.data
    body = json.loads(body)
    model = Model(body, flask.request.headers['id'])

    data, label = get_dataset(body, model.model)
    model.fit(data, label)

    return flask.jsonify(
        finished_time=datetime.now()
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
