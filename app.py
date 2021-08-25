import flask
from flask import Flask
from urllib import request as req
import zipfile
import tensorflow as tf
import json
import train
from dataset import get_dataset
from numba import cuda


app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/run', methods=['GET', 'POST'])
def run():
    # Get Saved model and Unzip
    r = req.Request("http://127.0.0.1:8080/model")

    open('./Model.zip', 'wb').write(req.urlopen(r).read())

    with zipfile.ZipFile('./Model.zip', 'r') as zip_ref:
        zip_ref.extractall('./')
        print('extracting...')

    # Load model
    model = tf.keras.models.load_model('./Model')

    # Load dataset
    body = flask.request.data
    body = json.loads(body)

    data, label = get_dataset(body, model)

    # Fit model
    try:
        train.fit_model(model, data, label)
    except RuntimeError as e:
        return e

    # for releasing GPU memory
    device = cuda.get_current_device()
    device.reset()

    print('Train finished.')

    return "I'm gonna fucking mad."


if __name__ == '__main__':
    app.run()
