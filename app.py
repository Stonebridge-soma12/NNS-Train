import flask
from flask import Flask
from urllib import request as req
import zipfile
import tensorflow as tf
import json
import pandas as pd
import train
from sklearn.preprocessing import MinMaxScaler


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

    data_set = body['data_set']

    df = pd.read_csv(data_set['train_uri'])
    label = df[data_set['label']]
    data = df.drop(axis=1, columns=[data_set['label']])

    # Get shape for data set reshaping for compatible with input shape of model.
    input_shape = list(*model.layers[0].output_shape)

    for i, val in enumerate(input_shape):
        if val == None:
            input_shape[i] = -1

    # Data Normalization
    mms = MinMaxScaler()
    data = mms.fit_transform(data)

    # Data reshape.
    data = data.reshape(input_shape)

    # Fit model
    train.fit_model(model, data, label)

    return "I'm gonna fucking mad."


if __name__ == '__main__':
    app.run()
