from flask import Flask
from urllib import request as req
import zipfile
import tensorflow as tf

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    r = req.Request("http://127.0.0.1:8080/getmodel")

    open('model.py', 'wb').write(req.urlopen(r).read())

    return 'Hello World!'


@app.route('/run')
def run():
    # Get Saved model and Unzip
    r = req.Request("http://127.0.0.1:8080/getmodel")

    open('./model.zip', 'wb').write(req.urlopen(r).read())

    with zipfile.ZipFile('./model.zip', 'r') as zip_ref:
        zip_ref.extractall('./')

    # Load model
    model = tf.keras.models.load_model('./MNIST')


if __name__ == '__main__':
    app.run()
