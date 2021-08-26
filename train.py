import os
import shutil
import zipfile
from urllib import request as req
import tensorflow as tf
from numba import cuda


class Model:
    __epochs = 10
    __batch_size = 32
    __validation_split = 0.3
    __early_stop = None
    __learning_rate_reduction = True
    __config = None
    model = None

    def __init__(self, config):
        self.__config = config
        self.__epochs = config['epochs']
        self.__batch_size = config['batch_size']
        self.__early_stop = self.config['early_stop']
        self.__learning_rate_reduction = self.config['learning_rate_reduction']
        self.model = get_model_from_url("http://127.0.0.1:8080/model")

    def __get_callbacks(self):
        callbacks = []

        if self.__early_stop['usage']:
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor=self.__early_stop['monitor'],
                patience=self.__early_stop['patience']
            )
            callbacks.append(early_stop)

        if self.__learning_rate_reduction['usage']:
            learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=self.__learning_rate_reduction['monitor'],
                patience=self.__learning_rate_reduction['patience'],
                verbose=1,
                factor=self.__learning_rate_reduction['factor'],
                min_lr=self.__learning_rate_reduction['min_lr']
            )
            callbacks.append(learning_rate_reduction)

        remote_monitor = tf.keras.callbacks.RemoteMonitor(
            root='http://localohst:8080',
            path='/publish/epoch/end',
            field='data',
            headers=None,
            send_as_json=True
        )
        callbacks.append(remote_monitor)

        return callbacks

    def fit(self, data, label):
        callbacks = self.__get_callbacks()

        self.model.fit(
            data, label,
            epochs=self.__epochs,
            batch_size=self.__batch_size,
            validation_split=self.__validation_split,
            callbacks=callbacks
        )

        # Remove model.
        shutil.rmtree('./Model')
        os.remove('./model.zip')

        # for releasing GPU memory
        device = cuda.get_current_device()
        device.reset()


def get_model_from_url(url):
    # Get Saved model and Unzip
    r = req.Request("http://127.0.0.1:8080/model")

    open('./Model.zip', 'wb').write(req.urlopen(r).read())

    with zipfile.ZipFile('./Model.zip', 'r') as zip_ref:
        zip_ref.extractall('./')
        print('extracting...')

    # Load model
    model = tf.keras.models.load_model('./Model')

    return model