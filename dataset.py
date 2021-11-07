import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from urllib import request as req
import numpy as np
import zipfile
import tensorflow as tf


def load_data(data_config):
    print('loading dataset...')
    data_type = data_config['kind']

    data = []
    label = []

    if data_type == 'TEXT':
        data = pd.read_csv(data_config['train_uri'])
        label = data[data_config['label']]
        data = data.drop(axis=1, columns=[data_config['label']])
    elif data_type == 'IMAGES':
        r = req.Request(data_config['train_uri'])
        data = open('./dataset.zip', 'wb')
        data.write(req.urlopen(r).read())
        data.close()

        with zipfile.ZipFile('./dataset.zip', 'r') as zip_ref:
            zip_ref.extractall('./dataset')

        data = []
        label = []

    print('load dataset successful')

    return data, label


def get_input_shape(data, shape):
    # Param shape must be pointer
    new_shape = shape

    # Add dimension for batch
    for i, val in enumerate(new_shape):
        if val == None:
            new_shape[i] = -1

    print(new_shape)

    return new_shape


def normalization(data, norm):
    res = data
    method = norm['method']

    if norm['usage'] == False:
        res = data
    else:
        if method == 'MinMax':
            mms = MinMaxScaler()
            res = mms.fit_transform(res)
        elif method == 'Standard':
            ss = StandardScaler()
            res = ss.fit_transform(res)

    return res


def get_dataset(data_config, model):
    shape = list(*model.layers[0].output_shape)

    data, label = load_data(data_config)
    norm_type = data_config['normalization']

    if data_config['kind'] == 'IMAGES':
        # preprocessing for image data
        datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.3)

        color_mode = 'rgb'
        if shape[3] == 1:
            color_mode = 'grayscale'
        elif shape[3] == 4:
            color_mode = 'rgba'

        train = datagen.flow_from_directory(
            directory='./dataset',
            shuffle=data_config['shuffle'],
            subset='training',
            color_mode=color_mode,
            target_size=shape[1:3]
        )
        valid = datagen.flow_from_directory(
            directory='./dataset',
            shuffle=data_config['shuffle'],
            subset='validation',
            color_mode=color_mode,
            target_size=shape[1:3]
        )

        data = [train, valid]
        label = []
    else:
        x = np.array(data)
        y = np.array(label)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, stratify=y,
                                                          shuffle=data_config['shuffle'])

        x_train = normalization(x_train, norm_type)
        x_val = normalization(x_val, norm_type)

        train = x_train.reshape(get_input_shape(x_train, shape))
        valid = x_val.reshape(get_input_shape(x_val, shape))
        data = [train, valid]
        label = [y_train, y_val]

    print(data)
    print(label)

    return data, label


def url_to_image(url, shape):
    r = req.Request(url)
    res = req.urlopen(r)
    image = np.asarray(bytearray(res.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, shape[1:3])

    return image


def get_image_data_from_csv(df, shape):
    images = []
    for url in df['url']:
        image = url_to_image(url, shape)
        images.append(image)

    return images


