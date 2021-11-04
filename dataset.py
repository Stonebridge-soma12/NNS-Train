import numpy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from urllib import request as req
import numpy as np


def load_data(data_config, shape):
    try:
        df = pd.read_csv(data_config['train_uri'])
    except ConnectionError as e:
        return e

    label = df[data_config['label']]

    if data_config['normalization']['method'] == 'Image':
        df = get_image_data_from_csv(df, shape)
    else:
        df = df.drop(axis=1, columns=[data_config['label']])

    return df, label


def get_input_shape(data, shape):
    # Param shape must be pointer
    new_shape = shape

    # Add dimension for batch
    for i, val in enumerate(new_shape):
        if val == None:
            new_shape[i] = -1

    return new_shape


def normalization(data, norm):
    res = data
    method = norm['method']

    if method == 'MinMax':
        mms = MinMaxScaler()
        res = mms.fit_transform(res)
    elif method == 'Standard':
        ss = StandardScaler()
        res = ss.fit_transform(res)
    elif method == 'Image':
        None
    else:
        res = data.to_numpy()

    return res


def get_dataset(data_config, model):
    shape = list(*model.layers[0].output_shape)

    data, label = load_data(data_config, shape)
    norm_type = data_config['normalization']

    print(norm_type)

    if norm_type['method'] == 'Image':
        # preprocessing for image data
        x = np.array(data)
        y = np.array(label)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, stratify=y)

        datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.3)
        train = datagen.flow(
            x=x_train, y=y_train, subset='training'
        )
        valid = datagen.flow(
            x=x_val, y=y_val, subset='validation'
        )
        
        data = [train, valid]
        label = None
    else:
        data = normalization(data, norm_type)
        data = data.values.reshape(get_input_shape(data, shape))

    return data, label


def url_to_image(url, shape):
    r = req.Request(url)
    res = req.urlopen(r)
    image = np.asarray(bytearray(res.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    print(shape)
    print(shape[1:3])
    image = cv2.resize(image, shape[1:3])

    return image


def get_image_data_from_csv(df, shape):
    images = []
    for url in df['url']:
        image = url_to_image(url, shape)
        images.append(image)

    return images


