import numpy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from urllib import request as req
import numpy as np


def load_data(data_config):
    try:
        df = pd.read_csv(data_config['train_uri'])
    except ConnectionError as e:
        return e

    label = df[data_config['label']]

    if data_config['normalization']['method'] == 'image':
        df = get_image_data_from_csv(df)
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

    data, label = load_data(data_config)
    norm_type = data_config['normalization']

    print(norm_type)

    if norm_type['method'] == 'Image':
        # preprocessing for image data
        datagen = ImageDataGenerator(rescale=1.0/255.0)
        data = datagen.flow(
            x=data, y=label,
        )
        label = None
    else:
        data = normalization(data, norm_type)
        data = data.values.reshape(get_input_shape(data, shape))

    return data, label


def url_to_image(url):
    r = req.Request(url)
    res = req.urlopen(r)
    image = np.asarray(bytearray(res.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image


def get_image_data_from_csv(df):
    images = []
    for url in df['image']:
        image = url_to_image(url)
        images.append(image)

    return images


