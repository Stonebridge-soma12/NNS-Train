import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import cv2
from urllib import request as req
import numpy as np


def load_data(data_config):
    try:
        df = pd.read_csv(data_config['train_uri'])
    except ConnectionError as e:
        return e

    label = df[data_config['label']]
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
        res /= 255.0
    else:
        pass

    return res


def get_dataset(req_body, model):
    data_config = req_body['data_set']

    shape = list(*model.layers[0].output_shape)

    data, label = load_data(data_config)
    data = normalization(data, data_config['normalization'])
    data = data.reshape(get_input_shape(data, shape))

    return data, label


def url_to_image(url):
    header = {
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 9_3_2 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0 Mobile/13F69 Safari/601.1',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3', 'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8', 'Connection': 'keep-alive',
    }
    r = req.Request(url, headers=header)
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

# TODO: 데이터타입을 보고 이미지일 경우 이미지에 맞게 프로세싱