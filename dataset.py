import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_data(data_config):
    try:
        df = pd.read_csv(data_config['train_uri'])
    except IOError as e:
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


def normalization(data, method):
    res = data
    if method == 'MinMax':
        mms = MinMaxScaler()
        res = mms.fit_transform(res)
    elif method == 'Standard':
        ss = StandardScaler()
        res = ss.fit_transform(res)
    elif method == 'Image':
        res /= 255.0

    return res


def get_dataset(req_body, model):
    data_config = req_body['data_set']

    shape = list(*model.layers[0].output_shape)

    data, label = load_data(data_config)
    data = normalization(data, data_config['normalization'])
    data = data.reshape(get_input_shape(data, shape))

    return data, label
