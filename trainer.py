import json
import urllib.error

import pika
from train import Model
from dataset import get_dataset
import tensorflow as tf
import requests
import os


def train_callback(ch, method, props, body):
    data = None
    label = None
    headers = {'Content-Type': 'application/json; charset=utf-8'}

    req_body = json.loads(body)

    model = Model(req_body['config'], req_body['id'])

    try:
        data, label = get_dataset(req_body['data_set'], model.model)
    except urllib.error.URLError as e:
        res = {'status': 500, 'msg': str(e.args[0])}
        res = json.dumps(res).encode('utf-8')
        print(e.args[0])
        try:
            res = requests.post(os.environ['REPLY_API'], data=res, headers=headers)
        except:
            pass

        return

    try:
        model.fit(data, label)
    except tf.errors.InvalidArgumentError as e:
        res = {'status': 500, 'msg': e}
    except tf.errors.AbortedError as e:
        res = {'status': 500, 'msg': e}
    else:
        res = {'status': 200, 'msg': None}

    print(res['msg'])

    res = json.dumps(res).encode('utf-8')
    try:
        res = requests.post(os.environ['REPLY_API'], data=res, headers=headers)
    except:
        pass

    model.save_model()

    print('Finished message callback.')


class Trainer:
    connection = None
    channel = None
    queue = ''

    def __init__(self, host, queue):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = self.connection.channel()
        self.queue = queue

    def run(self):
        # self.channel.queue_declare(queue=self.queue)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.queue, on_message_callback=train_callback, auto_ack=True)

        print(' [*] Waiting for messages. To exit press CTRL+C')
        self.channel.start_consuming()
