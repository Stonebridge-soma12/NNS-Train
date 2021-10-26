import json
import urllib.error

import pika
from train import Model
from dataset import get_dataset
import tensorflow as tf
import requests
import os
from numba import cuda


class Trainer:
    connection = None
    channel = None
    queue = ''

    def __init__(self, host, queue):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host, virtual_host=os.environ['VHOST']))
        self.channel = self.connection.channel()
        self.queue = queue

    def run(self):
        # self.channel.queue_declare(queue=self.queue)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.queue, on_message_callback=train_callback, auto_ack=True)

        print(' [*] Waiting for messages. To exit press CTRL+C')
        self.channel.start_consuming()


def train_callback(ch, method, props, body):
    data = None
    label = None

    req_body = json.loads(body)

    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'train_id': str(req_body['train_id'])
    }

    model = Model(req_body['config'], req_body['user_id'], req_body['train_id'], req_body['project_no'])

    try:
        data, label = get_dataset(req_body['data_set'], model.model)
    except urllib.error.URLError as e:
        res = {'status_code': 400, 'msg': str(e.args[0])}
        reply_request(f'https://{os.environ["API_SERVER"]}/api/project/{req_body["project_no"]}/train/{req_body["train_id"]}/reply', res, headers)

    try:
        model.fit(data, label)
    except tf.errors.InvalidArgumentError as e:
        res = {'status_code': 500, 'msg': e, 'train_id': req_body['train_id']}
        reply_request(f'https://{os.environ["API_SERVER"]}/api/project/{req_body["project_no"]}/train/{req_body["train_id"]}/reply', res, headers)
        return
    except tf.errors.AbortedError as e:
        res = {'status_code': 500, 'msg': e, 'train_id': req_body['train_id']}
        reply_request(f'https://{os.environ["API_SERVER"]}/api/project/{req_body["project_no"]}/train/{req_body["train_id"]}/reply', res, headers)
        return
    except tf.errors.FailedPreconditionError as e:
        res = {'status_code': 500, 'msg': e, 'train_id': req_body['train_id']}
        reply_request(f'https://{os.environ["API_SERVER"]}/api/project/{req_body["project_no"]}/train/{req_body["train_id"]}/reply', res, headers)
        return
    except tf.errors.UnknownError as e:
        res = {'status_code': 500, 'msg': e, 'train_id': req_body['train_id']}
        reply_request(f'https://{os.environ["API_SERVER"]}/api/project/{req_body["project_no"]}/train/{req_body["train_id"]}/reply', res, headers)
        return

    try:
        model.save_model()
    except:
        res = {'status_code': 500, 'msg': 'OS error', 'train_id': req_body['train_id']}
        reply_request(f'https://{os.environ["API_SERVER"]}/api/project/{req_body["project_no"]}/train/{req_body["train_id"]}/reply', res, headers)
        return

    # for releasing GPU memory
    # device = cuda.get_current_device()
    # device.reset()

    res = {'status_code': 200, 'msg': '', 'train_id': req_body['train_id']}
    reply_request(f'https://{os.environ["API_SERVER"]}/api/project/{req_body["project_no"]}/train/{req_body["train_id"]}/reply', res, headers)
    return


def reply_request(url, data, headers):
    data = json.dumps(data).encode('utf-8')
    try:
        res = requests.post(url, data=data, headers=headers)
    except urllib.error.URLError as e:
        return e
    else:
        return res
