import json
import os

import pika
from train import Model
from dataset import get_dataset
import tensorflow as tf


class Trainer:
    connection = None
    channel = None
    queue = ''

    def __init__(self, host, queue):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = self.connection.channel()
        self.queue = queue

    def train_callback(self, ch, method, props, body):
        req_body = json.loads(body)

        model = Model(req_body['config'], req_body['id'])

        data, label = get_dataset(body['dataset'], model.model)

        try:
            model.fit(data, label)
        except tf.errors.InvalidArgumentError as e:
            res = {'status': 500, 'msg': e}
        except tf.errors.AbortedError as e:
            res = {'status': 500, 'msg': e}
        else:
            res = {'status': 200, 'msg': None}

        res = json.dumps(res).encode('utf-8')

        self.channel.basic_publish(
            exchange='',
            routing_key=props.reply_to,
            properties=pika.BasicProperties(
                correlation_id=props.correlation_id,
            ),
            body=res
        )

    def run(self):
        self.channel.basic_consume(queue=self.queue, on_message_callback=self.train_callback, auto_ack=True)

        print(' [*] Waiting for messages. To exit press CTRL+C')
        self.channel.start_consuming()


if __name__ == '__main__':
    train = Trainer(host='0.0.0.0', queue='train')
    train.run()
