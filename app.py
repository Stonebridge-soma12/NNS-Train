import os

from trainer import Trainer

if __name__ == '__main__':
    print('run')
    host = os.environ['RABBIT_HOST']
    train = Trainer(host=host, queue='Request')
    train.run()

