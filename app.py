from trainer import Trainer

if __name__ == '__main__':
    print('run')
    train = Trainer(host='localhost', queue='Request')
    train.run()
