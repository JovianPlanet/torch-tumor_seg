from config import get_parameters
from train import train
from test import test
from analysis import assess

def main(config):
            
    if config['mode'] == 'train':

        train(config)

    elif config['mode'] == 'test':

        test(config)

    elif config['mode'] == 'assess':

        assess(config)

if __name__ == '__main__':
    config = get_parameters('train')
    main(config)