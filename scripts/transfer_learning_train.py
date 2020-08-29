"""
Uses transfer learning to train a network.
"""
import argparse
from models.tl import transfer_learning

if __name__ == '__main__':
    defaults = {'epochs': 5,
                'prefix': 'tl',
                }

    parser = argparse.ArgumentParser(description='Uses transfer learning to train a network.')

    parser.add_argument('-n', '--pretrained_network',
                        help='Name of the pretrained network (no quote). Can be: \
                        VGG16\
                        VGG19 \
                        Resnet50 \
                        InceptionV3 \
                        Xception', required=True)
    parser.add_argument('-e', '--epochs', default=defaults['epochs'],
                        help=f'Number of epochs to train the network (default: {defaults["epochs"]})')
    parser.add_argument('-a', '--data_augmentation', default=False, action='store_true',
                        help='Use data augmentation')
    parser.add_argument('-p', '--prefix', default=defaults['prefix'],
                        help=f'Prefix for the weight file to save (default: {defaults["prefix"]})')
    parser.add_argument('-o', '--overwrite', default=False, action='store_true',
                        help='Overwrite existing weight file')

    args = parser.parse_args()

    train_args = vars(args)
    train_args.update({'epochs': int(train_args['epochs'])})

    print('\n')
    if args.pretrained_network.lower() not in ['vgg16', 'vgg19', 'resnet50', 'inceptionv3', 'xception']:
        print(f"Pre trained network {args.pretrained_network} not found")
    else:
        transfer_learning.train_transfer_learning_net(**train_args)
