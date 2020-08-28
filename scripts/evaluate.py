"""
Evaluates the performance (in terms of test accuracy) of the saved models.
Iterates through the .hdf5 files of a specified folder and assigns the weights to the basic transfer learning network.
Then computes the test accuracy for each of them and reports it into a .csv file.
"""
import os
import argparse
import pandas as pd

from dog_breed.common import paths
from dog_breed.models import transfer_learning as tl


def evaluate_models_in_folder(models_folder) -> pd.DataFrame:
    weight_file_list = list(filter(lambda x: paths.is_weight_file(x),
                                   paths.listdir(models_folder)))  # all weight files
    base_net = list(
        map(lambda x: x.rsplit('.', 2)[1], weight_file_list))  # name of the bottleneck network for each model
    epochs = list(map(lambda x: x.rsplit('_')[3], weight_file_list))  # number of epochs for each model
    basename = list(map(lambda x: os.path.basename(x).split('_')[0], weight_file_list))  # prefix for each model
    augmented = list(map(lambda x: '_A_weight' in x, weight_file_list))

    saved_models_df = pd.DataFrame(data=(base_net, basename, epochs, augmented, weight_file_list),
                                   index=['bottleneck_features', 'prefix', 'epochs', 'augmented', 'path'],
                                   ).T

    if saved_models_df.empty:
        pass
    else:
        for bottle_net, df_by_net in saved_models_df.groupby('bottleneck_features'):
            pretrained_network = bottle_net
            for i in df_by_net.index:
                epochs = df_by_net.loc[i, 'epochs']
                prefix = df_by_net.loc[i, 'prefix']
                data_augmentation = df_by_net.loc[i, 'augmented']
                saved_models_df.loc[i, 'test_accuracy'] = tl.eval_performance(pretrained_network=pretrained_network,
                                                                              epochs=epochs,
                                                                              data_augmentation=data_augmentation,
                                                                              prefix=prefix,
                                                                              dataset='test',
                                                                              )

    return saved_models_df.sort_values(by='test_accuracy', ascending=False)


if __name__ == '__main__':
    defaults = {'models_folder': paths.Folders().models,
                'report_folder': paths.Folders().data,
                }

    parser = argparse.ArgumentParser(
        description='Evaluates the performance (in terms of test accuracy) of the saved models.')

    parser.add_argument('-m', '--models_folder', default=defaults['models_folder'],
                        help=f"Folder containing the weights' (.hdf5) files (default: {defaults['models_folder']})",
                        )
    parser.add_argument('-o', '--output', default=defaults['report_folder'],
                        help=f"Folder where the report (.csv) will be saved (default: {defaults['report_folder']})",
                        )
    args = parser.parse_args()
    arguments = vars(args)

    df = evaluate_models_in_folder(arguments['models_folder'])
    output_file = os.path.join(arguments['output'], 'report.csv')
    print(f"\nWriting report at\t{output_file}")
    df.to_csv(output_file, index=False)
