from dog_breed.data import datasets
from dog_breed.common import paths
from dog_breed.common import graph

import pandas as pd
import numpy as np
from sklearn.datasets import load_files
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

DOG_NAMES = datasets.get_dog_names()
FOLDERS = paths.Folders()


def _get_labels_from_data(data) -> pd.Series:
    unique, counts = np.unique(data['target'], return_counts=True)
    labels_set = pd.Series(index=[DOG_NAMES[u] for u in unique], data=counts)
    return labels_set.sort_values(ascending=False)


def _get_labels_from_folder(folder) -> pd.Series:
    data = load_files(folder)
    return _get_labels_from_data(data)


def get_train_labels(folder=FOLDERS.training_data) -> pd.Series:
    return _get_labels_from_folder(folder)


def get_test_labels(folder=FOLDERS.test_data) -> pd.Series:
    return _get_labels_from_folder(folder)


def get_valid_labels(folder=FOLDERS.validation_data) -> pd.Series:
    return _get_labels_from_folder(folder)


if __name__ == '__main__':
    test_set = get_test_labels()
    train_set = get_train_labels()
    valid_set = get_valid_labels()

    a, b, c = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]

    lenght_list = [len(x) for x in [train_set, valid_set, test_set]]
    outer_pie = train_set.tolist() + valid_set.tolist() + test_set.tolist()
    colors = [np.linspace(0, 1, ll) for ll in lenght_list]

    inner_pie = [x.sum() for x in [train_set, valid_set, test_set]]
    inner_labels = ['Train', 'Validation', 'Test']
    # plt.pie(,
    #         # labeldistance=1,
    #         startangle=90,
    #         pctdistance=.7,
    #         explode=[.1, .1, .1],
    #         labels=)
    plt.close('all')
    fig, ax = plt.subplots()
    ax.axis('equal')
    outer_pie_graph = ax.pie(outer_pie,
                             radius=1.3,
                             labels=[''] * len(outer_pie),
                             )
    plt.setp(outer_pie_graph, width=0.3, edgecolor='white')

    # Second Ring (Inside)
    inner_pie_graph, _ = ax.pie(inner_pie,
                                radius=1.3 - 0.3,
                                labels=inner_labels,
                                labeldistance=1.1,
                                colors=[x(0.5) for x in [a, b, c]],
                                # colors=[a(0.5), a(0.4), a(0.3), b(0.5), b(0.4),
                                #         c(0.6), c(0.5), c(0.4), c(0.3), c(0.2)])
                                )
    plt.setp(inner_pie_graph, width=0.4, edgecolor='white')
    plt.margins(0, 0)
    plt.show()

    print(f'Number of train samples {train_set.sum()}')
    print(f'Number of validation samples {valid_set.sum()}')
    print(f'Number of test samples {test_set.sum()}')

    plt.figure()
    sns.barplot(data=train_set.reset_index(drop=False), **graph.args_barplot)
    plt.title('Train')
    plt.figure()
    sns.barplot(data=valid_set.reset_index(drop=False), **graph.args_barplot)
    plt.title('Valudation')
    plt.figure()
    sns.barplot(data=test_set.reset_index(drop=False), **graph.args_barplot)
    plt.title('Test')
