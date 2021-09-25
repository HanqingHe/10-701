import os
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from HW1.knn import KNN
from HW1.nb import NB
from HW1.util import mpl_default_setting, split_nan, split_xy


def impute_datasets(train_path: str, test_path: str,
                    train_imputed_path: str, test_imputed_path: str,
                    k: int = 10):
    # Load datasets
    dset_train = pd.read_csv(train_path)
    dset_test = pd.read_csv(test_path)
    # Concatenate training and test set
    dset = pd.concat([dset_train, dset_test], ignore_index=True)
    dset_train_idx = dset_train.index
    dset_test_idx = dset.index.drop(dset_train_idx)
    # Split dataset with and without missing values
    dset_clean, dset_nan = split_nan(dset)
    dset_clean_X, dset_clean_y = dset_clean.drop(columns=target), dset_clean[target]
    dset_nan_X, dset_nan_y = dset_nan.drop(columns=target), dset_nan[target]
    # Train KNN
    knn = KNN(k)
    knn.train(X=dset_clean_X, y=dset_clean_y)
    # Impute missing values
    dset_nan_X_imputed = knn.impute(X=dset_nan_X)
    # Concatenate imputed features with labels
    dset_nan_imputed = pd.concat([dset_nan_X_imputed, dset_nan_y], axis=1)
    dset_nan_imputed.set_index(dset_nan.index)
    # Concatenate clean and imputed datasets
    dset_imputed = pd.concat([dset_clean, dset_nan_imputed])
    # Split imputed dataset into training and test sets
    dset_train_imputed = dset_imputed.loc[dset_train_idx]
    dset_test_imputed = dset_imputed.loc[dset_test_idx]
    # Save imputed datasets to files
    dset_train_imputed.to_csv(train_imputed_path, index=False)
    dset_test_imputed.to_csv(test_imputed_path, index=False)


def report_parameters(X: pd.DataFrame, y: pd.Series):

    nb = NB()
    nb.train(X, y)

    print('py'.center(50, '='))
    print(nb.py)

    print('px'.center(50, '='))
    for y_val, px in nb.px.items():
        print(str(y_val).center(50, '='))

        print('education-num'.center(50, '-'))
        print('mean', px['numerical'].loc['mean', 'education-num'])
        print('var', px['numerical'].loc['std', 'education-num'] ** 2)

        print('marital-status'.center(50, '-'))
        print(px['categorical']['marital-status'])

        print('race'.center(50, '-'))
        print(px['categorical']['race'])

        print('capital-gain'.center(50, '-'))
        print('mean', px['numerical'].loc['mean', 'capital-gain'])
        print('var', px['numerical'].loc['std', 'capital-gain'] ** 2)

    print('Log-posterior'.center(50, '-'))
    y_pred, LL = nb.predict(X=X_test_imputed[:10], return_LL=True)
    print(LL)


def evaluation(X_train: pd.DataFrame, y_train: pd.Series,
               X_train_imputed: pd.DataFrame, y_train_imputed: pd.Series) -> Tuple[NB, NB]:
    nb = NB()
    nb_imputed = NB()
    nb.train(X_train, y_train)
    nb_imputed.train(X_train_imputed, y_train_imputed)
    print(np.mean(y_train == nb.predict(X_train)))
    print(np.mean(y_train == nb_imputed.predict(X_train)))
    print(np.mean(y_test == nb.predict(X_test)))
    print(np.mean(y_test == nb_imputed.predict(X_test)))
    return nb, nb_imputed


def generate_learning_curves(dset_train: pd.DataFrame,
                             dset_test: pd.DataFrame,
                             dset_train_imputed: pd.DataFrame,
                             dset_test_imputed: pd.DataFrame,
                             target: str,
                             res_path: str, res_imputed_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    res = {'n_data': [], 'train_acc': [], 'test_acc': []}
    res_imputed = {'n_data': [], 'train_acc': [], 'test_acc': []}

    for i in range(5, 14, 1):
        m = int(2 ** i)
        # Data without imputation
        train = dset_train[:m].dropna()
        X_train, y_train = split_xy(train, target)
        # Data with imputation
        train_imputed = dset_train_imputed[:m].dropna()
        X_train_imputed, y_train_imputed = split_xy(train_imputed, target)

        nb = NB()
        nb_imputed = NB()
        nb.train(X_train, y_train)
        nb_imputed.train(X_train_imputed, y_train_imputed)

        res['n_data'].append(len(train))
        res['train_acc'].append(np.mean(y_train == nb.predict(X_train)))
        res['test_acc'].append(np.mean(y_test == nb.predict(X_test)))
        res_imputed['n_data'].append(len(train_imputed))
        res_imputed['train_acc'].append(np.mean(y_train_imputed == nb.predict(X_train_imputed)))
        res_imputed['test_acc'].append(np.mean(y_test_imputed == nb.predict(X_test_imputed)))

    res = pd.DataFrame(res)
    res_imputed = pd.DataFrame(res_imputed)
    res.to_csv(res_path, index=False)
    res_imputed.to_csv(res_imputed_path, index=False)

    return res, res_imputed


def plot_learning_curves(res_path: str, res_imputed_path: str):
    res = pd.read_csv(res_path)
    res_imputed = pd.read_csv(res_imputed_path)

    mpl_default_setting()

    plt.title('Accuracy vs data (without imputation)')
    plt.xlabel(r"# data ($m - m'$)")
    plt.ylabel("Accuracy")
    plt.xlim(0, 8500)
    plt.plot(res['n_data'], res['train_acc'], '-r', label='train')
    plt.plot(res['n_data'], res['test_acc'], '-b', label='test')
    plt.legend(loc='upper right')
    plt.show()

    plt.title('Accuracy vs data (with imputation)')
    plt.xlabel(r"# data ($m$)")
    plt.ylabel("Accuracy")
    plt.xlim(0, 8500)
    plt.plot(res_imputed['n_data'], res['train_acc'], '-r', label='train')
    plt.plot(res_imputed['n_data'], res['test_acc'], '-b', label='test')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    dset_dirpath = './data/'
    train_filename = 'census.csv'
    test_filename = 'adult.test.csv'
    train_imputed_filename = 'census_imputed.csv'
    test_imputed_filename = 'adult.test_imputed.csv'
    res_filename = 'res.csv'
    res_imputed_filename = 'res_imputed.csv'

    target = 'income'

    # Paths
    train_path = os.path.join(dset_dirpath, train_filename)
    test_path = os.path.join(dset_dirpath, test_filename)
    train_imputed_path = os.path.join(dset_dirpath, train_imputed_filename)
    test_imputed_path = os.path.join(dset_dirpath, test_imputed_filename)
    res_path = os.path.join(dset_dirpath, res_filename)
    res_imputed_path = os.path.join(dset_dirpath, res_imputed_filename)

    # Train a KNN to impute the missing values in the datasets
    # and save the imputed data to files
    impute_datasets(train_path, test_path,
                    train_imputed_path, test_imputed_path,
                    k=10)

    # Load datasets
    dset_train = pd.read_csv(train_path)
    dset_test = pd.read_csv(test_path)
    dset_train_imputed = pd.read_csv(train_imputed_path)
    dset_test_imputed = pd.read_csv(test_imputed_path)
    # Split datasets into features and labels
    X_train, y_train = split_xy(dset_train, target)
    X_test, y_test = split_xy(dset_test, target)
    X_train_imputed, y_train_imputed = split_xy(dset_train_imputed, target)
    X_test_imputed, y_test_imputed = split_xy(dset_test_imputed, target)

    report_parameters(X_train_imputed, y_train_imputed)

    evaluation(X_train, y_train, X_train_imputed, y_train_imputed)

    generate_learning_curves(dset_train, dset_test,
                             dset_train_imputed, dset_test_imputed,
                             target, res_path, res_imputed_path)

    plot_learning_curves(res_path, res_imputed_path)