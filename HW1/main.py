import os

import numpy as np
import pandas as pd

from HW1.knn import KNN
from HW1.util import split_nan, Timer, split_xy


def impute_datasets(train_path: str, test_path: str,
                    train_imputed_path: str, test_imputed_path: str,
                    k: int = 10):
    # Load datasets
    dset_train = pd.read_csv(train_path)
    dset_test = pd.read_csv(test_path)
    # Split dataset with and without missing values
    dset = pd.concat([dset_train, dset_test], ignore_index=True)
    dset_clean, dset_nan = split_nan(dset)
    dset_clean_X, dset_clean_y = dset_clean.drop(columns=target), dset_clean[target]
    dset_nan_X, dset_nan_y = dset_nan.drop(columns=target), dset_nan[target]
    # Train KNN
    knn = KNN(k)
    knn.train(X=dset_clean_X, y=dset_clean_y)
    # Impute data
    dset_nan_X_imputed = knn.impute(X=dset_nan_X)
    # Combine imputed dataset
    dset_nan_imputed = pd.concat([dset_nan_X_imputed, dset_nan_y], axis=1)
    dset_nan_imputed.set_index(dset_nan.index)
    dset_imputed = pd.concat([dset_clean, dset_nan_imputed])
    # Split imputed dataset into training and test sets
    dset_train_imputed = dset_imputed.loc[dset_train.index]
    dset_test_imputed = dset_imputed.loc[dset_test.index]
    # Save imputed datasets to files
    dset_train_imputed.to_csv(train_imputed_path, index=False)
    dset_test_imputed.to_csv(test_imputed_path, index=False)


def report_parameters(dset: pd.DataFrame) -> None:
    raise NotImplementedError


def evaluation(dset: pd.DataFrame) -> None:
    raise NotImplementedError


if __name__ == '__main__':
    dset_dirpath = './data/'
    train_filename = 'census.csv'
    test_filename = 'adult.test.csv'
    train_imputed_filename = 'census_imputed.csv'
    test_imputed_filename = 'adult.test_imputed.csv'

    target = 'income'

    # Paths
    train_path = os.path.join(dset_dirpath, train_filename)
    test_path = os.path.join(dset_dirpath, test_filename)
    train_imputed_path = os.path.join(dset_dirpath, train_imputed_filename)
    test_imputed_path = os.path.join(dset_dirpath, test_imputed_filename)

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
