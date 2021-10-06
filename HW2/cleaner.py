import os
from typing import Tuple

import numpy as np
import pandas as pd

from HW2.util import split_xy


def binary_encoder(dset: pd.DataFrame, features: list, to_replace: dict = 'YN') -> pd.DataFrame:
    r"""Replace binary features with numbers

    Args:
        dset: pd.DataFrame
        features: list
            Names of binary features
        to_replace: dict
            Replacement rule
    Returns:
        pd.DataFrame
    """
    if to_replace == 'YN':
        _to_replace = {col: {'Yes': 1, 'No': 0} for col in features}
    else:
        _to_replace = to_replace
    return dset.replace(_to_replace)


def onehot_encoder(dset: pd.DataFrame, feature: str, sep: str = '') -> pd.DataFrame:
    r"""One-hot encoder of categorical features

    Args:
        dset: pd.DataFrame
        feature: str
            Name of the categorical feature
        sep: str
            Separator between feature name and feature value
            E.g.
            For
                feature name: 'ShelveLoc'
                feature value: 'Good'
                sep: ''
            The name of the one-hot encoded feature column will be
                'ShelveLocGood'

    Returns:
        pd.DataFrame
    """
    feature_vals = dset[feature].value_counts().sort_index().index
    cat_cols = [sep.join([feature, val]) for val in feature_vals]
    # Split the dataset by columns
    dset_before_feature = dset.loc[:, :feature].drop(columns=feature)
    dset_after_feature = dset.loc[:, feature:].drop(columns=feature)
    # Duplicate the feature column
    dset_feature = pd.concat([dset[feature]] * len(cat_cols), axis=1)
    dset_feature.index = dset.index
    dset_feature.columns = cat_cols
    # One-hot encoding
    to_replace = {}
    for val, col in zip(feature_vals, cat_cols):
        col_to_replace = {v: 0 for v in feature_vals}
        col_to_replace[val] = 1
        to_replace[col] = col_to_replace
    dset_feature.replace(to_replace, inplace=True)
    # Concatenate dataset splits
    dset_encoded = pd.concat([dset_before_feature, dset_feature, dset_after_feature], axis=1)
    return dset_encoded


def standardize(train: pd.DataFrame, test: pd.DataFrame,
                ignore_cols: pd.Index = pd.Index([]), ddof: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""Standardize features using training data

    Args:
        train: pd.DataFrame
        test: pd.DataFrame
        ignore_cols: pd.Index
            Columns to be ignored during standardization
        ddof: int
            Degree of freedom

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]
    """
    mu = train.mean()
    sigma = train.std(ddof=ddof)
    # Ignore columns
    mu[ignore_cols] = 0
    sigma[ignore_cols] = 1
    # Standardize
    train_std = (train - mu) / sigma
    test_std = (test - mu) / sigma
    return train_std, test_std


if __name__ == '__main__':
    dset_dirpath = './data/'
    train_filename = 'carseats_train.csv'
    test_filename = 'carseats_test.csv'
    train_clean_filename = 'carseats_clean_train.csv'
    test_clean_filename = 'carseats_clean_test.csv'

    target = 'Sales'

    train_path = os.path.join(dset_dirpath, train_filename)
    test_path = os.path.join(dset_dirpath, test_filename)
    train_clean_path = os.path.join(dset_dirpath, train_clean_filename)
    test_clean_path = os.path.join(dset_dirpath, test_clean_filename)

    # Load datasets
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    # Split features and targets
    X_train, y_train = split_xy(train, target)
    X_test, y_test = split_xy(test, target)
    # Encode binary and categorical features
    X_train_encoded = onehot_encoder(binary_encoder(X_train, ['Urban', 'US']), 'ShelveLoc')
    X_test_encoded = onehot_encoder(binary_encoder(X_test, ['Urban', 'US']), 'ShelveLoc')
    X_train_clean, X_test_clean = standardize(X_train_encoded, X_test_encoded,
                                              ignore_cols=pd.Index(['ShelveLocBad',
                                                                    'ShelveLocGood',
                                                                    'ShelveLocMedium',
                                                                    'Urban',
                                                                    'US']))
    # Combine encoded features with targets
    train_clean = pd.concat([y_train, X_train_clean], axis=1)
    test_clean = pd.concat([y_test, X_test_clean], axis=1)
    # Save encoded dataset to file
    train_clean.to_csv(train_clean_path, index=False)
    test_clean.to_csv(test_clean_path, index=False)
