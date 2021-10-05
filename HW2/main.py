import os
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from HW2.cleaner import onehot_encoder, binary_encoder, standardize
from HW2.linear_model import LinearRegression
from HW2.util import split_xy, mpl_default_setting


def dataset_preprocess(train_path: str, test_path: str,
                       train_clean_path: str, test_clean_path: str,
                       target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""Encode binary and categorical features in the datasets and save to file

    Args:
        train_path: str
        test_path: str
        train_clean_path: str
        test_clean_path: str
        target: str

    Returns:
        train_clean: pd.DataFrame
        test_clean: pd.DataFrame
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train, y_train = split_xy(train, target)
    X_test, y_test = split_xy(test, target)

    X_train_encoded = onehot_encoder(binary_encoder(X_train, ['Urban', 'US']), 'ShelveLoc')
    X_test_encoded = onehot_encoder(binary_encoder(X_test, ['Urban', 'US']), 'ShelveLoc')
    X_train_clean, X_test_clean = standardize(X_train_encoded, X_test_encoded)

    train_clean = pd.concat([y_train, X_train_clean], axis=1)
    test_clean = pd.concat([y_test, X_test_clean], axis=1)

    train_clean.to_csv(train_clean_path, index=False)
    test_clean.to_csv(test_clean_path, index=False)

    return train_clean, test_clean


def plot_loss_curve(X: pd.DataFrame, y: pd.Series,
                    reg: str = None, pen: float = None,
                    lr: float = 0.01, n_epoch: int = 50):
    r"""Plot the training loss curve

    Args:
        X: pd.DataFrame
        y: pd.Series
        reg: str
            Type of regularization, can be None, 'l1', 'l2'
        pen: float
            Penalty (lambda), must be positive for L1 and L2 regularization
        lr: float
            Learning rate (eta), must be positive
        n_epoch: int
            Number of epochs, must be a positive integer

    Returns:
        None
    """
    # Train a linear regression model and compute losses
    mod = LinearRegression(reg, pen)
    losses = mod.train(X, y, lr, n_epoch)
    losses = np.square(losses)
    # Plot the loss curve
    mpl_default_setting()
    # Plot options
    title = r"Training loss vs $n_{data}$"
    xlabel = r"$Step = i$"
    ylabel = r"$Loss= (\hat{y}^{(i)} - y^{(i)}) ^ {2} $"
    n_locator = 5
    # Plot
    fig, ax = plt.subplots()
    ax.plot(losses)
    # Title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator(n_locator))
    # Textbox
    epoch_str = f"{n_epoch} epochs"
    lr_str = r"$\eta = {lr}$".format(lr=lr)
    if reg is None:
        reg_str = 'no regularization'
    else:
        reg_str = '\n'.join((r"$L_{n}$ regularization".format(n=reg[-1]),
                             r"$\lambda = {pen}$".format(pen=pen)))
    param_str = '\n'.join((epoch_str, lr_str, reg_str))
    bbox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.70, 0.95, param_str, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=bbox_props)
    # Show plot
    plt.show()


def training(train_path: str, target: str):
    r"""Plot training curves for Q5.5

    Args:
        train_path: str
        target: str

    Returns:

    """
    train = pd.read_csv(train_path)
    X_train, y_train = split_xy(train, target)
    # 5.5.1
    plot_loss_curve(X_train, y_train, reg=None, pen=None, lr=0.01, n_epoch=50)
    # 5.5.2
    plot_loss_curve(X_train, y_train, reg=None, pen=None, lr=0.001, n_epoch=50)
    # 5.5.3
    plot_loss_curve(X_train, y_train, reg='l2', pen=0.1, lr=0.001, n_epoch=50)
    # 5.5.4
    plot_loss_curve(X_train, y_train, reg='l1', pen=0.1, lr=0.001, n_epoch=50)


def print_test_loss(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series,
                    reg: str = None, pen: float = None,
                    lr: float = 0.01, n_epoch: int = 50):
    r"""Print test loss

    Args:
        X_train: pd.DataFrame
        y_train: pd.Series
        X_test: pd.DataFrame
        y_test: pd.Series
        reg: str
            Type of regularization, can be None, 'l1', 'l2'
        pen: float
            Penalty (lambda), must be positive for L1 and L2 regularization
        lr: float
            Learning rate (eta), must be positive
        n_epoch: int
            Number of epochs, must be a positive integer

    Returns:
        None
    """
    mod = LinearRegression(reg, pen)
    mod.train(X_train, y_train, lr, n_epoch)
    losses = mod.loss(X_test, y_test)
    test_loss = np.square(losses).sum() / len(losses)
    # Print test loss
    epoch_str = f"{n_epoch} epochs"
    lr_str = f"lr={lr}"
    if reg is None:
        reg_str = 'no regularization'
    else:
        reg_str = f"{reg} regularization, penalty={pen}"
    param_str = ', '.join((epoch_str, lr_str, reg_str))
    print('Test loss'.center(50, '='))
    print(param_str)
    print(f"{test_loss:.5f}")
    print(''.center(50, '='))


def evaluation(train_path: str, test_path: str, target: str):
    r"""Evaluate model and print test losses for Q5.6

    Args:
        train_path: str
        test_path: str
        target: str

    Returns:
        None
    """
    # Load and split datasets
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train, y_train = split_xy(train, target)
    X_test, y_test = split_xy(test, target)
    # 5.6.1
    print_test_loss(X_train, y_train, X_test, y_test,
                    reg=None, pen=None, lr=0.01, n_epoch=50)
    # 5.6.2
    print_test_loss(X_train, y_train, X_test, y_test,
                    reg=None, pen=None, lr=0.001, n_epoch=50)
    # 5.6.3
    print_test_loss(X_train, y_train, X_test, y_test,
                    reg='l2', pen=0.1, lr=0.001, n_epoch=50)
    # 5.6.4
    print_test_loss(X_train, y_train, X_test, y_test,
                    reg='l1', pen=0.1, lr=0.001, n_epoch=50)


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

    # 5.2
    train_clean, test_clean = dataset_preprocess(train_path, test_path,
                                                 train_clean_path, test_clean_path,
                                                 target)

    # 5.5
    training(train_clean_path, target)

    # 5.6
    evaluation(train_clean_path, test_clean_path, target)
