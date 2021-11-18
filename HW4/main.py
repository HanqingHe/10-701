import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from HW4.adaboost import Adaboost
from HW4.util import split_xy, mpl_default_setting


def read_data(dset_path: str,
              cols: tuple = ('x1', 'x2', 'y'),
              target: str = 'y') -> Tuple[pd.DataFrame, pd.Series]:
    dset = pd.read_csv(dset_path, names=cols)
    X, y = split_xy(dset, target)
    return X, y


if __name__ == '__main__':
    dset_dirpath = './data/'
    train_filename = 'train_adaboost.csv'
    test_filename = 'test_adaboost.csv'
    mod_dirpath = './models'
    mod_filename = 'adaboost_iter_50.pickle'
    acc_filename = 'adaboost_iter_50_acc.pickle'

    train_path = os.path.join(dset_dirpath, train_filename)
    test_path = os.path.join(dset_dirpath, test_filename)
    mod_path = os.path.join(mod_dirpath, mod_filename)
    acc_path = os.path.join(mod_dirpath, acc_filename)

    # Load dataset
    X_train, y_train = read_data(train_path)
    X_test, y_test = read_data(test_path)

    # Q6.2 ~ Q6.3
    n_iter = 50
    ada = Adaboost()
    ada.train(X_train, y_train, n_iter=n_iter)
    _, accs = ada.eval_model(X_test, y_test, full=True)
    # Save model and accs to file
    pickle.dump(ada, open(mod_path, 'wb'))
    pickle.dump(accs, open(acc_path, 'wb'))
    # Plot test accuracy
    mpl_default_setting()
    plt.plot(accs.index + 1, accs, 'ro--')
    plt.title('Test accuracy vs iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Test accuracy')
    plt.show()

    # Q6.1
    ada = pickle.load(open(mod_path, 'rb'))
    accs = pickle.load(open(acc_path, 'rb'))

    # Samples
    N_train, D_train = X_train.shape
    N_sample = 20
    X_sample, y_sample = X_train.iloc[:N_sample, :], y_train.iloc[:N_sample]
    xmin, xmax = (-2, 2)
    pos_x = X_sample.loc[y_sample == 1, :]
    neg_x = X_sample.loc[y_sample == -1, :]
    assert len(pos_x) + len(neg_x) == len(X_sample)

    # Weights
    T_sample = 3
    weights = [pd.Series(np.full(N_train, 1/N_train),
                         index=y_train.index, name=f"iter 0")]
    weights.extend([ada.w[f"iter {t}"] for t in range(1, T_sample)])

    for t in range(T_sample):
        w_sample = weights[t].iloc[:N_sample]
        pos_weights = w_sample.loc[y_sample == 1]
        neg_weights = w_sample.loc[y_sample == -1]

        # Decision boundaries
        db = ada.h[t].threshold
        inverse = ada.h[t].inverse

        # Plot weighted data
        neg_size = 100 * (neg_weights ** 2) * (N_train ** 2)
        pos_size = 100 * (pos_weights ** 2) * (N_train ** 2)
        plt.figure(figsize=(6, 6))
        plt.xlim(xmin, xmax)
        plt.ylim(xmin, xmax)
        plt.scatter(neg_x.iloc[:, 0], neg_x.iloc[:, 1], s=neg_size, color='red', marker='_')
        plt.scatter(pos_x.iloc[:, 0], pos_x.iloc[:, 1], s=pos_size, color='blue', marker='+')

        plt.title('Weighted Data', fontsize=16)
        plt.legend(['-1', '+1'], fontsize=16)
        plt.xlabel('$x_1$', fontsize=16)
        plt.ylabel('$x_2$', fontsize=16)
        plt.show()

        # Plot weighted data with decision boundary
        plt.figure(figsize=(6, 6))
        plt.xlim(xmin, xmax)
        plt.ylim(xmin, xmax)
        plt.scatter(neg_x.iloc[:, 0], neg_x.iloc[:, 1], s=neg_size, color='red', marker='_')
        plt.scatter(pos_x.iloc[:, 0], pos_x.iloc[:, 1], s=pos_size, color='blue', marker='+')

        plt.plot([db, db], [xmin, xmax], color='black', label='_nolegend_')
        if inverse:
            plt.fill([xmin, xmin, db, db], [xmin, xmax, xmax, xmin], color='blue', alpha=0.1, label='_nolegend_')
            plt.fill([xmax, xmax, db, db], [xmin, xmax, xmax, xmin], color='red', alpha=0.1, label='_nolegend_')
        else:
            plt.fill([xmin, xmin, db, db], [xmin, xmax, xmax, xmin], color='red', alpha=0.1, label='_nolegend_')
            plt.fill([xmax, xmax, db, db], [xmin, xmax, xmax, xmin], color='blue', alpha=0.1, label='_nolegend_')

        plt.title('Weighted Data w/ Tree', fontsize=16)
        plt.legend(['-1', '+1'], fontsize=16)
        plt.xlabel('$x_1$', fontsize=16)
        plt.ylabel('$x_2$', fontsize=16)
        plt.show()
