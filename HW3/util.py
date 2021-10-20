from __future__ import annotations

import itertools
from time import time
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Timer:
    def __init__(self, prompt: str = None) -> None:
        r"""Print the wall time of the execution of a code block

        Args:
            prompt (str): Specify the prompt in the output. Optional.

        Returns:
            None

        Examples:
            >>> from HW1.util import Timer
            >>> with Timer("TEST"):
            ...     # CODE_BLOCK
            Wall time of TEST: 0.0 seconds
            >>> from HW1.util import Timer
            >>> with Timer():
            ...     # CODE_BLOCK
            Wall time: 0.0 seconds
        """
        self.prompt = prompt

    def __enter__(self) -> Timer:
        self.start = time()  # <-- Record the time when Timer is called
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end = time()  # <-- Record the time when Timer exits
        self.seconds = self.end - self.start  # <-- Compute the time interval
        if self.prompt:
            print(f"Wall time of {self.prompt}: {self.seconds:.1f} seconds")
        else:
            print(f"Wall time: {self.seconds:.1f} seconds")


def mpl_default_setting():
    r"""Default matplotlib setting for plot formats

    Returns:
        None

    """
    mpl.rcParams['axes.labelsize'] = 'large'

    mpl.rcParams['figure.figsize'] = (8, 6)
    mpl.rcParams['figure.subplot.hspace'] = 0.3
    mpl.rcParams['figure.subplot.wspace'] = 0.3
    mpl.rcParams['figure.titlesize'] = 'large'

    mpl.rcParams['font.family'] = ['Arial']
    mpl.rcParams['font.size'] = 16

    mpl.rcParams['legend.fontsize'] = 'small'
    mpl.rcParams['legend.loc'] = 'upper right'

    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['xtick.labelsize'] = 'large'
    mpl.rcParams['xtick.major.size'] = 10
    mpl.rcParams['xtick.minor.size'] = 5
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['ytick.labelsize'] = 'large'
    mpl.rcParams['ytick.major.size'] = 10
    mpl.rcParams['ytick.minor.size'] = 5

    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.format'] = 'png'
    mpl.rcParams['savefig.transparent'] = False
    mpl.rcParams['savefig.bbox'] = 'tight'


def plot_image(vector, title=None):
    """
    Takes a vector as input of size (784) and saves as an image
    """
    image = np.asarray(vector).reshape(28, 28)
    plt.imshow(image, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='Blues'):
    r"""Plot confusion matrix

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    References:
        https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_diff(x: np.ndarray, x_ref: np.ndarray,
              tol: float = 1e-14, **kwargs):
    n_bins = kwargs.get('n_bins', 100)
    xlabel = kwargs.get('xlabel', r'$\log_{10}|x - x_{ref}|$')
    ylabel = kwargs.get('ylabel', 'Counts')
    title = kwargs.get('title', r'Distribution of $\log_{10}|x - x_{ref}|$')
    diff = np.abs(x - x_ref)
    log10_diff = np.log10(diff[diff >= tol])
    # Plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.hist(log10_diff, n_bins)
    plt.show()


def split_xy(dset: pd.DataFrame, target) -> Tuple[pd.DataFrame, pd.Series]:
    r"""Split dataset into a feature matrix and a label vector

    Args:
        dset: pd.DataFrame
        target:
            Name of the label

    Returns:
        X, y
    """
    _dset = dset.copy()
    if isinstance(target, int):
        _target = target if target >= 0 else dset.shape[1] + target
    else:
        _target = target
    X = _dset.drop(columns=_target)
    y = _dset[_target]
    return X, y
