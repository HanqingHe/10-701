from __future__ import annotations

from time import time
from typing import Tuple

import matplotlib as mpl
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


def split_nan(dset: pd.DataFrame, ignore_index: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""Separate rows with missing data from others

    Args:
        dset: pd.DataFrame
        ignore_index: bool

    Returns:
        dset_clean, dset_nan
    """
    dset_clean = dset.dropna(axis=0, how='any')
    dset_nan = dset[dset.isna().any(axis=1)]
    if ignore_index:
        dset_clean.reset_index(inplace=True)
        dset_nan.reset_index(inplace=True)
    return dset_clean, dset_nan


def split_xy(dset: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    r"""Split dataset into a feature matrix and a label vector

    Args:
        dset: pd.DataFrame
        target: str
            Name of the label

    Returns:
        X, y
    """
    _dset = dset.copy()
    X = _dset.drop(columns=target)
    y = _dset[target]
    return X, y
