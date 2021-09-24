from __future__ import annotations

import os
from time import time
from typing import Tuple

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


def split_nan(dset: pd.DataFrame, ignore_index=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dset_clean = dset.dropna(axis=0, how='any')
    dset_nan = dset[dset.isna().any(axis=1)]
    if ignore_index:
        dset_clean.reset_index(inplace=True)
        dset_nan.reset_index(inplace=True)
    return dset_clean, dset_nan


def split_xy(dset: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    _dset = dset.copy()
    X = _dset.drop(columns=target)
    y = _dset[target]
    return X, y
