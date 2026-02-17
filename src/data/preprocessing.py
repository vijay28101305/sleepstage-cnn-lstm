# src/data/preprocessing.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Set
import os
import numpy as np


@dataclass
class SplitConfig:
    seed: int = 42
    train_ratio: float = 0.8


def split_subjects(psg_files: List[str], cfg: SplitConfig) -> 
Tuple[Set[str], Set[str]]:
    """
    Split subjects into train/test based on subject IDs derived from psg 
file names.
    Assumes subject id is first 7 characters of filename (as in your 
notebook).
    """
    subjects = sorted(list(set([f[:7] for f in psg_files])))

    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(subjects)

    split_idx = int(cfg.train_ratio * len(subjects))
    train_subj = set(subjects[:split_idx])
    test_subj = set(subjects[split_idx:])

    return train_subj, test_subj


def build_dataset_from_files(
    psg_files: List[str],
    hyp_files: List[str],
    cassette_dir: str,
    train_subj: Set[str],
    load_sleep_record_fn,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds X_train, y_train, X_test, y_test by loading each subject record 
and concatenating epochs.

    Parameters
    ----------
    psg_files : list of PSG filenames (strings)
    hyp_files : list of hypnogram filenames (strings)
    cassette_dir : directory containing the files
    train_subj : set of subject IDs assigned to train split
    load_sleep_record_fn : function(psg_path, hyp_path) -> (X, y, sf)

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for psg_file in psg_files:
        subj_id = psg_file[:7]
        matching_hyps = [h for h in hyp_files if h.startswith(subj_id)]
        if len(matching_hyps) == 0:
            # skip if missing hypnogram
            continue

        hyp_file = matching_hyps[0]
        psg_path = os.path.join(cassette_dir, psg_file)
        hyp_path = os.path.join(cassette_dir, hyp_file)

        X, y, sf = load_sleep_record_fn(psg_path, hyp_path)

        if len(X) == 0:
            continue

        if subj_id in train_subj:
            X_train_list.append(X)
            y_train_list.append(y)
        else:
            X_test_list.append(X)
            y_test_list.append(y)

    if len(X_train_list) == 0 or len(X_test_list) == 0:
        raise ValueError("Train/Test lists are empty. Check file lists and 
subject split.")

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    return X_train, y_train, X_test, y_test

