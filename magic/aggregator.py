"""
Utilities for aggregating `MoleculeData` into histograms and
creating vectors from those histograms
"""
from typing import TypeVar
from openbabel import pybel as pb
from magic.vectorizer import calculate_mol_data, MoleculeData
from magic.config import aggregation as cfg
from scipy.stats import gaussian_kde
import numpy as np


bandwidth = cfg['kde-bandwidth']
KDE = TypeVar("KDE")


def compute_kde(data: np.ndarray, bandwidth: float) -> KDE:
    # Calculate the KDE
    kde = gaussian_kde(data, bw_method=bandwidth)

    # Original method samples the kde and computes derrivitaves
    kde.evaluate()

    # There is an option to use the MeanShift algorithm instead
    





