from collections import defaultdict
from itertools import islice
from math import inf
from pathlib import Path
from magic.aggregator import assign_bin, data_to_bin
from magic.vectorizer import calculate_mol_data, _should_reverse
from magic.parser import read_sdf_archive
import pytest
import numpy as np


array_reversal_test_data = [
    ([1, 1], False),
    ([1, 0], False),
    ([0, 1], True),
    ([-1, 1], True),
    ([1, 1, 1], False),
    ([1, 7, 0], False),
    ([6, 7, 8], True),
    ([1, 7, 8, 1], True),
    ([1, 8, 7, 1], False),
    ([1, 8, 7, 9], True),
    ([7, 8, 7, 1], False),
]


def test_feature_extraction():
    mol = next(read_sdf_archive(Path("dft_test_files/output.sdf.bz2")))

    mol_data = calculate_mol_data(mol)

    assert mol_data.atoms
    assert mol_data.atoms[1] > 0  # Hydrogen
    assert mol_data.atoms[6] > 0  # Carbon
    assert mol_data.atoms[7] > 0  # Nitrogen
    assert mol_data.atoms[8] > 0  # Oxygen

    assert mol_data.amines is not None

    assert len(mol_data.bonds.keys()) > 0
    assert len(mol_data.angles.keys()) > 0
    assert len(mol_data.dihedrals.keys()) > 0
    assert len(mol_data.hbonds.keys()) >= 0
    # In the first test molecule there are no hbonding interactions


@pytest.mark.parametrize("arr,truth", array_reversal_test_data)
def test_array_reversal(arr: list[int], truth):
    """Check the array reversal function works as expected"""
    assert _should_reverse(arr) == truth


def test_bin_generation():
    # Define a test histogram
    #         .
    #         .
    # .       .     .
    # .       . . . .
    # . .       . . . . .
    # 0 1 2 3 4 5 6 7 8 9
    data = np.array([0, 1, 1, 1, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9])
    bins = data_to_bin(data)
    assert len(bins) == 4
