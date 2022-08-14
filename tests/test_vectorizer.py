import argparse
from collections import defaultdict
from itertools import islice
from math import inf
from pathlib import Path
from molmagic.aggregator import _assign_type_bins, _get_type_bins
from ..molmagic.graphing import get_plot_name
from molmagic.vectorizer import calculate_mol_data, _should_reverse
from molmagic.parser import read_sdf_archive
from molmagic.cli import vectorize
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
    assert mol_data.atoms[1] >= 0  # Hydrogen
    assert mol_data.atoms[6] >= 0  # Carbon
    assert mol_data.atoms[7] >= 0  # Nitrogen
    assert mol_data.atoms[8] >= 0  # Oxygen

    assert mol_data.amines is not None

    assert len(mol_data.bonds.keys()) >= 0
    assert len(mol_data.angles.keys()) >= 0
    assert len(mol_data.dihedrals.keys()) >= 0
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
    # . .     . . . . . .
    # 0 1 2 3 4 5 6 7 8 9
    data = np.array(
        [0] * 30
        + [1] * 10
        + [4] * 50
        + [5] * 10
        + [6] * 10
        + [7] * 30
        + [8] * 10
        + [9] * 5
    )
    bins = _get_type_bins(data, get_plot_name("angle", (1, 6, 1)))  # Simulate a plot
    assert len(bins) == 2

    instance = np.array([-1, 2.5, 5, 7, 15])
    truth = np.array([1, 2, 2])

    predicted = _assign_type_bins(instance, bins)
    assert (predicted == truth).all()


@pytest.mark.dependency(depends=["test_encode"])
def test_revec():
    test_output = Path("dft_test_files/output.sdf.bz2")
    # Create hists and bins and metadata

    vec_output = Path("dft_test_files/vec_gen/")
    args = argparse.Namespace(input=test_output, output=vec_output)
    aggregate(args)

    revec_output = Path("dft_test_files/vec_regen/")
    args_2 = argparse.Namespace(
        input=test_output,
        metadata=Path("dft_test_files/vec_gen/metadata.yaml"),
        output=revec_output,
    )
    aggregate(args_2)

    # Load the generated vectors
    vec = np.load(vec_output/"features.npy")
    lab = np.load(vec_output/"labels.npy")

    revec = np.load(revec_output/"features.npy")
    relab = np.load(revec_output/"labels.npy")

    assert (vec == revec).all()
    assert (lab == relab).all()