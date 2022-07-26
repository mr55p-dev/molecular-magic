"""
Utilities for aggregating `MoleculeData` into histograms and
creating vectors from those histograms
"""
from typing import TypeVar, Union
from openbabel import pybel as pb
from magic.vectorizer import calculate_mol_data, MoleculeData
from magic.config import aggregation as cfg
from scipy.stats import gaussian_kde
import numpy as np


bandwidth = cfg["kde-bandwidth"]
KDE = TypeVar("KDE")


def _compute_bins(sample_values: np.ndarray, method=str) -> np.ndarray:
    """Computes bins based on the sample of a KDE
    ::args::
        sample_values: ndarray
            The values sampled from the KDE
        method: "grad" | "diff" | "stationary"
            The method to use for computing maxima and minima.
            "grad" uses numpy's gradient compute feature to calculate the
            gradient. Should be more accurate at the ends of each dataset
            but is untested.
            "diff" uses the first order differences in the data which works
            well with a large number of samples in sample_values.
    ::returns::
        maxima: ndarray
            array corresponding to the value at which each bin occurs
        assignment: ndarray
            the bin assignment of each point in

    """
    # Compute the derivitave based on these samples
    if method == "grad":
        # sample_derivitave = np.grad(sample_values, 1)
        raise NotImplementedError(
            "gradient-based differentiation is not yet implemented"
        )
    elif method == "diff":
        # sample_derivitave = np.diff(sample_values, 1)
        raise NotImplementedError("derrivitave-based binning is not implemented")

    # It might be more efficient to calculate where the derivitave crossed
    # the x axis rather than this shift left and right stuff
    return ...


def data_to_bin(mol_data: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Assign each point in data to a bin where the bin edges are
    assigned based on the kde minima of a histogram generated from data.

    Uses the Gaussian KDE and samples that function to determine the
    location of maxima and minima. From this, compute the bin edges based
    on the location of minima in the dataset. Finally, assign each point
    in data to one of the computed bins.

    mol_data: list[ndarray]
        List of 1-dimensional numpy arrays corresponding to one array
        for each molecule. These arrays are joined to a single continuous
        list for analysis
    """
    # Flatten the molecule data
    data = np.concatenate(mol_data).ravel()

    # Calculate the KDE
    kde = gaussian_kde(data, bw_method=bandwidth)

    # Original method samples the kde and computes derrivitaves
    # There is an option to use the MeanShift algorithm instead

    # Find the lower and upper bounds of the data to be analysed
    # note that data should be a 1-dimensional array
    lower_bound = 0  # could also be data.min()
    upper_bound = data.max()

    # Create a linear sample space from this data range
    # 10000 samples is the number used in MolE8
    # Some further analysis can be done to see if this is sufficient
    sample_space = np.linspace(lower_bound, upper_bound, 10000)

    # Compute the value of the kde at each point in the sample space
    sample_values = kde.evaluate(sample_space)

    # This code is lifted from the original
    # maxima occur where the surrounding values are lower than the current one
    # minima occur where the surrounding values are greater than the current one
    shift_left = np.roll(sample_values, -1)
    shift_right = np.roll(sample_values, 1)

    # Compute the location of stationary points
    is_maxima = np.logical_and(sample_values > shift_left, sample_values > shift_right)
    is_minima = np.logical_and(sample_values < shift_left, sample_values < shift_right)

    # Compute the values at these points
    maxima = sample_values[is_maxima]
    minima = sample_values[is_minima]

    # Check that the number of minima and maxima is correct
    assert len(minima) + 1 == len(maxima)

    # Minima define the boundary of bins
    # Define additional bins at -ooo and +ooo
    # Ensure the bins are monotonic and increasing
    bins = sorted(minima)

    return bins, maxima


def assign_bin(data: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Assign each value to a bin and return a vector where each item
    corresponds to the number of occurences of items falling into bin n

    Note the returned matrix is of shape len(bins) + 1, as there will be
    n+1 bins defined by n boundaries (additional bins below and above the
    specified ones are valid"""
    # Using bins sorted low to high and right=False,
    # the criterion is bins[i-1] <= x < bins[i]
    binned = np.digitize(data, bins)
    vec = np.zeros((len(bins) + 1,))

    # This is slow find a better way
    for bin_idx in binned:
        vec[bin_idx] += 1

    return vec


if __name__ == "__main__":
    # Get our molecule set from somewhere
    mols: list[MoleculeData] = ...

    # Example for bonds

    # Compute bins
    bond_bins = data_to_bin([i.bonds for i in mols])
    angle_bins = data_to_bin([i.angles for i in mols])
    dihedral_bins = data_to_bin([i.dihedrals for i in mols])
    

    # Create a bond vector for each molecule
    for mol in mols:
        bond_vec = assign_bin(mol.bonds, bond_bins)
