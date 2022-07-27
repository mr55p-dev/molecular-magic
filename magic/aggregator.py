"""
Utilities for aggregating `MoleculeData` into histograms and
creating vectors from those histograms
"""
from collections import defaultdict
from functools import partial
from itertools import zip_longest
import itertools
from pathlib import Path
from typing import TypeVar, Union
from openbabel import pybel as pb
from tqdm import tqdm
from magic.parser import read_sdf_archive
from magic.vectorizer import HistogramData, calculate_mol_data, MoleculeData
from magic.config import aggregation as cfg
from scipy.stats import gaussian_kde
import numpy as np


bandwidth = cfg["kde-bandwidth"]
resolution = cfg["resolution"]
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


def data_to_bins(data: np.ndarray) -> np.ndarray:
    """Assign each point in data to a bin where the bin edges are
    assigned based on the kde minima of a histogram generated from data.

    Uses the Gaussian KDE and samples that function to determine the
    location of maxima and minima. From this, compute the bin edges based
    on the location of minima in the dataset. Finally, assign each point
    in data to one of the computed bins.

    ::args::
        mol_data: list[ndarray]
            List of 1-dimensional numpy arrays corresponding to one array
            for each molecule. These arrays are joined to a single continuous
            list for analysis
    ::returns::
        bins: ndarray
            The boundaries of the histogram bins
    """
    # Calculate the KDE
    kde = gaussian_kde(data, bw_method=bandwidth)

    # Original method samples the kde and computes derrivitaves
    # There is an option to use the MeanShift algorithm instead

    # Find the lower and upper bounds of the data to be analysed
    # note that data should be a 1-dimensional array
    lower_bound = data.min()
    upper_bound = data.max()

    # Create a linear sample space from this data range
    # 10000 samples is the number used in MolE8
    # Some further analysis can be done to see if this is sufficient
    n_samples = int(resolution * (upper_bound - lower_bound))
    sample_space = np.linspace(lower_bound, upper_bound, n_samples)

    # Compute the value of the kde at each point in the sample space
    sample_values = kde.evaluate(sample_space)

    # Find the minima
    """
    [0, 1, 2, 3, 4, 5, 6, 7] <= shifted left
       [0, 1, 2, 3, 4, 5, 6, 7] <= original
          [0, 1, 2, 3, 4, 5, 6, 7] <= shifted right

    [2, 3, 4, 5, 6, 7] <= truncated left
    [1, 2, 3, 4, 5, 6] <= truncated original
    [0, 1, 2, 3, 4, 5] <= truncated right

    At a minima, original is less than left and right
    """

    minima_left = sample_values[1:-1] < sample_values[2:]
    minima_right = sample_values[1:-1] < sample_values[:-2]

    # Minima are where the sample is smaller than left and right shifted values
    # is_maxima = np.logical_and(sample_values > shift_left, sample_values > shift_right)
    is_minima = np.logical_and(minima_left, minima_right)

    # Compute the values at these points
    minima = sample_space[1:-1][is_minima]

    # Minima define the boundary of bins
    # Define additional bins at the lower and upper bounds of the data (optional)
    # Ensure the bins are monotonic and increasing
    bins = np.concatenate(
        (np.expand_dims(lower_bound, 0), minima, np.expand_dims(upper_bound, 0))
    )

    return bins


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


def compute_histogram_vectors(molecules: list[MoleculeData], feature: str) -> np.array:
    # Extract the property we want (this could be faster by accessing
    # the object dict directly)
    feature_data = [getattr(i, feature) for i in molecules]

    # Get every key from every dict in the molecules data and select
    # only unique ones
    feature_types = set(i for d in feature_data for i in d)

    # Extract the per-type values for each instance
    aggregate_data: HistogramData = defaultdict(list)
    for feature_type in tqdm(feature_types, leave=False, desc="Aggregating type data"):
        for instance in feature_data:
            aggregate_data[feature_type].append(instance[feature_type])

    # Calcuate the bin boundaries for each type
    type_bins = {}
    for feature_type, values in tqdm(
        aggregate_data.items(), leave=False, desc="Calculating bins"
    ):
        # Define fallback behaviour if there are fewer than N instances of a type
        # in the dataset. For now, just ignore it
        if len([i for j in values for i in j]) < 2:
            continue
        flat_values = np.concatenate(values).ravel()
        type_bins[feature_type] = data_to_bins(flat_values)

    # Go over the type bins and get the vectors for each molecule
    bin_col_vecs = []
    for bin_key, bin_values in tqdm(
        type_bins.items(), leave=False, desc="Assigning bins"
    ):
        f = partial(assign_bin, bins=bin_values)
        vecs = map(f, (i[bin_key] for i in feature_data))

        # Append this columns data (which is a N x (M+1) vector)
        # where N is the number of molecules and M is the number
        # of bins
        bin_col_vecs.append(np.stack(list(vecs)))

    # Join the column vectors to get a feature matrix for all of the
    # types within this feature
    return np.concatenate(bin_col_vecs, axis=1)


if __name__ == "__main__":
    # Get our molecule set
    mols = read_sdf_archive(
        Path("./test.sdf.bz2")
    )

    # Extract the molecular properties
    mols = list(
        tqdm(
            map(calculate_mol_data, mols),
            leave=False,
            desc="Extracting molecular properties",
        )
    )

    # Get the energy vector
    energy_vector = np.array([i.energy for i in mols])

    # Get the atom count vectors
    atom_vectors = np.array(
        [
            [i.atoms[atom] for i in mols]
            for atom in (1, 6, 7, 8)  # For hydrogen carbon, nitrogen, oxygen
        ]
    ).T

    # Get the amine count vectors
    amine_vectors = np.array(
        [
            [i.amines[amine] for i in mols]
            for amine in (1, 2, 3)  # For degree 1, 2, 3 amines
        ]
    ).T

    # Get the histogam vectors
    hist_data = np.concatenate(
        [
            compute_histogram_vectors(mols, feature)
            for feature in tqdm(
                ["bonds", "angles", "dihedrals", "hbonds"],
                leave=False,
                desc="Histogramming",
            )
        ],
        axis=1,
    )

    feature_vector = np.concatenate((atom_vectors, amine_vectors, hist_data), axis=1)

    np.save(Path("./test"), feature_vector)
    np.save(Path("./test_y"), energy_vector)
