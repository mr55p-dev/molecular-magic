"""
Utilities for aggregating `MoleculeData` into histograms and
creating vectors from those histograms
"""
from collections import defaultdict
from functools import partial
from typing import Any, Callable, TypeVar
from tqdm import tqdm
from molmagic.graphing import get_plot_name
from molmagic.vectorizer import HistogramData, MoleculeData
from molmagic.config import aggregation as cfg
from scipy.stats import gaussian_kde
import numpy as np
from molmagic.graphing import draw_and_save_hist


# Get config vars
resolution = cfg["resolution"]
bandwidth = cfg["bandwidth"]

TypeBinDict = TypeVar("TypeBinDict", bound=dict[tuple[int], list[float]])
TypeAggregate = TypeVar("TypeAggregate", bound=dict[tuple[int], list[float]])


def bin_mols(
    molecules: list[MoleculeData],
    data: dict[str, TypeBinDict],
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Load a representaiton based on molecule instances, pre-computed bins and the
    features used.

    This is not guaranteed to fail if incorrect information is used, as there is no way
    to validate the features used so beware."""
    # Get the static parts
    static_representation, target_vector = _make_static_parts(molecules, metadata)

    # Assign bins based on the data loaded
    binned_features = [
        _assign_feature_bins(data[feature], _get_feature_data(molecules, feature))
        for feature in metadata["feature-types"]
    ]
    hist_vectors = np.concatenate(binned_features, axis=1)

    # put it all together
    feature_vector = np.concatenate([static_representation, hist_vectors], axis=1)

    return feature_vector, target_vector


def autobin_mols(
    molecules: list[MoleculeData], plot_histograms: bool = False
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Take an iterator of molecules, compute histograms based on the properties
    specified in config.yml and bin the features in each molecule

    ::args::
        mols : list(like)<MoleculeData>
            List or iterable of MoleculeData objects, one for each item in the dataset
        plot_histograms : bool (default False)
            Whether to save the histograms generated
        bin_output : Path?
            Where to output the data for reconstructing this representation.
    ::returns::
        feature_vector: ndarray
            Features
        target_vector: ndarray
            Labels
        metadata: dict<str, Any>:
            Metadata for reconstructing the representation. Object contains two keys:
            - "metadata": the relevant portions of the configuration which contributed to
                the generation of these features
            - "data": The computed bin boundaries for every type of every feature
                - "feature-name":
                    - tuple[type]: list<float>
            Note if this information is to be serialised then it **must** be done so in a
            way which conserves the order of the keys, as this is crucial to how the
            representation is constructed and shuffling it destroys everything


    """
    # Get the static part of the representation
    static_representation, target_vector = _make_static_parts(molecules, cfg)

    # Get the histogam vectors. The features are defined in config
    accounted_features = cfg["feature-types"]
    hist_vector_list, type_bins = zip(
        *[
            autobin_feature(
                molecules,
                feature,
                graphing_callback=draw_and_save_hist if plot_histograms else None,
            )
            for feature in tqdm(
                accounted_features,
                leave=False,
                desc="Histogramming",
            )
        ]
    )  # zip converts from a list of pairs into a pair of lists

    # Save the type bins
    metadata = {
        "metadata": cfg,
        "data": {k: v for k, v in zip(accounted_features, type_bins)},
    }

    # Concatenate the feature vectors
    hist_vectors = np.concatenate(
        hist_vector_list,
        axis=1,
    )

    # Concatenate all the vectors
    feature_vector = np.concatenate((static_representation, hist_vectors), axis=1)
    return feature_vector, target_vector, metadata


def autobin_feature(
    mol_data: list[MoleculeData], feature: str, graphing_callback: Callable = None
) -> tuple[np.ndarray, TypeBinDict]:
    """Takes a dataset of molecules and a property to calculate and computes
    histograms for every type of interaction (ie for bonds CC, CH) and then
    bins every instance of each type in the dataset into a fixed-length vector
    determined by the number of bins on each histogram for the property.

    ::args::
        mol_data: list[MoleculeData]
            Dataset of molecules with data extracted into the MoleculeData class
        feature: str
            Method of MoleculeData which points to a defaultdict where each key is a
            type of the feature (ie CC, CH for bonds) and each value is a list of
            values for that feature in the molecule
    ::returns::
        vector: np.ndarray
            NxM vector where N is the number of instances in the molecules argument
            and M is determined by the number of features extracted and number of bins
            for each feature extracted.
            ie if we are looking at bonds, if the dataset contains CC, CH and CN bonds
            there will be a histogram created for each of these. From the minima of the
            KDE calculated for that histogram, there will be n+1 bins, where n is the
            number of minima. So the size of M in the returned vector will be
            n_bins_CC + n_bins_CH + n_bins_CN, where each of those values are determined
            from the dataset itself.
    """
    # Extract the property we want (this could be faster by accessing
    # the object dict directly)
    feature_data = _get_feature_data(mol_data, feature)

    # Get every key from every dict in the molecules data and select
    # only unique ones
    feature_types = set(i for d in feature_data for i in d)

    # Extract the per-type values for each instance
    aggregate_data: HistogramData = defaultdict(list)
    for feature_type in tqdm(feature_types, leave=False, desc="Aggregating type data"):
        for instance in feature_data:
            aggregate_data[feature_type].append(instance[feature_type])

    # Compute the bins for every type
    type_bins = _get_feature_bins(feature, graphing_callback, aggregate_data)

    # Go over the type bins and get the vectors for each molecule
    return _assign_feature_bins(type_bins, feature_data), type_bins


def _make_static_parts(
    mols: list[MoleculeData], config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Deals with creating fixed parts of representations that dont require binning"""
    # Get target vector. This should be encoded in the SDF archive in
    # the first step
    moldata = [
        (float(i.data["scf_energy"]), float(i.data["free_energy"])) for i in mols
    ]
    target_vector = np.array(moldata).astype(np.float32)

    # Get the atom count vectors. The atoms used are defined in config
    accounted_atom_types = config["atom-types"]
    atom_vectors = np.array(
        [[i.atoms[atom] for i in mols] for atom in accounted_atom_types]
    ).T

    # Get the amine count vectors. The degrees are defined in config
    accounted_amine_degrees = config["amine-types"]
    amine_vectors = np.array(
        [[i.amines[amine] for i in mols] for amine in accounted_amine_degrees]
    ).T

    # Add in any other calculated frequencies
    structure_vectors = np.array([i.structures for i in mols])

    return (
        np.concatenate((atom_vectors, amine_vectors, structure_vectors), axis=1),
        target_vector,
    )


def _get_feature_bins(
    feature: str,
    graphing_callback: Callable,
    aggregate_data: TypeAggregate,
) -> TypeBinDict:
    """Compute the bins for each feature based on aggregate_data, a dict specifying
    each type as a key and each value as a list of all the occurences in the dataset
    ::args::
        feature: str
            The name of the feature, used if the graphing callback is specified
        graphing_callback : callable
            Function to call which generates and saves a plot of the histogram
        aggregate_data : dict[type_key, list[float]]
            Every occurence of each type in the dataset
    """
    # Extract this into a new function too
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
        type_bins[feature_type] = _get_type_bins(
            flat_values,
            get_plot_name(feature, feature_type),
            graphing_callback=graphing_callback,
        )

    return type_bins


def _assign_feature_bins(
    type_bins: TypeBinDict, feature_data: list[HistogramData]
) -> np.ndarray:
    """Take a series of bins and a series of molecules and assign them to their correct bins,
    producing a fixed-length vector for each molecule
    ::args::
        type_bins : dict[type_key, list[bin_boundaries]]
            The dictionary mapping each type key to the boundaries it has (computed by )
    ::returns::
    """
    bin_col_vecs = []
    for bin_key, bin_values in tqdm(
        type_bins.items(), leave=False, desc="Assigning bins"
    ):
        f = partial(_assign_type_bins, bins=bin_values)
        vecs = map(f, (i[bin_key] for i in feature_data))

        # Append this columns data (which is a N x (M+1) vector)
        # where N is the number of molecules and M is the number
        # of bins
        bin_col_vecs.append(np.stack(list(vecs)))

    # Join the column vectors to get a feature matrix for all of the
    # types within this feature
    return np.concatenate(bin_col_vecs, axis=1)


def _get_type_bins(
    data: np.ndarray, name: tuple[str], graphing_callback: Callable = None
) -> list[float]:
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

    # Original method samples the kde and computes derivatives
    # There is an option to use the MeanShift algorithm instead

    # Find the lower and upper bounds of the data to be analysed
    # note that data should be a 1-dimensional array
    if cfg["use-minmax"]:
        lower_bound = data.min()
        upper_bound = data.max()
    else:
        assert name
        bond_feature = name[0] in ["bonds", "hbonds"]
        lower_bound = 0.8 if bond_feature else 0
        upper_bound = 3.0 if bond_feature else 200

    # Create a uniformly spaced set of samples between the minimum and maximum datapoint
    # 10000 samples is the number used in MolE8
    # Some further analysis can be done to see if this is sufficient
    sample_space = np.linspace(lower_bound, upper_bound, resolution)

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
    bins = sample_space[1:-1][is_minima]

    # Minima define the boundary of bins
    # Define additional bins at the lower and upper bounds of the data (optional)
    # Ensure the bins are monotonic and increasing
    if bins.shape[0] == 0:
        bins = np.array([-np.inf, np.inf])

    if graphing_callback and name:
        graphing_callback(data, (sample_space, sample_values), bins, name)

    return bins.tolist()


def _assign_type_bins(data: np.ndarray, bins: np.ndarray) -> np.ndarray:
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

    # TO-DO: This may be able to be improved using counter or list comprehension

    return vec


def _get_feature_data(mols: list[MoleculeData], feature: str) -> list[Any]:
    """Extract a single attribute (feature) from the MoleculeData class.
    Usually used to get a list of HistogramData instances with a smaller footprint"""
    return [getattr(i, feature) for i in mols]
