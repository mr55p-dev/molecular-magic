###################################################
# Imports and initialisation                      #
###################################################

import numpy as np
import tensorflow as tf
import optuna as op

from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split


def generate_data(batch_size, epochs, random_seed):

    tf.random.set_seed(random_seed)
    gpus = tf.config.list_logical_devices("GPU")

    ###################################################
    # Load and transform the data                     #
    ###################################################

    atom_order = ["C", "H", "N", "O"]

    # Load the data
    data_basepath = Path("static_data/create_features_output/data")
    X = np.load(data_basepath / "features.npy", allow_pickle=True)
    y = np.load(data_basepath / "labels.npy", allow_pickle=True)

    # Compute the molecular formulae for each molecule
    def create_mol_formula(atom_vec) -> str:
        return "".join(
            map(lambda x: f"{x}{atom_vec[atom_order.index(x)]}", ["C", "N", "O", "H"])
        )

    atom_vector = (X[:, -15:-11] / 100).astype("int")
    molecular_formulae = list(map(create_mol_formula, atom_vector))
    unique_molecular_formulae = set(molecular_formulae)

    # Group by formula
    X_dict = defaultdict(list)
    y_dict = defaultdict(list)
    for form, vec, target in zip(molecular_formulae, X, y):
        X_dict[form].append(vec)
        y_dict[form].append(target)

    # Put this back into a standard dictionary
    X_dict = dict(**X_dict)
    y_dict = dict(**y_dict)

    # Split the data into train and test set
    dropped_structures = []
    train_items = []
    test_items = []
    for formula in X_dict:
        if len((candidates := X_dict[formula])) == 1:
            # Discard structure
            dropped_structures.append(formula)
        elif len(candidates) == 2:
            # Assign one each to train and test
            train_candidate, test_candidate = candidates
            train_energy, test_energy = y_dict[formula]

            train_items.append((train_candidate, train_energy))
            test_items.append((test_candidate, test_energy))
        else:
            s_train, s_test, e_train, e_test = train_test_split(
                candidates, y_dict[formula], test_size=0.33, random_state=random_seed
            )
            train_items += zip(s_train, e_train)
            test_items += zip(s_test, e_test)

    # Create the train set
    train_features, train_labels = zip(*train_items)
    X_train = np.array(train_features)
    y_train = np.array(train_labels)

    # Create the test set
    test_features, test_labels = zip(*test_items)
    X_test = np.array(test_features)
    y_test = np.array(test_labels)

    # Rescale the features such that they have variance 0
    # Note they do not scale the mean to zero, only the variation about the mean to 1
    X_train_norm = X_train / X_train.std()
    X_test_norm = X_test / X_train.std()


    ###################################################
    # Experimental setup                              #
    ###################################################

    # Define save locations
    experiment_basepath = Path("experiments/")
    experiment_basepath.mkdir(parents=True, exist_ok=True)

    # Define batch size and number of epochs
    # batch_size = 64
    # epochs = 5

    # Create tf.data.Dataset
    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.convert_to_tensor(X_train_norm, dtype=tf.float32),
                tf.convert_to_tensor(y_train, dtype=tf.float32),
            )
        )
        .batch(batch_size=batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.convert_to_tensor(X_test_norm, dtype=tf.float32),
                tf.convert_to_tensor(y_test, dtype=tf.float32),
            )
        )
        .batch(batch_size=batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, test_ds, X_train.shape[1]

# generate_data(10, 10)