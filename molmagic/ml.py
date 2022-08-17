from pathlib import Path
from typing import Callable
import wandb
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import shutil
import numpy as np


run = wandb.init(
    project="MolecularMagic",
    entity="molecular-magicians",
)


def get_artifact(name: str):
    """Download an artifact by name"""
    artifact = run.use_artifact(name, type="vectors")
    download_path = Path("/tmp/")
    artifact.download(download_path)

    return download_path


def log_model(model: tf.keras.Model) -> None:
    """Save a model to weights and baises as an artifact"""
    output_path = Path("tmp/") / run.name
    model.save(output_path)

    artifact = wandb.Artifact(run.name, type="model")
    artifact.add_dir(output_path, name="model")
    run.log_artifact(artifact)

    shutil.rmtree(
        output_path, onerror=lambda *_: print("Failed to clean up model upload")
    )


def get_label_type(arr: np.ndarray, label_type: str) -> np.ndarray:
    if label_type == 'electronic_energy':
        return arr[0]
    elif label_type == 'free_energy':
        return arr[1]


def get_split(type: str) -> Callable:
    if type == "stoichiometric":
        return stoichiometric_split
    elif type == "random":
        return train_test_split
    else:
        raise NotImplementedError(f"{type} not implemented")


def stoichiometric_split(
    X: np.ndarray, y: np.ndarray, random_state: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """train_test_split-like function which implements the MolE8 custom split
    ::args::
        X : ndarray (rank 2)
            Feature matrix. Must have the atom frequencies in the order
            [H, C, N, O] as the array slice [:, :4] in order to function
            properly.
        y : ndarray (rank 2)
            Corresponding label matrix
        random_state : int
            Random state to use for the internal train_test_split
    ::returns::
        X_train : ndarray (rank 2)
            Training features
        X_test : ndarray (rank 2)
            Testing features
        y_train : ndarray (rank 2)
            Training labels
        y_test : ndarray (rank 2)
            Testing labels
    """

    # Coerce each slice into a tuple of ints rather than
    # an array slice for ease of use
    # Note if more atoms are included than HCNO then we will
    # need to make this '4' more flexible
    keys = map(tuple, X[:, :4])

    # Group by formula
    X_dict = defaultdict(list)
    y_dict = defaultdict(list)
    for form, vec, target in zip(keys, X, y):
        X_dict[form].append(vec)
        y_dict[form].append(target)

    # Split the data into a train and test set
    dropped_structures = []
    train_items = []
    test_items = []
    for formula in X_dict:
        # Case: There is only one example of this stoichiometry
        if len((candidates := X_dict[formula])) == 1:
            dropped_structures.append(formula)
        # Case: There are two examples of this stoichiometry
        elif len(candidates) == 2:
            # Assign one each to train and test
            train_candidate, test_candidate = candidates
            train_target, test_target = y_dict[formula]
            train_items.append((train_candidate, train_target))
            test_items.append((test_candidate, test_target))
        # Case: There are more than two examples of this stoichiometry
        else:
            # Use train_test_split to divide them up (ironic)
            s_train, s_test, e_train, e_test = train_test_split(
                candidates, y_dict[formula], test_size=0.33, random_state=random_state
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

    print(f"Dropped {len(dropped_structures)} structures with only one occurence")
    print(f"Train set: {X_train.shape[0]}\nTest set: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test
