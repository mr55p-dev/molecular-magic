# Imports and initialisation
import itertools
from argparse import ArgumentParser, Namespace
import numpy as np
import wandb
from molmagic import ml
from tqdm import tqdm, trange
from molmagic.ml import run_controller
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import AdaBoostRegressor
import tensorflow as tf


"""
MAKE SURE THAT YOU DONT MEASURE VALIDATION MAE, THAT IS POINTLESS AS THE SPLIT SIZE IS CHANGING!

Instead lets get one of the generalisation sets, probably qm9-test-HCNOF-...:latest

"""


def construct_wrapper(**kwargs):
    """Wrapper to enable constructing a model from its wandb parameters"""
    random_seed = kwargs.get("seed", 50)
    algorithm = kwargs["algorithm"]
    if algorithm == "Keras":
        tf.random.set_seed(random_seed)
    elif algorithm in ["Ridge", "RidgeCV"]:
        return Ridge(alpha=kwargs["ridge_alpha"], random_state=random_seed)
    elif algorithm == "Elastic":
        return ElasticNet(
            random_state=random_seed,
            max_iter=kwargs["max_iter"],
            l1_ratio=kwargs["l1_ratio"],
            alpha=kwargs["elastic_alpha"],
        )
    elif algorithm == "AdaBoost":
        return AdaBoostRegressor(
            n_estimators=kwargs["n_estimators"],
            loss_function=kwargs["loss_function"],
            alpha=kwargs["ridge_alpha"],
            random_state=random_seed,
        )
    else:
        raise NotImplementedError("Algorithm %s not implemented", algorithm)


def fit_wrapper(model, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
    """Wrapper to fit a model regardless of how its built"""
    algorithm = kwargs["algorithm"]
    if algorithm == "Keras":
        _ = model.fit(x=X_train, y=y_train, epochs=kwargs["epochs"])
        return model
    elif algorithm in ["Ridge", "RidgeCV", "Elastic", "AdaBoost"]:
        return model.fit(X_train, y_train.squeeze())
    else:
        raise NotImplementedError("Algorithm %s not implemented", algorithm)


def main(args: Namespace):
    run = run_controller.use_run("data-efficiency")
    run_basis_name = args.run

    api = wandb.Api()
    basis_run = api.run(run_basis_name)
    basis_config = basis_run.config

    # Log that this run is using the previous one
    basis_run_path = "/".join([basis_run.entity, basis_run.project, basis_run.id])
    basis_model_artifact = ml.get_artifact_of_kind(basis_run_path, "model")
    run.use_artifact(basis_model_artifact)

    # SKlearn setup
    # random_seed = basis_config["random_state"]
    n_folds = args.n_folds
    run.config.update({"n_folds": n_folds})

    # Get the parameters
    label_type = basis_run.config["label_name"]
    training_artifact = basis_run.config["training_artifact"]
    testing_artifact = "qm9-test-HCNOF-0.8-0.32:v0"

    # Dataset loading
    train_basepath = ml.get_vector_artifact(training_artifact)
    X_train = np.load(train_basepath / "features.npy")
    y_train_raw = np.load(train_basepath / "labels.npy").astype(np.double)
    y_train = ml.get_label_type(y_train_raw, label_type)

    test_basepath = ml.get_vector_artifact(testing_artifact)
    X_test = np.load(test_basepath / "features.npy")
    y_test_raw = np.load(test_basepath / "labels.npy").astype(np.double)
    y_test = ml.get_label_type(y_test_raw, label_type)

    # Setup cross validation with a range of values
    for n_groups_in_split in trange(1, n_folds + 1):
        # Do cross validation with a train/test split size == split
        maes = []
        mses = []

        X_splits = np.array_split(X_train, n_folds, axis=0)
        y_splits = np.array_split(y_train, n_folds, axis=0)
        X_combinations = itertools.combinations(X_splits, n_groups_in_split)
        y_combinations = itertools.combinations(y_splits, n_groups_in_split)

        # combination_iter = 1
        for X_group_list, y_group_list in tqdm(
            zip(X_combinations, y_combinations),
            f"Iterating all {n_groups_in_split} item combinations of the training set",
            leave=False,
        ):
            # Concatenate lists into coherent vectors
            X_group = np.concatenate(X_group_list, axis=0)
            y_group = np.concatenate(y_group_list, axis=0)

            # Initialize the model
            model = construct_wrapper(**basis_config)

            # Fit the model
            fitted_model = fit_wrapper(model, X_group, y_group, **basis_config)

            # Evaluate
            y_pred = fitted_model.predict(X_test).squeeze()
            maes.append(mean_absolute_error(y_test.squeeze(), y_pred))
            mses.append(mean_squared_error(y_test.squeeze(), y_pred))


        # Compute the mean MAE and MSE
        cross_val_mae = np.mean(maes)
        cross_val_mse = np.mean(mses)

        # Log them for this step
        wandb.log(
            {
                "val_mae": cross_val_mae,
                "val_mse": cross_val_mse,
                "train_split_size": n_groups_in_split,
            }
        )
    wandb.finish()


def cli_tool():
    parser = ArgumentParser("data-efficiency")
    parser.add_argument("run", type=str, help="Run to be analysed")
    parser.add_argument(
        "-n",
        "--n-folds",
        type=int,
        help="The number of groups to split the data into for analysis (default 10)",
        default=10,
    )
    args = parser.parse_args()
    main(args)
