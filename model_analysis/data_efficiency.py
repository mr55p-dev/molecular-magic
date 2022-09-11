# Imports and initialisation
from argparse import ArgumentParser, Namespace
import numpy as np
import wandb
from molmagic import ml
from molmagic.ml import run_controller
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import AdaBoostRegressor
import tensorflow as tf


def construct_wrapper(algorithm: str, **kwargs):
    """Wrapper to enable constructing a model from its wandb parameters"""
    random_seed = kwargs.get(["seed"], 50)
    if algorithm == "Keras":
        tf.random.set_seed(random_seed)
    elif algorithm in ["Ridge", "RidgeCV"]:
        return Ridge(alpha=kwargs["ridge_alpha"], random_state=kwargs["random_seed"])
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


def fit_wrapper(
    model, algorithm: str, X_train: np.ndarray, y_train: np.ndarray, **kwargs
):
    """Wrapper to fit a model regardless of how its built"""
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
    basis_run_path = "/".join([basis_run.project, basis_run.entity, basis_run.name])
    basis_model_artifact = ml.get_artifact_of_kind(basis_run_path, "model")
    run.use_artifact(basis_model_artifact)

    # SKlearn setup
    random_seed = basis_config["random_state"]

    # Get the parameters
    algorithm = basis_run.config["algorithm"]
    label_type = basis_run.config["label_name"]
    training_artifact = basis_run.config["training_artifact"]

    # Dataset loading
    basepath = ml.get_vector_artifact(training_artifact)

    X = np.load(basepath / "features.npy")
    y_raw = np.load(basepath / "labels.npy").astype(np.double)
    y = ml.get_label_type(y_raw, label_type)

    # Setup cross validation with a range of values
    train_split_sizes = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for train_split in train_split_sizes:
        # Do cross validation with a train/test split size == split
        maes = []
        mses = []
        for fold in ...:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=(1 - train_split), random_state=random_seed
            )

            # Initialize the model
            model = construct_wrapper(basis_run.config, **basis_config)

            # Fit the model
            fitted_model = fit_wrapper(model, algorithm, X_train, y_train)

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
                "train_split_size": train_split,
            }
        )
    wandb.finish()


def cli_tool():
    parser = ArgumentParser("data-efficiency")
    parser.add_argument("run", type=str, help="Run to be analysed")
