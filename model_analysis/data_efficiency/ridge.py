# Imports and initialisation
import numpy as np
import pandas as pd
import wandb
from molmagic import ml
from molmagic.ml import get_artifact_of_kind, run_controller
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


"""Parameters to sweep

Optimise RidgeCV first and use that parameter for AdaBoost

RidgeCV:
- Regularisation (alphas): (Update sweep runs to include model calculated alpha)

Analyse the inference time of each model as supporting evidence for its use case
The major bottleneck for most cases will be transforming the molecules into the
aggregated representation
"""

run = run_controller.use_run("data-efficiency")
run_basis_name = ...

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
split_type = basis_run.config["splitting_type"]
label_type = basis_run.config["label_name"]
training_artifact = basis_run.config["training_artifact"]
alpha = basis_run.config["ridge_alpha"]

# Dataset loading
basepath = ml.get_vector_artifact(training_artifact)

X = np.load(basepath / "features.npy")
y_raw = np.load(basepath / "labels.npy").astype(np.double)
y = ml.get_label_type(y_raw, label_type)

##### Setup cross validation with a range of values
train_split_sizes = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
scores = []
for train_split in train_split_sizes:
    #### Do cross validation with a train/test split size == split

    maes = []
    mses = []
    for fold in ...:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(1 - train_split), random_state=random_seed
        )
        # Define the model
        model = Ridge(alpha=alpha, random_state=random_seed)
        fitted_model = model.fit(X_train, y_train.squeeze())

        y_pred = model.predict(X_test).squeeze()
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
