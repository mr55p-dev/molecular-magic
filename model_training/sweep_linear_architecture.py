# Imports and initialisation
import numpy as np
import pandas as pd
import wandb
from molmagic import ml
from molmagic.ml import run_controller
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


"""Parameters to sweep

Optimise RidgeCV first and use that parameter for AdaBoost

RidgeCV:
- Regularisation (alphas): (Update sweep runs to include model calculated alpha)

Analyse the inference time of each model as supporting evidence for its use case
The major bottleneck for most cases will be transforming the molecules into the
aggregated representation
"""

# WandB setup (populated due to the sweep)
run = wandb.init(
    entity="molecular-magicians",
    project="MolecularMagic",
    name="linear_regression",
)
run_controller.set_run(run)
run.config.update({"algorithm": "Linear"})

run.config.update(
    {
        "seed": 50,
        "splitting_type": "random",
        "label_name": "free_energy",
        "training_artifact": "molecular-magicians/MolecularMagic/qm9-std-train-0.48-scott:latest",
    }
)

# Experimental setup
seed = run.config["seed"]
split_type = run.config["splitting_type"]
label_type = run.config["label_name"]
training_artifact = run.config["training_artifact"]

# Dataset loading
basepath = ml.get_vector_artifact(training_artifact)

X = np.load(basepath / "features.npy")
y_raw = np.load(basepath / "labels.npy").astype(np.double)
y = ml.get_label_type(y_raw, label_type)

splitter = ml.get_split(split_type)
X_train, X_test, y_train, y_test = splitter(X, y, random_state=seed)

# Define the model
model = LinearRegression()

# Fit the model
fitted_model = model.fit(X_train, y_train.squeeze())

# Save the model to wandb
ml.log_sklearn_model(fitted_model)

# Make predictions on the train set for error distribution analysis
y_pred = model.predict(X_test).squeeze()
val_mse = mean_squared_error(y_test.squeeze(), y_pred)
val_mae = mean_absolute_error(y_test.squeeze(), y_pred)
val_mape = mean_absolute_percentage_error(y_test.squeeze(), y_pred)

data = np.concatenate((y_test.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)
errors = pd.DataFrame(data, columns=["label", "prediction"])

err_table = wandb.Table(data=errors)

wandb.log(
    {
        "val_mae": val_mae,
        "val_mse": val_mse,
        "val_mape": val_mape,
        "Error table": err_table,
    }
)
