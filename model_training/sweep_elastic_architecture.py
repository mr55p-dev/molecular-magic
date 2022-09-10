# Imports and initialisation
import numpy as np
import pandas as pd
import wandb
from molmagic import ml
from molmagic.ml import run_controller
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error


"""Parameters to sweep

Optimise RidgeCV first and use that parameter for AdaBoost

RidgeCV:
- Regularisation (alphas): (Update sweep runs to include model calculated alpha)

Analyse the inference time of each model as supporting evidence for its use case
The major bottleneck for most cases will be transforming the molecules into the
aggregated representation
"""

# SKlearn setup
random_seed = 50

# WandB setup (populated due to the sweep)
run = wandb.init()
run_controller.set_run(run)
run.config.update({"algorithm": "Ridge", "seed": random_seed})

# Experimental setup
split_type = run.config["splitting_type"]
label_type = run.config["label_name"]
training_artifact = run.config["training_artifact"]
alpha = run.config["elastic_alpha"]
max_iter = run.config["max_iter"]
l1_ratio = run.config["l1_ratio"]

# Dataset loading
basepath = ml.get_vector_artifact(training_artifact)

X = np.load(basepath / "features.npy")
y_raw = np.load(basepath / "labels.npy").astype(np.double)
y = ml.get_label_type(y_raw, label_type)

splitter = ml.get_split(split_type)
X_train, X_test, y_train, y_test = splitter(X, y, random_state=random_seed)

# Define the model
model = ElasticNet(alpha=alpha, random_state=random_seed, max_iter=max_iter, l1_ratio=l1_ratio, alpha=alpha)

# Fit the model
fitted_model = model.fit(X_train, y_train.squeeze())

# Save the model to wandb
ml.log_sklearn_model(fitted_model)

# Make predictions on the train set for error distribution analysis
y_pred = model.predict(X_test).squeeze()
val_mse = mean_squared_error(y_test.squeeze(), y_pred)
val_mae = mean_absolute_error(y_test.squeeze(), y_pred)

absolute_err = np.abs(y_test.squeeze() - y_pred).reshape(-1, 1)
err_table = wandb.Table(data=pd.DataFrame(absolute_err, columns=["Absolute error"]))

wandb.log(
    {
        "val_mae": val_mae,
        "val_mse": val_mse,
        "Error table": err_table,
        "Error histogram": wandb.plot.histogram(
            err_table, "Absolute error", "Absolute error distribution"
        ),
        "Solver": fitted_model.solver,
    }
)
