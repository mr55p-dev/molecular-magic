
###################################################
# Imports and initialisation                      #
###################################################

import wandb

import numpy as np
import tensorflow as tf
import optuna as op

from collections import defaultdict
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow import keras
from wandb.keras import WandbCallback
from molmagic.split import stoichiometric_split
from molmagic.config import aggregation as cfg_aggregation, extraction as cfg_extraction

random_seed = 50
tf.random.set_seed(random_seed)
gpus = tf.config.list_logical_devices("GPU")

###################################################
# Dataset                                         #
###################################################

# X = np.load("./auto_bandwidth_features/features.npy")
# y = np.load("./auto_bandwidth_features/labels.npy").astype(np.double)
X = np.load('./part1_vec/features.npy')
y = np.load('./part1_vec/labels.npy').astype(np.double)

# Use the MolE8 train_test_split logic
X_train, X_test, y_train, y_test = stoichiometric_split(
    X, y, random_state=random_seed
)

# Use standard train_test_split logic
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=random_seed)

###################################################
# Experimental setup                              #
###################################################

batch_size = 64
epochs = 500
# lr = 1e-5

# Define learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[691 * 10, 691 * 30, 691 * 60, 691 * 400],
    values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    name="lr_decay",
)

# Build the model
def objective(trial: op.Trial):

    # Define model architecture
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(len(X[0]))))
    for i in range(trial.suggest_int("num_layers", 2, 5)):
        model.add(keras.layers.Dense(
            units=trial.suggest_categorical(f"l{i}_dims", [64, 128, 256]),
            activation="relu")
            )
    model.add(keras.layers.Dense(units=1, activation="linear"))

    # Test different activation functions
    # Test dropout layer...

    print("Defined model")
    print(model.summary())

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_function = keras.losses.MeanSquaredError()

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=["mse", "mae"],
    )
    print("Compiled model")

    # Configure the weights and biases experiment
    wandb_config = {
        "trial_number": trial.number,
        "training_params": {
            "learning_rate": "20,40,400,1600,2800 st 1e-2",
            "batch_size": batch_size,
            "n_layers": trial.params["num_layers"],
        },
        "model_params": trial.params,
        "parser_params": cfg_extraction,
        "aggregation_params": cfg_aggregation,
    }  # Automatically includes the parameters we are using
    wandb.init(
        reinit=True,
        project="MolecularMagic",
        entity="molecular-magicians",
        group=exp_name,
        config=wandb_config,
    )

    # Fit the model
    hist = model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            op.integration.TFKerasPruningCallback(trial, "val_loss"),
            WandbCallback(monitor="val_loss", save_model=False, log_weights=True),
            # Consider early stopping callback
        ],
        validation_data=(X_test, y_test),
    )

    # Mark the experiment as ended
    wandb.finish(quiet=True)

    return hist.history["val_loss"][-1]  # Return the end validation loss


###################################################
# Model training and logging                      #
###################################################
def generate_experiment_name(epochs, batch_size):
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    return str(time_str + "_" + str(epochs) + "ep_" + str(batch_size) + "bs")

exp_name = generate_experiment_name(epochs, batch_size)
exp_name += "_lrsched"

# Setup the hyperparmeter search
# This is if you want to use optunas local optuna-dashboard system, which is pretty good
# but not really as good as wandb
# storage = op.storages.RedisStorage(
#     url="redis://localhost:6379/optuna",
# )

storage = op.storages.InMemoryStorage()
pruner = op.pruners.HyperbandPruner()
sampler = op.samplers.RandomSampler(
    seed=random_seed
)  # There are some more options in optuna that work well

study = op.create_study(
    study_name=exp_name,
    storage=storage,
    sampler=sampler,
    pruner=pruner,
    direction="minimize",
)

# Run the optimization
study.optimize(
    objective,
    n_trials=1,
)

# TO-DO: Get rid of timestamp command line output