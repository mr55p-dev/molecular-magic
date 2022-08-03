
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

random_seed = 50
tf.random.set_seed(random_seed)
gpus = tf.config.list_logical_devices("GPU")

###################################################
# Dataset                                         #
###################################################

X = np.load("/home/luke/code/molecular-magic/auto_bandwidth_features/features.npy")
y = np.load("/home/luke/code/molecular-magic/auto_bandwidth_features/labels.npy").astype(np.double)

# TO-DO: Incorperate MolE8 train test split logic
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=random_seed)

###################################################
# Experimental setup                              #
###################################################

batch_size = 64
epochs = 500
# lr = 1e-5

# Define learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[579 * 20, 579 * 100, 579 * 400],
    values=[1e-3, 1e-4, 1e-5, 1e-6],
    name="lr_decay",
)

# Build the model
def objective(trial: op.Trial):

    # Define model architecture
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(len(X[0]))))
    for i in range(trial.suggest_int("num_layers", 1, 3)):
        model.add(keras.layers.Dense(
            units=trial.suggest_categorical(f"l{i}_dims", [16, 64, 128]),
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
    # wandb_config = dict(
    #     **{
    #         "trial_number": trial.number,
    #         "learning_rate": "20,40,400,1600,2800 st 1e-2",
    #         "batch_size": batch_size,
    #         "n_layers": trial.params["num_layers"],
    #     },
    #     **trial.params,
    # )  # Automatically includes the parameters we are using
    # wandb.init(
    #     reinit=True,
    #     project="MolecularMagic",
    #     entity="molecular-magicians",
    #     group=exp_name,
    #     config=wandb_config,
    # )

    # Fit the model
    hist = model.fit(
        x = X_train,
        y = y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            op.integration.TFKerasPruningCallback(trial, "val_loss"),
            # WandbCallback(monitor="val_loss", save_model=False, log_weights=True),
            # Consider early stopping callback
        ],
        validation_data=(X_test, y_test),
    )

    # Mark the experiment as ended
    # wandb.finish(quiet=True)

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
    # n_trials=12,
)

# TO-DO: Get rid of timestamp command line output