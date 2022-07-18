###################################################
# Imports and initialisation                      #
###################################################

from importlib import reload

import setup
reload(setup)

import wandb

import numpy as np
import tensorflow as tf
import optuna as op

from collections import defaultdict
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from wandb.keras import WandbCallback

from keras import initializers
from keras.regularizers import l2

random_seed = 50
tf.random.set_seed(random_seed)
gpus = tf.config.list_logical_devices("GPU")

###################################################
# Experimental setup                              #
###################################################

batch_size = 64
epochs = 7000

train_ds, test_ds, data_len = setup.generate_data(batch_size, epochs, random_seed)

# Define learning rate schedule
learning_rate = 1e-7

# Weight initializations
kernel_initialiser = initializers.RandomUniform(minval=-500, maxval=100)
kernel_initialiser2 = initializers.RandomUniform(minval=0, maxval=0.1)

bias_initialiser = initializers.RandomUniform(minval=0, maxval=10)
bias_initialiser2 = initializers.RandomUniform(minval=0, maxval=0.01)

# Build the model
def objective(trial: op.Trial):

    # Define model architecture
    model = keras.Sequential()
    model.add(layers.Input(shape=(data_len)))
    model.add(layers.Dense(
                units= 761,
                activation="relu",
                kernel_initializer=kernel_initialiser,
                bias_initializer=bias_initialiser,
                kernel_regularizer=l2(0.1),
                bias_regularizer=l2(0.1)
                ))
    model.add(layers.Dense(
                units= 761,
                activation="relu",
                kernel_initializer=kernel_initialiser2,
                bias_initializer=bias_initialiser2,
                kernel_regularizer=l2(0.1),
                bias_regularizer=l2(0.1)
                ))
    model.add(layers.Dense(
                units=1,
                activation="linear",
                kernel_initializer=kernel_initialiser2,
                bias_initializer=bias_initialiser2,
                kernel_regularizer=l2(0.1),
                bias_regularizer=l2(0.1)
                ))

    # Test different activation functions
    # Test dropout layer...

    print("Defined model")
    print(model.summary())

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_function = keras.losses.MeanSquaredError()

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=["mse", "mae"],
    )
    print("Compiled model")

    # Configure the weights and biases experiment
    wandb_config = dict(
        **{
            "trial_number": trial.number,
            "learning_rate": "1e-7",
            "batch_size": batch_size,
            "hidden_layers": 2,
            "hidden_dims": data_len
        },
        **trial.params,
    )  # Automatically includes the parameters we are using
    wandb.init(
        reinit=True,
        project="MolecularMagic",
        entity="molecular-magicians",
        group=exp_name,
        config=wandb_config,
    )

    # Fit the model
    hist = model.fit(
        train_ds,
        validation_data=test_ds,
        callbacks=[
            op.integration.TFKerasPruningCallback(trial, "val_loss"),
            WandbCallback(monitor="val_loss", save_model=False, log_weights=True),
            # Consider early stopping callback
        ],
        batch_size=batch_size,
        epochs=epochs,
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
exp_name += "_match_supp"

# Setup the hyperparmeter search
# This is if you want to use optunas local optuna-dashboard system, which is pretty good
# but not really as good as wandb
# storage = op.storages.RedisStorage(
#     url="redis://localhost:6379/optuna",
# )

storage = op.storages.InMemoryStorage()
pruner = op.pruners.HyperbandPruner()
sampler = op.samplers.RandomSampler(
    seed=42
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