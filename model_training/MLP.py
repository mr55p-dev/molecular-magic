###################################################
# Imports and initialisation                      #
###################################################

import wandb

import numpy as np
import optuna as op
import tensorflow as tf

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import keras
from wandb.keras import WandbCallback

from magic.split import stoichiometric_split
from magic.config import aggregation as cfg_aggregation, extraction as cfg_extraction

random_seed = 50
tf.random.set_seed(random_seed)
gpus = tf.config.list_logical_devices("GPU")

###################################################
# Dataset                                         #
###################################################

X = np.load("/home/luke/code/molecular-magic/autoband_badh_freeeng/features.npy")
y = np.load("/home/luke/code/molecular-magic/autoband_badh_freeeng/labels.npy").astype(np.double)

# Using MolE8 train_test_split logic
X_train, X_test, y_train, y_test = stoichiometric_split(
    X, y, random_state=random_seed)

###################################################
# Experimental setup                              #
###################################################

# Note: MolE8 uses batch size of 64

epochs = 800

# Define static learning rate
# lr = 1e-5

# Define learning rate schedule (batch size 64)
batch_size = 64
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[576 * 10, 576 * 30, 576 * 300], 
    values=[1e-2, 1e-3, 1e-4, 1e-5],
    name="lr_decay",
)

# Define learning rate schedule (batch size 32) - seems less performat after preliminary tests
# batch_size = 32
# lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
#     boundaries=[1151 * 10, 1151 * 100, 1151 * 300], 
#     values=[1e-3, 1e-4, 1e-5, 1e-6],
#     name="lr_decay",
# )


def define_model(trial: op.Trial):
    # Optimize the number of layers, and hidden units
    n_layers = trial.suggest_int("n_layers", 2, 4)
    units = trial.suggest_categorical("all_units", [256, 512])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0, 0.1])
    activation_function = trial.suggest_categorical("activation_function", ["sigmoid", "relu", "tanh"])

    # Define model architecture
    model = keras.Sequential()

    # Append to the model
    model.add(keras.layers.Input(shape=X.shape[1]))
    for i in range(n_layers):
        model.add(keras.layers.Dense(
            # units=trial.suggest_categorical(f"l{i}_dims", [64, 128, 256]),
            units=units,
            kernel_constraint=keras.constraints.unit_norm(),
            activation=activation_function)
            )
    model.add(keras.layers.Dropout(rate=dropout_rate))
    model.add(keras.layers.Dense(units=1, activation="linear"))

    # Test different activation functions: sigmoid, tanh, leaky relu, etc...
    # Test dropout layer...

    print("Defined model")
    print(model.summary())

    return model

def objective(trial: op.Trial):

    # Generate the model
    model = define_model(trial)

    # Generate the optimizers
    # optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "RMSprop", "SGD"])
    # lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # optimizer = getattr(optim, optimizer_name)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_function = keras.losses.MeanSquaredError()

    model.compile(
    optimizer=optimizer,
    loss=loss_function,
    metrics=["mse", "mae"],
    )

    wandb_config = dict(trial.params) # Clean up the config? add aggregation?
    wandb_config["trial.number"] = trial.number
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
            WandbCallback(monitor="val_loss", save_model=False, log_weights=False),
            keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=40)
            # Consider early stopping callback
        ],
        validation_data=(X_test, y_test),
    )

    wandb.finish() #quiet=True

    return hist.history["val_loss"][-1]



def generate_experiment_name(epochs, batch_size):
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    return str(time_str + "_" + str(epochs) + "ep_" + str(batch_size) + "bs")

exp_name = generate_experiment_name(epochs, batch_size)
exp_name += "_preliminary_testing"

storage = op.storages.InMemoryStorage()
pruner = op.pruners.HyperbandPruner()
# pruner = op.pruners.MedianPruner()
sampler = op.samplers.RandomSampler(
    seed=random_seed
)  # There are some more options in optuna that work well

study = op.create_study(
    direction="minimize",
    study_name=exp_name,
    storage=storage,
    pruner=pruner,
    sampler=sampler,
)

study.optimize(
    objective,
    # n_trials=100,
)

#     optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
#     loss_function = keras.losses.MeanSquaredError()



#     model.compile(
#         optimizer=optimizer,
#         loss=loss_function,
#         metrics=["mse", "mae"],
#     )
#     print("Compiled model")

#     # Configure the weights and biases experiment
#     wandb_config = {
#         "trial_number": trial.number,
#         "training_params": {
#             "learning_rate": "schedule",
#             "batch_size": batch_size,
#             "num_layers": trial.params["num_layers"],
#             "all_dims": trial.params["all_dims"]
#         },
#         "model_params": trial.params,
#         "parser_params": cfg_extraction,
#         "aggregation_params": cfg_aggregation,
#     }  # Automatically includes the parameters we are using
#     wandb.init(
#         reinit=True,
#         project="MolecularMagic",
#         entity="molecular-magicians",
#         group=exp_name,
#         config=wandb_config,
#     )

#     # Fit the model
#     hist = model.fit(
#         x=X_train,
#         y=y_train,
#         batch_size=batch_size,
#         epochs=epochs,
#         callbacks=[
#             op.integration.TFKerasPruningCallback(trial, "val_loss"),
#             WandbCallback(monitor="val_loss", save_model=False, log_weights=True),
#             # Consider early stopping callback
#         ],
#         validation_data=(X_test, y_test),
#     )

#     # Mark the experiment as ended
#     # wandb.finish(quiet=True)

#     return hist.history["val_loss"][-1]  # Return the end validation loss


# ###################################################
# # Model training and logging                      #
# ###################################################
# def generate_experiment_name(epochs, batch_size):
#     now = datetime.now()
#     time_str = now.strftime("%Y-%m-%d_%H:%M:%S")
#     return str(time_str + "_" + str(epochs) + "ep_" + str(batch_size) + "bs")

# exp_name = generate_experiment_name(epochs, batch_size)
# exp_name += "_preliminary_testing"

# # Setup the hyperparmeter search
# # This is if you want to use optunas local optuna-dashboard system, which is pretty good
# # but not really as good as wandb
# # storage = op.storages.RedisStorage(
# #     url="redis://localhost:6379/optuna",
# # )

# storage = op.storages.InMemoryStorage()
# pruner = op.pruners.HyperbandPruner()
# sampler = op.samplers.RandomSampler(
#     seed=random_seed
# )  # There are some more options in optuna that work well

# study = op.create_study(
#     study_name=exp_name,
#     storage=storage,
#     sampler=sampler,
#     pruner=pruner,
#     direction="minimize",
# )

# # Run the optimization
# study.optimize(
#     objective,
#     # n_trials=1,
# )