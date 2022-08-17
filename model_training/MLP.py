# Imports and initialisation
import numpy as np
import tensorflow as tf

from datetime import datetime
from molmagic import ml
from molmagic.config import aggregation as cfg_aggregation, extraction as cfg_extraction
from wandb.keras import WandbCallback

# TF setup
random_seed = 50
tf.random.set_seed(random_seed)
gpus = tf.config.list_logical_devices("GPU")

# Dataset loading
basepath = ml.get_artifact("qm9-filtered:latest")

X = np.load(basepath / "features.npy")
y = np.load(basepath / "labels.npy").astype(np.double)

splitter = ml.get_split("stoichiometric")
X_train, X_test, y_train, y_test = splitter(X, y, random_state=random_seed)

# Experimental setup
batch_size = 64
epochs = 50

# Define learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[691 * 10, 691 * 30, 691 * 60, 691 * 400],
    values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    name="lr_decay",
)

# Define the model
l_input = tf.keras.Input(shape=(X_train.shape[1],))
l_hidden = tf.keras.layers.Dense(256)(l_input)
l_hidden = tf.keras.layers.Dense(256)(l_hidden)
l_output = tf.keras.layers.Dense(1)(l_hidden)

# Compile the model
loss = tf.keras.losses.MeanSquaredError()
op = tf.keras.optimizers.Adam(lr_schedule)
model = tf.keras.Model(inputs=l_input, outputs=l_output)
model.compile(loss=loss, optimizer=op, metrics=["mse", "mae"])

# Create the configuration object
wandb_config = {}
ml.run.config.update(wandb_config)

# Fit the model
callbacks = [
    WandbCallback(
        monitor="val_loss",
        log_weights=True,
        save_model=False,
        validation_data=(X_test, y_test),
        input_type="auto",
        output_type="label",
    )
]
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    epochs=epochs,
    batch_size=batch_size * len(gpus),  # Consider the number of replicas
)

# Save the model to wandb
ml.log_model(model)
