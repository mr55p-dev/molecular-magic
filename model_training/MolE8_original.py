from datetime import datetime
import tensorflow as tf
import numpy as np
from molmagic.split import stoichiometric_split as split
from pathlib import Path
from wandb.keras import WandbCallback

# from molmagic.config import
import wandb


# Load dataset
data_dir = Path("./data/qm9/qm9")
features = np.load(data_dir / "features.npy")
labels = np.load(data_dir / "labels.npy")
X_train, X_test, y_train, y_test = split(features, labels)

# Set parameters
tensorboard_output = Path("./MolE8/") / "tb_log"
n_features = X_train.shape[1]
n_nodes = 761
batch_size = 64
epochs = 7000
learning_rate = 1e-7

# Set initializers
kernel_initialiser = tf.keras.initializers.RandomUniform(minval=-500, maxval=100)
kernel_initialiser2 = tf.keras.initializers.RandomUniform(minval=0, maxval=0.1)
bias_initialiser = tf.keras.initializers.RandomUniform(minval=0, maxval=10)
bias_initialiser2 = tf.keras.initializers.RandomUniform(minval=0, maxval=0.01)

# Create the model
input = tf.keras.Input(shape=n_features)
layer = tf.keras.layers.Dense(
    units=761,
    activation="relu",
    kernel_initializer=kernel_initialiser,
    bias_initializer=bias_initialiser,
    kernel_regularizer=tf.keras.regularizers.l2(0.1),
    bias_regularizer=tf.keras.regularizers.l2(0.1),
)(input)
layer = tf.keras.layers.Dense(
    units=761,
    activation="relu",
    kernel_initializer=kernel_initialiser,
    bias_initializer=bias_initialiser,
    kernel_regularizer=tf.keras.regularizers.l2(0.1),
    bias_regularizer=tf.keras.regularizers.l2(0.1),
)(layer)
output = tf.keras.layers.Dense(
    units=761,
    activation="linear",
    kernel_initializer=kernel_initialiser,
    bias_initializer=bias_initialiser,
    kernel_regularizer=tf.keras.regularizers.l2(0.1),
    bias_regularizer=tf.keras.regularizers.l2(0.1),
)(layer)

# Compile the model
model = tf.keras.Model(inputs=input, outputs=output)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_function = tf.keras.losses.MeanSquaredError()

model.compile(
    optimizer=optimizer,
    loss=loss_function,
    metrics=["mse", "mae"],
)


def get_tensorboard_callback(log_root: Path = Path("logs/")):
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    return tf.keras.callbacks.TensorBoard(log_dir=str(log_root / time_str))


# Experiment tracking
wandb_config = {
    "partitioning method": split.__name__,
    "learning rate": learning_rate,
    "batch size": batch_size,
    "number of features": n_features,
    "MolE8 model": True,
}
wandb.init(
    project="MolecularMagic",
    entity="molecular-magicians",
    config=wandb_config,
)

# Version dataset
dataset_artifact = wandb.Artifact(
    name="qm9_filtered",
    type="dataset",
)
dataset_artifact.add_dir(data_dir)
wandb.log_artifact(dataset_artifact)

# Fit the model
hist = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[
        get_tensorboard_callback(tensorboard_output),
        WandbCallback(
            monitor="val_loss",
            log_weights=True,
            training_data=(X_train, y_train),
            validation_data=(X_test, y_test),
        ),
    ],
)
