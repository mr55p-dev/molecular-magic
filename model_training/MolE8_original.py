# Imports and initialisation
import numpy as np
import tensorflow as tf
import wandb
from molmagic import ml
from molmagic.ml import run_controller
from wandb.keras import WandbCallback


# Vars
split_type = "stoichiometric"
label_type = "free_energy"
training_artifact = "qm9-std_scott:latest"
batch_size = 64
epochs = 7000
learning_rate = 1e-7
n_nodes = 761

# TF setup
random_seed = 50
tf.random.set_seed(random_seed)
gpus = tf.config.list_logical_devices("GPU")
strategy = tf.distribute.MirroredStrategy(gpus)

# WandB setup
run = wandb.init(
    project="MolecularMagic",
    entity="molecular-magicians",
    name="test",
    # group="test_group",
    job_type="training",
)
run_controller.set_run(run)

# Dataset loading (also inits a wandb run if not done explicitly)
basepath = ml.get_vector_artifact(training_artifact)

X = np.load(basepath / "features.npy")
y_raw = np.load(basepath / "labels.npy").astype(np.double)
y = ml.get_label_type(y_raw, label_type)

splitter = ml.get_split(split_type)
X_train, X_test, y_train, y_test = splitter(X, y, random_state=random_seed)
n_features = X_train.shape[1]

with strategy.scope():
    # Set initializers
    kernel_initialiser = tf.keras.initializers.RandomUniform(minval=-500, maxval=100)
    kernel_initialiser2 = tf.keras.initializers.RandomUniform(minval=0, maxval=0.1)
    bias_initialiser = tf.keras.initializers.RandomUniform(minval=0, maxval=10)
    bias_initialiser2 = tf.keras.initializers.RandomUniform(minval=0, maxval=0.01)

    # Create the model
    input_layer = tf.keras.Input(shape=n_features)
    layer = tf.keras.layers.Dense(
        units=761,
        activation="relu",
        kernel_initializer=kernel_initialiser,
        bias_initializer=bias_initialiser,
        kernel_regularizer=tf.keras.regularizers.l2(0.1),
        bias_regularizer=tf.keras.regularizers.l2(0.1),
    )(input_layer)
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
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_function = tf.keras.losses.MeanSquaredError()

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=["mse", "mae"],
    )

    # Experiment tracking
    wandb_config = {"splitting_type": split_type, "target_name": label_type}
    run.config.update(wandb_config)

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
    ml.log_keras_model(model)
