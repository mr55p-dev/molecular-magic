# Imports and initialisation
import numpy as np
import tensorflow as tf
import wandb
from molmagic import ml
from molmagic.ml import run_controller
from wandb.keras import WandbCallback

# TF setup
random_seed = 50
tf.random.set_seed(random_seed)
gpus = tf.config.list_logical_devices("GPU")
strategy = tf.distribute.MirroredStrategy(gpus)

# WandB setup
run = wandb.init(
    project="MolecularMagic",
    entity="molecular-magicians",
    # name="test",
    # group="sweep",
    job_type="training",
)
run_controller.set_run(run)
config = run.config

# Experimental setup
batch_size = config["batch_size"]
epochs = config["epochs"]
split_type = config["train-test-splitter"]
label_type = "free_energy"

# Dataset loading (also inits a wandb run if not done explicitly)
data_path = ml.get_vector_artifact("qm9-light-bw_scott:latest")

X = np.load(data_path / "features.npy")
y_raw = np.load(data_path / "labels.npy").astype(np.double)
y = ml.get_label_type(y_raw, label_type)

splitter = ml.get_split(split_type)
X_train, X_test, y_train, y_test = splitter(X, y, random_state=random_seed)

# Define learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[1500, 3000],
    values=[1e-2, 1e-3, 1e-5],
    name="lr_decay",
)

with strategy.scope():
    # Define the model
    l_input = tf.keras.Input(shape=(X_train.shape[1],))
    l_hidden = tf.keras.layers.Dense(config["l1_size"], activation=config["activation"])(
        l_input
    )
    l_hidden = tf.keras.layers.Dense(config["l2_size"], activation=config["activation"])(
        l_hidden
    )
    l_output = tf.keras.layers.Dense(1, activation="linear")(l_hidden)

    # Compile the model
    loss = tf.keras.losses.MeanSquaredError()
    op = tf.keras.optimizers.Adam(lr_schedule)
    model = tf.keras.Model(inputs=l_input, outputs=l_output)
    model.compile(loss=loss, optimizer=op, metrics=["mse", "mae"])

    # Create the configuration object
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
    ml.log_model(model)
