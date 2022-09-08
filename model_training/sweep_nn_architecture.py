# Imports and initialisation
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import tensorflow as tf
import wandb
from molmagic import ml
from molmagic.ml import run_controller
from wandb.keras import WandbCallback


"""Parameters to sweep

Optimise RidgeCV first and use that parameter for AdaBoost

RidgeCV:
- Regularisation (alphas): (Update sweep runs to include model calculated alpha)

AdaBoost: Use Ridge as base estimator
- number of estimators: [32, 64, 128]
- learning rate: [0.5, 0.75, 1.0]
- loss: [square, exponential]

Need to decide on validation split methodology
Add pruning through hyperband

Initial params
- batch_size: 64 (dont forget to scale with n_gpus)
- LR: 1e-5
- Actvation: relu
- Epochs: 4096
- Early stopping:
    - Paitience: 50
    - No threshold

Baseline sweep
- n_layers: (2, 3, 4, 5)
- node_size: (256, 512, 768) (per layer) (non-uniform)
- regularisation
    - L1
    - L2
    - L1_L2 (0, 0.01) (implement as just l1_l2)


Further sweep
- Batch size (to scale with LR params) (32, ...)
- LR schedule
- activation (leaky_relu, elu, possibly sigmoid/tanh although we already have evidency they are bad)
- Optimizers (...)
- Dropout (only useful if larger networks are in play)
- Regularisation (potentially if important from first sweep)

Additional analysis
- Fine tuning
- Multi-output models

Pick the best models based on Generalisation performance
Sample the top10 neural networks/etc

Final analysis
- Ensembles

Analyse the inference time of each model as supporting evidence for its use case
The major bottleneck for most cases will be transforming the molecules into the
aggregated representation

"""

# TF setup
random_seed = 50
tf.random.set_seed(random_seed)
gpus = tf.config.list_logical_devices("GPU")
strategy = tf.distribute.MirroredStrategy(gpus)

# WandB setup (populated due to the sweep)
run = wandb.init()
run_controller.set_run(run)
run.config.update({"algorithm": "Keras", "seed": random_seed})

# Experimental setup
split_type = run.config["splitting_type"]
label_type = run.config["label_name"]
learning_rate = run.config["learning_rate"]
activation_function = run.config["activation_function"]
epochs = run.config["epochs"]
paitence = run.config["paitence"]
batch_size = run.config["batch_size"]
n_layers = run.config["n_layers"]
n_nodes = run.config["layer_size"]
l1_regularization = run.config["l1_regularisation"]
l2_regularization = run.config["l2_regularisation"]
loss = run.config["loss_function"]
optimizer = run.config["optimizer"]
training_artifact = run.config["training_artifact"]

# Dataset loading
basepath = ml.get_vector_artifact(training_artifact)

X = np.load(basepath / "features.npy")
y_raw = np.load(basepath / "labels.npy").astype(np.double)
y = ml.get_label_type(y_raw, label_type)

splitter = ml.get_split(split_type)
X_train_np, X_test_np, y_train_np, y_test_np = splitter(X, y, random_state=random_seed)

# Convert to tf.data
n_devices = len(gpus) if len(gpus) else 1
train = (
    tf.data.Dataset.from_tensor_slices((X_train_np, y_train_np))
    .batch(batch_size * n_devices)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
test = (
    tf.data.Dataset.from_tensor_slices((X_test_np, y_test_np))
    .batch(batch_size * n_devices)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

# Create the architecture in distributed scope
# scope = strategy.scope().__enter__()
# Define the model
l_input = tf.keras.Input(shape=(X_train_np.shape[1],))
l_hidden = l_input
for _ in range(n_layers):
    l_hidden = tf.keras.layers.Dense(
        n_nodes,
        activity_regularizer=tf.keras.regularizers.l1_l2(
            l1=l1_regularization, l2=l2_regularization
        ),
    )(l_hidden)
    l_hidden = tf.keras.layers.Activation(activation_function)(l_hidden)
l_output = tf.keras.layers.Dense(1)(l_hidden)

# Compile the model
loss_func = tf.keras.losses.get(loss)
optimizer_func = tf.keras.optimizers.get(
    {"class_name": optimizer, "config": {"learning_rate": learning_rate}}
)

model = tf.keras.Model(inputs=l_input, outputs=l_output)
model.compile(
    loss=loss_func,
    optimizer=optimizer_func,
    metrics=["mse", "mae"],
)

# Fit the model
callbacks = [
    WandbCallback(
        monitor="val_loss",
        log_weights=True,
        save_model=False,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=paitence,
        restore_best_weights=True,
    ),
]
history = model.fit(
    train,
    validation_data=test,
    callbacks=callbacks,
    epochs=epochs,
)

# Save the model to wandb
ml.log_keras_model(model)

# Make predictions on the train set for error distribution analysis
y_pred = model.predict(test)

absolute_err = np.abs(y_test_np - y_pred)
err_table = wandb.Table(data=pd.DataFrame(absolute_err, columns=["Absolute error"]))

wandb.log(
    {
        "Error table": err_table,
        "Error histogram": wandb.plot.histogram(
            err_table, "Absolute error", "Absolute error distribution"
        ),
    }
)
# scope.__exit__()
