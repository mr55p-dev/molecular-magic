"""
Reimplementation of `NN_custom_split.py`
"""
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from tensorflow.keras.regularizers import l2
import numpy as np
from pathlib import Path
from collections import defaultdict


from datetime import datetime


def get_tensorboard_callback(log_root: Path = Path("logs/")):
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    return keras.callbacks.TensorBoard(log_dir=str(log_root / time_str))


# Define save locations
basepath = Path("static_data/NN_rewrite/")
basepath.mkdir(parents=True, exist_ok=True)

model_checkpoints_output = basepath / "model_checkpoint"
model_output = basepath / "model"
tensorboard_output = basepath / "tb_log"
metric_output = basepath / "metrics.csv"

# Define parameters
seed = 50
lr = 1e-5
decay_rate = 3e-5
batch_size = 64
epochs = 7000

atom_order = ["C", "H", "N", "O"]

# Load data
data_basepath = Path("static_data/create_features_output/data")
X = np.load(data_basepath / "features.npy", allow_pickle=True)
y = np.load(data_basepath / "labels.npy", allow_pickle=True)

# Compute the molecular formulae for each molecule
def create_mol_formula(atom_vec) -> str:
    return "".join(
        map(lambda x: f"{x}{atom_vec[atom_order.index(x)]}", ["C", "N", "O", "H"])
    )


atom_vector = (X[:, -15:-11] / 100).astype("int")
molecular_formulae = list(map(create_mol_formula, atom_vector))
unique_molecular_formulae = set(molecular_formulae)

X_dict = defaultdict(list)
y_dict = defaultdict(list)
for form, vec, target in zip(molecular_formulae, X, y):
    X_dict[form].append(vec)
    y_dict[form].append(target)

X_dict = dict(**X_dict)
y_dict = dict(**y_dict)

# Split the data into train and test set based on the rules:
dropped_structures = []
train_items = []
test_items = []
for formula in X_dict:
    if len((candidates := X_dict[formula])) == 1:
        # Discard structure
        dropped_structures.append(formula)
    elif len(candidates) == 2:
        # Assign one each to train and test
        train_candidate, test_candidate = candidates
        train_energy, test_energy = y_dict[formula]

        train_items.append((train_candidate, train_energy))
        test_items.append((test_candidate, test_energy))
    else:
        s_train, s_test, e_train, e_test = train_test_split(
            candidates, y_dict[formula], test_size=0.33, random_state=50
        )
        train_items += zip(s_train, e_train)
        test_items += zip(s_test, e_test)

# Create the train set
train_features, train_labels = zip(*train_items)
X_train = np.array(train_features)
y_train = np.array(train_labels)

# Create the test set
test_features, test_labels = zip(*test_items)
X_test = np.array(test_features)
y_test = np.array(test_labels)

# Rescale the features such that they have variance 0
# Pretty sure there is a bug in the original code where they forgot to scale the mean to zero aswell...
def rescale(x: np.ndarray) -> np.ndarray:
    # return (x - x.mean()) / x.std()
    return x / x.std()


X_train = rescale(X_train)  # 28302
y_train = rescale(y_train)  # 14120

X_test = rescale(X_test)
y_test = rescale(y_test)

# Define the weight and bias initializers
kernel_initialiser = keras.initializers.RandomUniform(minval=-500, maxval=100)
kernel_initialiser_input = keras.initializers.RandomUniform(minval=0, maxval=0.1)

bias_initialiser = keras.initializers.RandomUniform(minval=0, maxval=10)
bias_initialiser_input = keras.initializers.RandomUniform(minval=0, maxval=0.01)

# Construct the NN
n_features = X_train.shape[1]
l_input = keras.layers.Input(shape=(n_features))
l_hidden = keras.layers.Dense(
    761,
    activation="relu",
    kernel_regularizer=l2(0.1),
    bias_regularizer=l2(0.1),
    kernel_initializer=kernel_initialiser_input,
    bias_initializer=bias_initialiser_input,
)(l_input)
l_hidden = keras.layers.Dense(
    761,
    activation="relu",
    kernel_regularizer=l2(0.1),
    bias_regularizer=l2(0.1),
    kernel_initializer=kernel_initialiser,
    bias_initializer=bias_initialiser,
)(l_hidden)
l_output = keras.layers.Dense(
    761,
    activation="relu",
    kernel_regularizer=l2(0.1),
    bias_regularizer=l2(0.1),
    kernel_initializer=kernel_initialiser,
    bias_initializer=bias_initialiser,
)(l_hidden)

model = keras.Model(inputs=l_input, outputs=l_output)

print("Defined model")
print(model.summary())

model.compile(
    optimizer=keras.optimizers.Adam(lr=lr),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()],
)
print("Compiled model")

# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    # TODO #6 implement wandb monitoring
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath=model_checkpoints_output,
            monitor="val_mean_absolute_error",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
        ),
        get_tensorboard_callback(tensorboard_output),
    ],
)

# Save model
model.save(model_output)

# Save output
hist = history.history
columns = ["loss", "validation_loss", "mse", "validation_mse"]
output_data = pd.DataFrame(zip(*[hist[i] for i in columns]), columns=columns)
output_data.to_csv(metric_output)
