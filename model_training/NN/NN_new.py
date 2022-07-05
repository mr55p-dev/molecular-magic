"""
Reimplementation of `NN_custom_split.py`
"""

###################################################
# Imports and initialisation                      #
###################################################

# import wandb

import numpy as np
import keras_tuner as kt
import pandas as pd
import tensorflow as tf

from collections import defaultdict
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from wandb.keras import WandbCallback

random_seed = 50
tf.random.set_seed(random_seed)
gpus = tf.config.list_logical_devices("GPU")


###################################################
# Load and transform the data                     #
###################################################

atom_order = ["C", "H", "N", "O"]

# Load the data
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

# Group by formula
X_dict = defaultdict(list)
y_dict = defaultdict(list)
for form, vec, target in zip(molecular_formulae, X, y):
    X_dict[form].append(vec)
    y_dict[form].append(target)

# Put this back into a standard dictionary
X_dict = dict(**X_dict)
y_dict = dict(**y_dict)

# Split the data into train and test set
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
            candidates, y_dict[formula], test_size=0.33, random_state=random_seed
        )
        train_items += zip(s_train, e_train)
        test_items += zip(s_test, e_test)

# strategy = tf.distribute.MirroredStrategy(gpus)
# # with strategy.scope():
# scope = strategy.scope()
# scope.__enter__()

# Create the train set
train_features, train_labels = zip(*train_items)
X_train = np.array(train_features)
y_train = np.array(train_labels)

# Create the test set
test_features, test_labels = zip(*test_items)
X_test = np.array(test_features)
y_test = np.array(test_labels)

# Rescale the features such that they have variance 0
# Note they do not scale the mean to zero, only the variation about the mean to 1
def rescale(x: np.ndarray) -> np.ndarray:
    # return (x - x.mean()) / x.std()
    return x / x.std()


X_train_norm = X_train / X_train.std()
X_test_norm = X_test / X_train.std()



###################################################
# Parameter setup                                 #
###################################################

# Define save locations
basepath = Path("static_data/NN_rewrite/")
basepath.mkdir(parents=True, exist_ok=True)

model_checkpoints_output = basepath / "model_checkpoint"
model_output = basepath / "model"
tensorboard_output = basepath / "tb_log"
metric_output = basepath / "metrics.csv"
tuner_output = basepath / "tuner_checkpoint"



# Define parameters

# lr = 1e-5

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 1e-2,
        decay_steps=443*20, # STEPS != EPOCHS. At batch size 64, there are 443*100 steps per epoch
        decay_rate=0.25,
        staircase=True)

# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate = 1e-4,
#     decay_steps=100,
#     decay_rate=0.90,
#     staircase=True)

batch_size = 64
epochs = 500

regularization_degree = 0.1
# decay_rate = 3e-5


# Create tf.data.Dataset
train_ds = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.convert_to_tensor(X_train_norm, dtype=tf.float32),
            tf.convert_to_tensor(y_train, dtype=tf.float32),
        )
    )
    .batch(batch_size=batch_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.convert_to_tensor(X_test_norm, dtype=tf.float32),
            tf.convert_to_tensor(y_test, dtype=tf.float32),
        )
    )
    .batch(batch_size=batch_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

def model_builder(hp):

    # Define the weight and bias initializers
    # kernel_initialiser = keras.initializers.RandomUniform(minval=-500, maxval=100)
    # kernel_initialiser_input = keras.initializers.RandomUniform(minval=0, maxval=0.1)

    # bias_initialiser = keras.initializers.RandomUniform(minval=0, maxval=10)
    # bias_initialiser_input = keras.initializers.RandomUniform(minval=0, maxval=0.01)

    # Construct the NN
    # n_features = X_train.shape[1]
    # if hp.Boolean("cust_init"):
    #     l_input = keras.layers.Input(shape=(X_train.shape[1]))
    #     l_hidden = keras.layers.Dense(
    #         units=hp.Choice("l1_dims", [128]), #761
    #         # 761,
    #         activation="relu",
    #         kernel_regularizer=l2(regularization_degree),
    #         bias_regularizer=l2(regularization_degree),
    #         kernel_initializer=kernel_initialiser_input,
    #         bias_initializer=bias_initialiser_input,
    #     )(l_input)
    #     l_hidden = keras.layers.Dense(
    #         units=hp.Choice("l2_dims", [128]), #761
    #         # 761,
    #         activation="relu",
    #         kernel_regularizer=l2(regularization_degree),
    #         bias_regularizer=l2(regularization_degree),
    #         kernel_initializer=kernel_initialiser,
    #         bias_initializer=bias_initialiser,
    #     )(l_hidden)
    #     l_output = keras.layers.Dense(
    #         1,
    #         activation="linear",
    #         kernel_regularizer=l2(regularization_degree),
    #         bias_regularizer=l2(regularization_degree),
    #         kernel_initializer=kernel_initialiser,
    #         bias_initializer=bias_initialiser,
    #     )(l_hidden)
    # else:
    l_input = keras.layers.Input(shape=(X_train.shape[1]))
    l_hidden = keras.layers.Dense(
        units=hp.Choice("l1_dims", [128]), #761
        # 761,
        activation="relu",
        # kernel_regularizer=l2(regularization_degree),
        # bias_regularizer=l2(regularization_degree),
        # kernel_initializer=kernel_initialiser_input,
        # bias_initializer=bias_initialiser_input,
    )(l_input)
    l_hidden = keras.layers.Dense(
        units=hp.Choice("l2_dims", [128]), #761
        # 761,
        activation="relu",
        # kernel_regularizer=l2(regularization_degree),
        # bias_regularizer=l2(regularization_degree),
        # kernel_initializer=kernel_initialiser,
        # bias_initializer=bias_initialiser,
    )(l_hidden)
    l_output = keras.layers.Dense(
        1,
        activation="linear",
        # kernel_regularizer=l2(regularization_degree),
        # bias_regularizer=l2(regularization_degree),
        # kernel_initializer=kernel_initialiser,
        # bias_initializer=bias_initialiser,
    )(l_hidden)

    
    model = keras.Model(inputs=l_input, outputs=l_output)

    print("Defined model")
    print(model.summary())

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule),
        loss = keras.losses.MeanSquaredError(),
        metrics=['mse', 'mae'],
    )
    print("Compiled model")

    return model



###################################################
# Weights and Biases setup                        #
###################################################

# wandb.init(project="MolecularMagic", entity="molecular-magicians")
# wandb.config = {
#     "learning_rate": lr,
#     "epochs": epochs,
#     "batch_size": batch_size,
#     "loss_function": loss_function.name,
#     "optimizer": optimizer._name,
#     "regularization": regularization_degree,
# }

###################################################
# Fit the model                                   #
###################################################
def generate_experiment_name(epochs, batch_size):
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    return str(time_str + "_" + str(epochs) + "ep_" + str(batch_size) + "bs")

exp_name = generate_experiment_name(epochs, batch_size)
exp_name += "_decaylr"


# Train
tuner = kt.RandomSearch(
    hypermodel=model_builder,
    objective='val_loss',
    # overwrite=True,
    executions_per_trial=1,
    directory=tuner_output,
    project_name=exp_name,
    seed=random_seed
)

tuner.search(
    train_ds,
    # steps_per_epoch=500,
    validation_data=test_ds,
    # validation_steps=300,
    epochs=epochs, #epochs
    verbose=1,
    batch_size=batch_size,
    callbacks=[
        # keras.callbacks.ModelCheckpoint(
        #     filepath=model_checkpoints_output,
        #     monitor="val_mean_absolute_error",
        #     mode="min",
        #     save_best_only=True,
        #     save_weights_only=True,
        # ),
        keras.callbacks.TensorBoard(tensorboard_output / exp_name)
        # WandbCallback(),
    ],
)

# history = model.fit(
#     train_ds,
#     validation_data=test_ds,
#     epochs=epochs,
#     batch_size=batch_size,
#     callbacks=[
#         keras.callbacks.ModelCheckpoint(
#             filepath=model_checkpoints_output,
#             monitor="val_mean_absolute_error",
#             mode="min",
#             save_best_only=True,
#             save_weights_only=True,
#         ),
#         get_tensorboard_callback(tensorboard_output),
#         WandbCallback(),
#     ],
# )

# scope.__exit__()

# Save model
# model = tuner.get_best_models(num_models=1)[0]
# model.save(model_output)

# Save output
# hist = history.history
# columns = ["loss", "validation_loss", "mse", "validation_mse"]
# output_data = pd.DataFrame(zip(*[hist[i] for i in columns]), columns=columns)
# output_data.to_csv(metric_output)
