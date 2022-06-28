from collections import defaultdict
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from keras import initializers
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

AtomOrder = ["C", "H", "N", "O"]

# Data loading
Random_State = 50

# Model parameters
model_name = "NN_Energies"
model_output_dir = Path("static_data/NN_output/TrialA1/")

# Hyperparam
learning_rate = 1e-5
decay_rate = 3e-5
batch_size = 64
epochs = 7000

# Network params
activation_function = "linear"
activation_function2 = "relu"
loss_function = "mean_squared_error"
optimizer_type = Adam(learning_rate=learning_rate)

# Weight initializations
kernel_initialiser = initializers.RandomUniform(minval=-500, maxval=100)
kernel_initialiser2 = initializers.RandomUniform(minval=0, maxval=0.1)

bias_initialiser = initializers.RandomUniform(minval=0, maxval=10)
bias_initialiser2 = initializers.RandomUniform(minval=0, maxval=0.01)

# -- Import feature and target vectors
data_basepath = Path("static_data/create_features_output/data")
X_load = np.load(data_basepath / "features.npy", allow_pickle=True)
y_load = np.load(data_basepath / "labels.npy", allow_pickle=True)

# Round the targets to 5 dp (why?)
y_rounded = np.round(y_load, 5)

# X_load = X_load.tolist()
# y_load = y_load.tolist()
y_load_rounded = y_rounded.tolist()


def create_mol_formula(AtomList) -> str:
    return "".join(
        map(lambda x: f"{x}{AtomList[AtomOrder.index(x)]}", ["C", "N", "O", "H"])
    )


AtomSet = (X_load[:, -15:-11] / 100).astype("int")
molecular_formulae = list(map(create_mol_formula, AtomSet))
Unique_Mol_Forms = set(molecular_formulae)

X_dict = defaultdict(list)
y_dict = defaultdict(list)
for form, vec, target in zip(molecular_formulae, X_load, y_load):
    X_dict[form].append(vec)
    y_dict[form].append(target)

X_dict = dict(**X_dict)
y_dict = dict(**y_dict)

X_train = []
X_test = []
y_train = []
y_test = []

# Custom splitting method
for MolForm in Unique_Mol_Forms:

    X_MolFormList = X_dict[MolForm]
    y_MolFormList = y_dict[MolForm]

    if len(y_MolFormList) == 1:
        pass
    elif len(y_MolFormList) == 2:
        X_MolFormTrain, X_MolFormTest, y_MolFormTrain, y_MolFormTest = train_test_split(
            X_MolFormList, y_MolFormList, test_size=0.5, random_state=Random_State
        )
        X_train += X_MolFormTrain
        X_test += X_MolFormTest
        y_train += y_MolFormTrain
        y_test += y_MolFormTest
    else:
        X_MolFormTrain, X_MolFormTest, y_MolFormTrain, y_MolFormTest = train_test_split(
            X_MolFormList, y_MolFormList, test_size=0.33, random_state=Random_State
        )
        X_train += X_MolFormTrain
        X_test += X_MolFormTest
        y_train += y_MolFormTrain
        y_test += y_MolFormTest

### Checking the energies are properly assigned
removed_energies = []

for index1 in range(len(y_load)):

    # # print(index1)

    X_vector = X_load[index1]
    y_energy = y_load[index1]
    # # print(y_energy)

    if y_energy in y_train:
        index2 = y_train.index(y_energy)
        X_vector_test = X_train[index2]

        if (X_vector != X_vector_test).all():
            print("ERROR: X_vector does not match X_vector_test")
            print("X_vector: ", X_vector)
            print("X_vector_test: ", X_vector)
            raise ValueError("ERROR: X_vector does not match X_vector_test")

    elif y_energy in y_test:
        index2 = y_test.index(y_energy)
        X_vector_test = X_test[index2]

        if (X_vector != X_vector_test).all():
            print("X_vector: ", X_vector)
            print("X_vector_test: ", X_vector)
            raise ValueError("ERROR: X_vector does not match X_vector_test")

    else:
        removed_energies += [y_energy]

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

X_trainmean = X_train.mean()
X_trainstd = X_train.std()
y_trainmean = y_train.mean()
y_trainstd = y_train.std()

X_trainscaled = X_train / X_trainstd
X_testscaled = X_test / X_trainstd

y_testmatrix = []
y_trainmatrix = []

for element in y_train:
    y_trainmatrix.append([element])

for element in y_test:
    y_testmatrix.append([element])

y_trainmatrix = np.array(y_trainmatrix)
y_testmatrix = np.array(y_testmatrix)

###############################################################################
# Construct the NN architecture
###############################################################################

no_features = len(X_trainscaled[0])
# print("")
# print("Number of features detected: ", str(no_features))

# logging.info("")
# logging.info("Number of features detected: " + str(no_features))

# model set up: build architecture
model = Sequential(name=model_name)

model.add(Dense(761, \
                input_dim=no_features, \
                activation=activation_function2, \
                kernel_initializer=kernel_initialiser2, \
                bias_initializer=bias_initialiser2, \
                kernel_regularizer=l2(0.1), \
                bias_regularizer=l2(0.1), \
                name='layer1'))

#model.add(Dropout(0.2))

model.add(Dense(761, \
                activation=activation_function2, \
                kernel_initializer=kernel_initialiser, \
                bias_initializer=bias_initialiser, \
                kernel_regularizer=l2(0.1), \
                bias_regularizer=l2(0.1), \
                name='layer2'))

#model.add(Dropout(0.2))

model.add(Dense(1, \
                activation=activation_function, \
                kernel_initializer=kernel_initialiser, \
                bias_initializer=bias_initialiser, \
                kernel_regularizer=l2(0.1), \
                bias_regularizer=l2(0.1), \
                name='layer3'))

print("Settings for neural network training")
print(model.summary())

modelsummarystringlist = []
model.summary(print_fn=lambda x: modelsummarystringlist.append(x))

model.compile(
    loss=loss_function,
    optimizer=optimizer_type,
    metrics=["mean_squared_error", "mean_absolute_error"],
)


###############################################################################
# model train
###############################################################################

model_history = model.fit(
    X_trainscaled,
    y_trainmatrix,
    validation_data=(X_testscaled, y_testmatrix),
    epochs=epochs,
    batch_size=batch_size,
    verbose=2,
)


with open(model_output_dir / "errordata.csv", "w") as errorfile:
    error_write = csv.writer(errorfile)
    epoch_data = range(0, len(model_history.history["loss"]))
    csv_rows = zip(
        epoch_data,
        model_history.history["loss"],
        model_history.history["val_loss"],
        model_history.history["mean_squared_error"],
        model_history.history["val_mean_squared_error"],
    )
    error_write.writerows(csv_rows)


# Evaluate Keras Model
train_error = model.evaluate(X_trainscaled, y_trainmatrix, verbose=0)
test_error = model.evaluate(X_testscaled, y_testmatrix, verbose=0)

print("train_mse: ", train_error)
print("test_mse: ", test_error)

predictions = model.predict(X_testscaled, batch_size=batch_size)

unwrapped_predictions = []

for pred_list in predictions:
    unwrapped_predictions += [pred_list[0]]

error_dataframe = pd.DataFrame(
    data={
        "Actual Energy [kcal/mol]": y_test,
        "Predicted Energy [kcal/mol]": unwrapped_predictions,
    }
)

error_dataframe["error"] = abs(
    error_dataframe["Actual Energy [kcal/mol]"]
    - error_dataframe["Predicted Energy [kcal/mol]"]
)
error_dataframe = error_dataframe.sort_values(["error"], ascending=[False])

data_largeerror = error_dataframe[error_dataframe.error > 0.0]

print("Largest absolute errors in kcal mol:")
print(data_largeerror)

# error_indexes will contain the index of the structures giving large error from the original y vector

print("looping over all predictions where the error is greater than 1 kcal mol")

error_indexes = []

for df_index, df_row in data_largeerror.iterrows():
    y_rowenergy = df_row["Actual Energy [kcal/mol]"]
    error_indexes += [y_load_rounded.index(round(y_rowenergy, 5))]

error_array = np.array(error_indexes)
data_largeerror["y_index"] = error_array

data_largeerror.to_csv(model_output_dir / "largeerrors.csv")

# -- save keras model
model.save(model_output_dir / "model")

print("y_testunscaled:")
print(y_testmatrix)
print("predictions:")
print(predictions)
print("Training completed")
