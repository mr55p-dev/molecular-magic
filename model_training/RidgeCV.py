from sklearn.linear_model import RidgeCV
import numpy as np
from sklearn.metrics import mean_absolute_error
from molmagic import ml

# Configuration
training_data_artifact = "qm9-std_scott:latest"
label_name = "electronic_energy"
splitting_type = "random"
random_seed = 50

# Load data
data_basepath = ml.get_vector_artifact(training_data_artifact)
X = np.load(data_basepath / "features.npy")
y_raw = np.load(data_basepath / "labels.npy").astype(np.double)
y = ml.get_label_type(y_raw, label_name)

# Split training and testing data
splitter = ml.get_split(splitting_type)
X_train, X_test, y_train, y_test = splitter(X, y, random_state=random_seed)

# Fit a model
reg = RidgeCV()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Save the results
run = ml.run_controller.use_run()
run.config.update({
    "target_name": label_name,
    "splitting_type": splitting_type,
    "algorithm": "RidgeCV"
})
run.log({
    "mean_absolute_error": mae,
})
ml.log_model(reg)

print(mae)