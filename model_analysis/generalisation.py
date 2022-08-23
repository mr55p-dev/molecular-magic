import tensorflow as tf
import numpy as np
from molmagic.ml import get_vector_parent, run_controller, get_model_artifact, get_vector_artifact
from molmagic.parser import read_sdf_archive


# Configure wandb run
run_controller.use_run("evaluation")

# Load the model
model = get_model_artifact(...)

# Load the generalisation dataset
evaluation_path = get_vector_artifact(...)
X_eval = np.load(evaluation_path / "features.npy")
y_eval = np.load(evaluation_path / "labels.npy")
idx_eval = np.load(evaluation_path / "identities.npy")

# Load the dataset which produced those vectors
dataset_path = get_vector_parent(...)
...

# Run inference
y_pred = model.predict(X_eval)
y_err = np.abs(y_pred - y_eval)

# Read structures from the sdf archive
mols = read_sdf_archive(dataset_path / "archive.sdf.bz2")

# Get a mask of molecules with error > threshold kcal/mol
threshold = 10.0
large_err_mol_ids = idx_eval[y_err > threshold]
large_err_mols = filter(lambda x: x.data["id"] in large_err_mol_ids, mols)

# Analyse how errors are distributed (separate script)
