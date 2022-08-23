import tensorflow as tf
import numpy as np
from molmagic.ml import get_vector_parent, run_controller, get_model_artifact, get_vector_artifact, get_label_type
from molmagic.parser import read_sdf_archive


# variables
model_name = "glorious-sweep-59:v0"
eval_vectors_name = "qm9-heavy-bw_scott:latest"
low_error_threshold = 1.0
high_error_threshold = 10.0
label_type = "free_energy"

# Configure wandb run
run_controller.use_run("evaluation")

# Load the model
model = get_model_artifact(model_name)

# Load the generalisation dataset
evaluation_path = get_vector_artifact(eval_vectors_name)
X_eval = np.load(evaluation_path / "features.npy")
y_eval_all = np.load(evaluation_path / "labels.npy")
y_eval = get_label_type(y_eval_all, label_type)
idx_eval = np.load(evaluation_path / "identities.npy")

# Load the dataset which produced those vectors
dataset_path = get_vector_parent(eval_vectors_name)
mols = read_sdf_archive(dataset_path / "archive.sdf.bz2")

# Run inference (ERRORS HERER)
y_pred = model.predict(X_eval)
y_err = np.abs(y_pred - y_eval)

# Get a mask of molecules with error > threshold kcal/mol
large_err_mol_ids = idx_eval[y_err > high_error_threshold]
large_err_mols = filter(lambda x: x.data["id"] in large_err_mol_ids, mols)

# Analyse how errors are distributed (separate script)
