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

# Load the dataset which produced those vectors
dataset_path = get_vector_parent(...)
...

# Run inference
y_pred = model.predict(X_eval)

# Read structures from the sdf archive
mols = read_sdf_archive(dataset_path / "archive.sdf.bz2")

# Analyse how errors are distributed (separate script)
