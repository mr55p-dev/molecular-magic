import oyaml as yaml
from pathlib import Path
from shutil import rmtree
import numpy as np
from sklearn.linear_model import RidgeCV
from molmagic import ml
from molmagic.parser import read_sdf_archive
from sklearn.metrics import mean_squared_error, mean_absolute_error
from molmagic import config as cfg
from molmagic.vectorizer import calculate_mol_data
from molmagic.aggregator import autobin_mols

# Collect the weights and biases
run = ml.run_controller.use_run()
run_config = run.config

# Setup the vector directory
vector_dir = Path("/tmp/vectors/")
if vector_dir.exists():
    rmtree(vector_dir) if vector_dir.is_dir() else vector_dir.unlink()

# Set the configuration options
seed = 50
split_type = "random"
target = "free_energy"
run_config.update(
    {
        "random_seed": seed,
        "splitting_type": split_type,
        "algorithm": "RidgeCV",
        "target_name": target,
    }
)

# Collect and set experiment parameters
bond_bandwidth = run_config["bond-bandwidth"]
angle_bandwidth = run_config["angle-bandwidth"]
cfg.aggregation["bond-bandwidth"] = bond_bandwidth
cfg.aggregation["angle-bandwidth"] = angle_bandwidth
cfg.plotting["save-dir"] = vector_dir

dataset_base = run_config["dataset_base"]
dataset_version = run_config["dataset_version"]
dataset_dir = ml.get_filtered_artifact(f"{dataset_base}:{dataset_version}")

# Parse the filtered base dataset
molecules = list(read_sdf_archive(dataset_dir))

# Generate the dataset
molecule_data = [calculate_mol_data(molecule) for molecule in molecules]
features, labels_raw, metadata = autobin_mols(molecule_data, plot_histograms=True)
identities = np.array([mol.data["id"] for mol in molecules]).astype(np.int32)

# Save the dataset artifact
vector_dir.mkdir(exist_ok=True)
np.save(vector_dir / "features.npy", features)
np.save(vector_dir / "labels.npy", labels_raw)
np.save(vector_dir / "identities.npy", identities)
with (vector_dir / "metadata.yml").open("w") as metadata_file:
    yaml.dump(metadata, metadata_file)

artifact_name = dataset_base + f"-{str(bond_bandwidth)}-{str(angle_bandwidth)}"
ml.log_vector_artifact(artifact_name, features, vector_dir, plot_histograms=True)

# Create train test split
labels = ml.get_label_type(labels_raw, target)
splitter = ml.get_split(split_type)
X_train, X_test, y_train, y_test = splitter(features, labels, random_state=seed)

# Run the experiment
model_base = RidgeCV()
model = model_base.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the predictions
val_mse = mean_squared_error(y_test, y_pred)
val_mae = mean_absolute_error(y_test, y_pred)

# Save the model
ml.log_sklearn_model(model)

# Log the results
# props = {"mse": mse, "mae": mae, "val_mse": val_mse, "val_mae": val_mae}
run.log({"val_mse": val_mse, "val_mae": val_mae})
