# %%
import tensorflow as tf
import numpy as np
from molmagic.ml import (
    get_vector_parent,
    run_controller,
    get_model_artifact,
    get_vector_artifact,
    get_label_type,
)
from molmagic.vectorizer import _search_substructure
from molmagic.parser import read_sdf_archive
import seaborn as sns

# %% variables
model_name = "glorious-sweep-59:v0"
eval_vectors_name = "qm9-heavy-bw_scott:latest"
low_error_threshold = 1.0
high_error_threshold = 1000.0
label_type = "free_energy"

# %% Configure wandb run
run_controller.use_run("evaluation")

# Load the model
model = get_model_artifact(model_name)

# Load the generalisation dataset
evaluation_path = get_vector_artifact(eval_vectors_name)
X_eval = np.load(evaluation_path / "features.npy")
y_eval_all = np.load(evaluation_path / "labels.npy")
y_eval = get_label_type(y_eval_all, label_type).reshape(-1, 1)
idx_eval = np.load(evaluation_path / "identities.npy").reshape(-1, 1)

# # Load the dataset which produced those vectors
dataset_path = get_vector_parent(eval_vectors_name)
mols = read_sdf_archive(dataset_path / "archive.sdf.bz2")

# %% Run inference
y_pred = model.predict(X_eval)
y_err = np.abs(y_pred.reshape(-1, 1) - y_eval)
mae = y_err.mean()
print(y_err.mean())

# %% Get a mask of molecules with error > threshold kcal/mol
large_err_mask = y_err > high_error_threshold
large_err_mol_ids = idx_eval[large_err_mask]
large_err_mols = [
    (mol, err) for mol, id, err in zip(mols, idx_eval, y_err) if id in large_err_mol_ids
]
# frame = pd.DataFrame(large_err_mols, columns=["mol_id", "error"])
# frame.to_csv("./frame.csv")
# %% Define ha counter
def count_ha(mol):
    return len([i for i in mol.atoms if i.atomicnum != 1])


# %% Analyse how errors are distributed (separate script)
amide_counts = [
    _search_substructure(i[0].OBMol, "[NX3][CX3](=[OX1])[#6]") for i in large_err_mols
]
aromatic_carbon_counts = [_search_substructure(i[0].OBMol, "c") for i in large_err_mols]
aromatic_nitrogen_counts = [_search_substructure(i[0].OBMol, "n") for i in large_err_mols]
n_heavy_atoms = [count_ha(i[0]) for i in large_err_mols]
# %%
errors = y_err[large_err_mask]
sns.histplot(errors)
# %%
sns.barplot(x=aromatic_carbon_counts, y=errors)
# %%
sns.barplot(x=aromatic_nitrogen_counts, y=errors)

# %%
