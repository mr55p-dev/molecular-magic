# %%
from argparse import ArgumentParser, Namespace
from rdkit import Chem
from shutil import copy, rmtree
import oyaml as yaml
from pathlib import Path
import pickle
import numpy as np
from molmagic.ml import (
    get_artifact_of_kind,
    get_filtered_artifact,
    get_sklearn_model,
    get_vector_metadata,
    log_test_vector_artifact,
    run_controller,
    get_keras_model,
    get_label_type,
)
from molmagic.parser import convert_pb_to_rdkit, read_sdf_archive
from molmagic.aggregator import bin_mols
import pandas as pd
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error

from molmagic.vectorizer import calculate_mol_data


def main(args: Namespace):
    api = wandb.Api()
    # variables
    model_run_name = "MolecularMagic/xq9zfdf6"  # The run which generated the model
    eval_dataset_name = "qm9-test-HCNOF:v0"
    label_type = "free_energy"

    # Configure wandb run
    run = run_controller.use_run("evaluation")

    # Identify the generated model artifact and type of experiment
    model_artifact = get_artifact_of_kind(model_run_name, "model")
    model_run = api.run(model_run_name)
    model_type = model_run.config.get("algorithm", "Keras")
    assert model_type in ["RidgeCV", "AdaBoost", "Keras"]

    # Load the model
    if model_type == "RidgeCV":
        model_data = get_sklearn_model(model_artifact)
        with model_data.open("rb") as model_file:
            model = pickle.loads(model_file.read())
    elif model_type == "Keras":
        model = get_keras_model(model_artifact)

    # Create the generalisation dataset
    # 1. Get the vectors artifact produced by this run
    train_vectors = get_artifact_of_kind(model_run_name, "vectors")
    train_metadata = get_vector_metadata(train_vectors)

    # Extract the data and metadata from the config
    with train_metadata.open("r") as fi:
        constructor = yaml.load(fi, Loader=yaml.CLoader)
    data = constructor["data"]
    metadata = constructor["metadata"]

    # 2. Get the filtered dataset generalisation artifact
    eval_set = get_filtered_artifact(eval_dataset_name)
    molecules = list(read_sdf_archive(eval_set))
    molecule_data = [calculate_mol_data(mol) for mol in molecules]
    features, labels = bin_mols(molecule_data, data, metadata)
    identities = np.array([mol.data["id"] for mol in molecule_data]).astype(np.int32)

    # 3. Save everything
    output_dir = Path("/tmp/vectors")
    if output_dir.is_dir():
        rmtree(output_dir)
    elif output_dir.is_file():
        output_dir.unlink()
    output_dir.mkdir()

    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "identities.npy", identities)
    copy(train_metadata, output_dir / "metadata.yml")

    eval_vectors_name = eval_dataset_name.split(":")[0]
    eval_vectors_name += f"-{model_run.config['bond-bandwidth']}"
    eval_vectors_name += f"-{model_run.config['bond-bandwidth']}"
    log_test_vector_artifact(eval_vectors_name, features, output_dir)

    # "Load" the generalisation dataset
    X_eval = features
    y_eval_all = labels
    y_eval = get_label_type(y_eval_all, label_type).reshape(-1, 1)
    idx_eval = identities.reshape(-1, 1)

    # Run inference
    y_pred = model.predict(X_eval).reshape(-1, 1)
    err = y_pred - y_eval
    abs_err = np.abs(err)
    mae = mean_absolute_error(y_eval, y_pred)
    mse = mean_squared_error(y_eval, y_pred)

    run.log({"mse": mse, "mae": mae})

    # Create a table of absolute errors
    err_frame = pd.DataFrame(
        np.concatenate((y_eval, y_pred, abs_err), axis=1),
        columns=["Actual value", "Predicted value", "Absolute Error"],
    )
    err_table = wandb.Table(data=err_frame)
    err_hist = wandb.plot.histogram(
        err_table, "Absolute Error", title="Prediction error distribution"
    )

    run.log({"Error table": err_table, "Error plot": err_hist})

    # Get top10 error molecules
    top10_idx = np.argsort(abs_err, axis=0)[-10:, 0]
    mol_err_iterator = zip(molecules, idx_eval.squeeze(), abs_err.squeeze())
    top10_mols, top10_ids, top10_errs = zip(
        *[
            (mol, id, err)
            for mol, id, err in mol_err_iterator
            if id in idx_eval.squeeze()[top10_idx]
        ]
    )

    # Create an error artifact and attach the structures
    err_artifact = wandb.Artifact(run.name, "errors")

    rdkit_mols = [Chem.rdmolops.AddHs(mol) for mol in convert_pb_to_rdkit(top10_mols)]
    artifact_mols = [
        wandb.data_types.Molecule.from_rdkit(
            mol,
            caption=f"Moleucle id: {id}, error: {err}",
        )
        for mol, id, err in zip(rdkit_mols, top10_ids, top10_errs)
    ]

    for artifact_mol, id in zip(artifact_mols, top10_ids):
        err_artifact.add(artifact_mol, str(id))

    # Create an error table
    err_data = pd.DataFrame(
        zip(*[top10_ids, top10_errs]),
        columns=["Molecule ID", "Error (Kcal/mol)"],
    )
    err_table = wandb.Table(data=err_data)
    err_artifact.add(err_table, "errors")

    # Log the artifact
    run.log_artifact(err_artifact)


if __name__ == "__main__":
    main(...)
    # parser = ArgumentParser()
    # parser.add_argument("model", type=str, help="The model which is to be tested")
    # parser.add_argument(
    #     "test_set",
    #     type=str,
    #     help="The test dataset to be vectorized by the scheme of the model. Should be of type FilteredDataset",
    # )
