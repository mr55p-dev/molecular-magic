import pickle
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
import wandb
from molmagic.ml import (
    get_artifact_of_kind,
    get_keras_model,
    get_label_type,
    get_sklearn_model,
    get_vector_artifact,
    get_vector_parent,
    run_controller,
)
from molmagic.parser import convert_pb_to_rdkit, read_sdf_archive
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main(model_run_name: str):
    api = wandb.Api()
    # variables
    eval_vectors_name = "qm9-test-HCNOF-0.8-0.32:latest"
    label_type = "free_energy"

    # Configure wandb run
    run = run_controller.use_run("evaluation")

    # Get the model
    model_artifact = get_artifact_of_kind(model_run_name, "model")
    model_run = api.run(model_run_name)
    model_type = model_run.config.get("algorithm", "Keras")
    assert model_type in ["Ridge", "RidgeCV", "Elastic", "AdaBoost", "Keras"]

    # Load the model
    if model_type == "Keras":
        model = get_keras_model(model_artifact)
    else:
        model_data = get_sklearn_model(model_artifact)
        with model_data.open("rb") as model_file:
            model = pickle.loads(model_file.read())

    # Get the generalization data
    eval_vectors = get_vector_artifact(eval_vectors_name)
    eval_dataset = get_vector_parent(eval_vectors_name)

    # Load the generalisation vectors
    X_eval = np.load(eval_vectors / "features.npy")
    y_eval_all = np.load(eval_vectors / "labels.npy")
    y_eval = get_label_type(y_eval_all, label_type).reshape(-1, 1)
    idx_eval = np.load(eval_vectors / "identities.npy").reshape(-1, 1)

    # Run inference
    y_pred = model.predict(X_eval).reshape(-1, 1)
    err = y_pred - y_eval
    abs_err = np.abs(err)
    mae = mean_absolute_error(y_eval.squeeze(), y_pred.squeeze())
    mse = mean_squared_error(y_eval.squeeze(), y_pred.squeeze())

    # Create an error table
    err_data = pd.DataFrame(
        zip(*[idx_eval, y_eval, y_pred, abs_err]),
        columns=[
            "Molecule ID",
            "Free energy (Kcal/mol)",
            "Predicted energy (Kcal/mol)",
            "Error (Kcal/mol)",
        ],
    )
    err_table = wandb.Table(data=err_data)
    err_hist = wandb.plot.histogram(
        err_table, value="Error (Kcal/mol)", title="Prediction error distribution"
    )

    run.log(
        {"mse": mse, "mae": mae, "Error table": err_table, "Error histogram": err_hist}
    )

    # Create a table of absolute errors
    err_frame = pd.DataFrame(
        np.concatenate((y_eval, y_pred, abs_err), axis=1),
        columns=["Actual value", "Predicted value", "Absolute Error"],
    )
    top10_err_table = wandb.Table(data=err_frame)
    top10_err_hist = wandb.plot.histogram(
        top10_err_table, "Absolute Error", title="Prediction error distribution"
    )

    run.log({"Error table": top10_err_table, "Error plot": top10_err_hist})

    # Load the evaluation dataset
    molecules = read_sdf_archive(eval_dataset)

    # Get top10 error molecules
    top10_idx = np.argsort(abs_err, axis=0)[-10:, 0]
    mol_err_iterator = zip(
        molecules,
        idx_eval.squeeze(),
        abs_err.squeeze(),
        y_pred.squeeze(),
        y_eval.squeeze(),
    )
    top10_mols, top10_ids, top10_errs, top10_preds, top10_labels = zip(
        *[
            (mol, id, err, pred, label)
            for mol, id, err, pred, label in mol_err_iterator
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
    top10_err_data = pd.DataFrame(
        zip(*[top10_ids, top10_labels, top10_preds, top10_errs]),
        columns=[
            "Molecule ID",
            "Free energy (Kcal/mol)",
            "Predicted energy (Kcal/mol)",
            "Error (Kcal/mol)",
        ],
    )
    top10_err_table = wandb.Table(data=top10_err_data)
    top10_err_hist = wandb.plot.histogram(
        top10_err_table,
        "Error (Kcal/mol)",
        "Distribution of prediction errors for top 10 molecules",
    )
    err_artifact.add(top10_err_table, "Top10 Error table")
    run.log({"Top 10 error distribution": top10_err_hist})

    # Log the artifact
    run.log_artifact(err_artifact)
    run.finish()


def cli_tool():
    parser = ArgumentParser("top10_generalization")
    parser.add_argument("run_name", type=str)

    args = parser.parse_args()
    main(args.run_name)

    # parser = ArgumentParser()
    # parser.add_argument("model", type=str, help="The model which is to be tested")
    # parser.add_argument(
    #     "test_set",
    #     type=str,
    #     help="The test dataset to be vectorized by the scheme of the model. Should be of type FilteredDataset",
    # )
