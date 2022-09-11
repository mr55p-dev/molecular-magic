import pickle
from argparse import ArgumentParser, Namespace
from time import time

import numpy as np
import wandb
from molmagic.ml import (
    get_artifact_of_kind,
    get_keras_model,
    get_label_type,
    get_sklearn_model,
    get_vector_artifact,
    run_controller,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main(args: Namespace):
    api = wandb.Api()

    model_run_name = args.model
    eval_vectors_name = "qm9-test-HCNOF-0.8-0.32:v0"
    label_type = "free_energy"

    # Configure wandb run
    run = run_controller.use_run("inference")
    run.config.update(
        {"model_name": model_run_name, "eval_set_name": eval_vectors_name}
    )

    # Get the model
    basis_run = api.run(model_run_name)
    basis_model_type = basis_run.config.get("algorithm", "Keras")
    model_artifact = get_artifact_of_kind(model_run_name, "model")
    run.use_artifact(model_artifact)
    assert basis_model_type in ["Ridge", "RidgeCV", "Elastic", "AdaBoost", "Keras"]

    # Load the model
    if basis_model_type == "Keras":
        model = get_keras_model(model_artifact)
    else:
        basis_model_data = get_sklearn_model(model_artifact)
        with basis_model_data.open("rb") as model_file:
            model = pickle.loads(model_file.read())

    # Get the generalization data
    eval_vectors = get_vector_artifact(eval_vectors_name)

    # Load the generalisation vectors
    X_eval = np.load(eval_vectors / "features.npy")
    y_eval_all = np.load(eval_vectors / "labels.npy")
    y_eval = get_label_type(y_eval_all, label_type).reshape(-1, 1)

    # Run inference
    start_inference = time()
    y_pred = model.predict(X_eval).reshape(-1, 1)
    end_inference = time()
    time_to_inference = end_inference - start_inference

    mae = mean_absolute_error(y_eval.squeeze(), y_pred.squeeze())
    mse = mean_squared_error(y_eval.squeeze(), y_pred.squeeze())

    run.log({"mse": mse, "mae": mae, "inference_time": time_to_inference})
    print(f"Time to inference: {time_to_inference}")


def cli_tool():
    parser = ArgumentParser("inference_time")
    parser.add_argument("model", type=str, help="The model to evaluate")

    args = parser.parse_args()
    main(args)
