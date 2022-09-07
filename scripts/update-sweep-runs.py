import wandb


# Get all the associated runs
api = wandb.Api()
sweep = api.sweep("molecular-magicians/MolecularMagic/lyndnrg3")

for run in sweep.runs:
    # Get the generated vector dataset
    produced_vectors = [
        artifact for artifact in run.logged_artifacts() if artifact.type == "vectors"
    ]
    assert len(produced_vectors) == 1
    vectors = produced_vectors[0]

    # Get the number of instances and features
    n_features = vectors.metadata["n_features"]
    n_instances = vectors.metadata["n_instances"]

    # Update the run config and save
    run.summary.update({"n_features": n_features, "n_instances": n_instances})
    run.update()



