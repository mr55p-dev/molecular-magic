from shutil import rmtree
import wandb
import pickle as p
from pathlib import Path
from molmagic.ml import get_artifact_of_kind
from tqdm import tqdm
import numpy as np


# Get all the associated runs
api = wandb.Api()
sweep = api.sweep("molecular-magicians/MolecularMagic/beckmq75")

for run in tqdm(sweep.runs):
    # Get the generated vector dataset
    # artifact = get_artifact_of_kind(
    #     run.entity + "/" + run.project + "/" + run.id, "vectors"
    # )
    # artifact_path = Path(artifact.get_path("features.npy").download())
    # vectors = np.load(artifact_path)

    run.config.update({"algorithm": "Lasso"})
    run.update()
    # artifact_path.unlink()

# Get all the associated runs
# api = wandb.Api()
# sweep = api.sweep("molecular-magicians/MolecularMagic/e9b8c8je")

# for run in tqdm(sweep.runs):
#     # Get the generated vector dataset
#     # artifact = get_artifact_of_kind(
#     #     run.entity + "/" + run.project + "/" + run.id, "model"
#     # )
#     # artifact_path = Path(artifact.download())
#     # model = artifact_path / "model" / "model"
#     # with model.open("rb") as f:
#     #     m = p.loads(f.read())
#     # run.summary.update({"alpha": m.alpha_})
#     # run.update()
#     # rmtree(artifact_path)
#     run.summary.update({"best_val_mae": run.history()["val_mae"].min()})
#     run.update()
