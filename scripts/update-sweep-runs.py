from shutil import rmtree
import wandb
import pickle as p
from pathlib import Path
from molmagic.ml import get_artifact_of_kind
from tqdm import tqdm


# Get all the associated runs
api = wandb.Api()
# runs = api.runs(filters={"tags": {"$in": ["Neural-Network"]}})
sweep = api.sweep("molecular-magicians/MolecularMagic/9iguqpio")

for run in tqdm(sweep.runs):
    # Get the generated vector dataset
    # artifact = get_artifact_of_kind(
    #     run.entity + "/" + run.project + "/" + run.id, "model"
    # )
    # artifact_path = Path(artifact.download())
    # model = artifact_path / "model" / "model"
    # with model.open("rb") as f:
    #     m = p.loads(f.read())
    # run.summary.update({"alpha": m.alpha_})
    # run.update()
    # rmtree(artifact_path)
    run.summary.update({"best_val_mae": run.history()["val_mae"].min()})
    run.update()
