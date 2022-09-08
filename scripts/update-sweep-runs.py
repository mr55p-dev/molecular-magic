from shutil import rmtree
import wandb
import pickle as p
from pathlib import Path
from molmagic.ml import get_artifact_of_kind
from tqdm import tqdm


# Get all the associated runs
api = wandb.Api()
sweep = api.sweep("molecular-magicians/MolecularMagic/hqb9gzxa")

for run in tqdm(sweep.runs):
    # Get the generated vector dataset
    artifact = get_artifact_of_kind(run.entity + "/" + run.project + "/" + run.id, "model")
    artifact_path = Path(artifact.download())
    model = artifact_path / "model" / "model"
    with model.open("rb") as f:
        m = p.loads(f.read())
    run.summary.update({"alpha": m.alpha_})
    run.update()
    rmtree(artifact_path)



