import bz2
from pathlib import Path

import wandb
from rdkit import Chem

from molmagic.ml import run_controller

# Setup archive path
archive_path = Path("data/MolE8/heavy.sdf.bz2")

# Identify and parse SDF file
with bz2.BZ2File(archive_path, mode="rb") as archive:
    contents = archive.read().decode("utf-8")

supplier = Chem.rdmolfiles.SDMolSupplier()
supplier.SetData(contents)

# Create an artifact
mol = wandb.data_types.Molecule.from_rdkit(supplier[1])
run = run_controller.use_run("test")
artifact = wandb.Artifact("test", "model")
artifact.add(mol, "test")
run.log_artifact(artifact)


# View this!
print("Hello")