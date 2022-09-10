from model_analysis.top10_errors import main
from tqdm import tqdm
import wandb

api = wandb.Api()

runs = api.runs(filters={"tags": {"$in": ["Ridge-alpha-sweep-generalize"]}})

for run in runs:
    main(f"MolecularMagic/{run.id}")