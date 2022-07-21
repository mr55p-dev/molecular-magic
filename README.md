# Molecular Magic

Working repository for modifications to the work done in https://github.com/sanha0213/MolE8
****
## Setup

### Basic environment setup
- Run `conda env install -f environment.yml` to setup the environment.
- Run `conda activate molemagic` to activate the environment.
- Install Git LFS (https://git-lfs.github.com/) and run the following commands in the local git folder:
  - `git lfs install`
  - `git lfs fetch`
  - `git lfs pull`

### Utility setup
You can install the `molmagic` cli tool to make the dataset creation a little easier.
Run `pip install -e .` to install the `molmagic` script locally.


## `molmagic` usage
Run `molmagic -h` for a list of subcommands. The database cleaning can be invoked with `molmagic parser -i <input directory> -o <output archive>`.
## Tests
Tests can be run with pytest (`python3 -m pytest`).
## Weights and biases
Weights and biases is used to record experiments. In order to perform and track an experimient, please sign into weights and biases locally (see [documentation](https://docs.wandb.ai/)).
The workspace which contains runs and information is hosted [here](https://wandb.ai/molecular-magicians/MolecularMagic).

# Legacy code
`clean_database.py` takes roughly 4 minutes
`create_features.py` takes roughly 10 minutes.
`NN.py` takes on order of 5-10 hours.
## Openbabel
`openbabel` used in the project seems to be version `2.x.x` however this can't be verified due to no requirements file being included. Instead of using this it seems better to migrate to version `3.1.1` and moving forward take advantage of the `pybabel` API rather than the auto-generated C++ bindings. Some corrections which appear to be related to the version change are being made to the original code.

## Visualisation of Results
- Tensorboard can be installed to the conda environment using the following command:
  - `conda install -c conda-forge tensorboard`
- Tensorboard logs can be visualised using the following command:
  - `tensorboard --logdir=./`
- NN logs are store in `./static_data/NN_rewrite`.
- TO-DO: Define weights and biases setup and usage

----

# Reimplementation

## Create cleaned annotated sdf files
Run the script `clean.py` with its available arguments: