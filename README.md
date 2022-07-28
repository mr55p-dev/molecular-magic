# Molecular Magic

Working repository for the rewriting of the code in: https://github.com/sanha0213/MolE8

Work is performed as part of the University of Nottingham Machine Learning in Science MSc project of Luke Rawlence and Ellis Lunnon.
****
# MolE8 Reimplementation

## Setup

### Virtual Environment
- Run `conda env create -f environment.yml` to setup the environment.
- Run `conda activate molemagic` to activate the environment.


### MolMagic CLI
- You can install the `molmagic` cli tool to make the dataset creation easier.
- Run `pip install -e .` to install the `molmagic` script locally.
- From a local terminal session, invoke the `molmagic` command to see allt he available subcommands

### Dataset
- Go to the [University of Nottingham repository](https://unow.nottingham.ac.uk/handle/internal/9356?show=full) and download one or more dataset zip files (p1 to p7).
- Create a folder called `moldata` and extract the files to this folder

----

## Running the scripts

### 0. Install the CLI tool
- Clone this repo and install the anaconda environment with `codna env create -f environment.yml`. This downlads the correct version of python and all its dependencies.
  - Activate the environment with `conda activate molemagic`
- Once this is done, install the `magic` module locally using pip. Run the command `pip install -e .` to install the script endpoints in your environment.
  - Running the command `molmagic` in your terminal should print out the help message for the tool
### 1. Create cleaned annotated sdf files
- Run `molmagic parser -i <path_to_moldata_directory> -o <path_to_output_file>`, where `<path_to_output_file> specifys the path to a file which does not yet exist
  - The input directory should contain frequency files for all the structures to be analysed. The parser command will walk through all subdirectories of the one specified so there is no need to unpack every file into the same directory.
  - Note that the order in the output is not guaranteed to be the same as the input (and is not likely to be either).
  - The output file you provide will be appended with the extension `.sdf.bz2`
  - Running the command `molmagic parser -i ./moldata -o ./cleaned_data` will result in a reading all `g09` frequency files from `moldata` and writing molecular structures and energies into a `bz2` archive using the `sdf` format, `./cleaned_data.sdf.bz2`.
- Running this command should display a progress bar, and after a short duration the command should exit, leaving behind the file specified after the `-o` flag in your command

### 2. Convert the structure archive into vectors
- The utility `molmagic vectorizer` can be used to convert a `.sdf.bz2` archive into saved numpy vectors (`.npx` files). Run `molmagic vectorizer -h` to see the required arguments.
- The syntax is the same as for the parser; use the `-i` flag to specify the input file (in this case it should be `path/to/output/of/parser.sdf.bz2`) and the `-o` flag to specify a directory. This directory will be created if it does not exist. There will be two new files created in that directory specified, `features.npy` and `labels.npy`. Labels will be extracted based on the **sdf key** speficied in `config.yml`.

----

## Additional Information
### `molmagic` usage
- Run `molmagic -h` for a list of subcommands.
- The database cleaning can be invoked with `molmagic parser -i <input directory> -o <output archive>`.

### Tests
- Tests can be run with pytest (`python3 -m pytest`).

### Weights and biases
- Weights and biases is used to record experiments. In order to perform and track an experimient, please sign into weights and biases locally (see [documentation](https://docs.wandb.ai/)).
The workspace which contains runs and information is hosted [here](https://wandb.ai/molecular-magicians/MolecularMagic).


# Legacy code
## Script Information
- `clean_database.py` takes roughly 4 minutes
- `create_features.py` takes roughly 10 minutes.
- `NN.py` takes on order of 5-10 hours.
## Openbabel
`openbabel` used in the project seems to be version `2.x.x` however this can't be verified due to no requirements file being included. Instead of using this it seems better to migrate to version `3.1.1` and moving forward take advantage of the `pybabel` API rather than the auto-generated C++ bindings. Some corrections which appear to be related to the version change are being made to the original code.

<!-- - Install Git LFS (https://git-lfs.github.com/) and run the following commands in the local git folder:
  - `git lfs install`
  - `git lfs fetch`
  - `git lfs pull` -->

<!-- ## Visualisation of Results
- Tensorboard can be installed to the conda environment using the following command:
  - `conda install -c conda-forge tensorboard`
- Tensorboard logs can be visualised using the following command:
  - `tensorboard --logdir=./`
- NN logs are store in `./static_data/NN_rewrite`.
- TO-DO: Define weights and biases setup and usage -->