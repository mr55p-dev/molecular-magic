# Molecular Magic

This project is a toolkit for generating fixed-length data-driven molecular representations. It reimplements the tools described by the authors of MolE8, with enhancements to the implementation, observability and flexibility of the data creation process.

Work is performed as part of the University of Nottingham Machine Learning in Science MSc project of Luke Rawlence and Ellis Lunnon.
****
</br>


# Setup
- Clone this repo and install the anaconda environment with `conda env create -f environment.yml`. This downlads the correct version of python and all its dependencies.
- Activate the environment with `conda activate molmagic`
- Once this is done, install the `molmagic` module locally using pip. Run the command `pip install -e .` to install the script endpoints in your environment.
- Running the command `magic` in your terminal should print out the help message for the tool
- Download the dataset zip files from the [University of Nottingham repository](https://unow.nottingham.ac.uk/handle/internal/9356?show=full).
  - Create a folder called `moldata` in the repo folder and extract the files there.
# Weights and biases
Weights and biases is used to record experiments and track dataset dependencies. In order to perform and track an experimient, please sign into weights and biases locally (see [documentation](https://docs.wandb.ai/)).
The workspace which contains runs and information is hosted [here](https://wandb.ai/molecular-magicians/MolecularMagic).
Note the current implementation uses a hard-coded project entity, which needs to be set in the `molmagic.ml` submodule.

# CLI-tool usage
## Parse
  - Use the `magic parser` command to convert a set of files into an `sdf.bz2` archive.
  - This operation applies some basic filters to ensure that there are no fragmented instances included (checks for `.` in the SMILES string).
  - Can output to a local file and a weights and biases `dataset` artifact.
  - Currently supports the following input formats:
    - Directory of `g09`-formatted gaussian16 frequency calculation files. Each file should match the glob `/**/*f.out` of the directory passed.
    -`.xyz.tar` archive. Specifically, this refers to the format compatable with the `qm9` dataset when downloaded from [here](https://www.nature.com/articles/sdata201422).
  - Note the output archive can be decompressed into a single `sdf` file using the `bzip2`-cli tool.
## Filter
  - Use the `magic filter` command to filter a parsed `.sdf.bz2` archive or artifact based on the rules specified in `config.yml`
  - This operation can generate a local `.sdf.bz2` archive and a weights and biases `dataset` archive.
  - The input can be specified as a `dataset` artifact or a local `.sdf.bz2` file.
## Vectorize
  - Use the `magic vectorize` command to convert a set of instances into vectors.
  - This uses the vectorizing rules specified in `config.yml`.
  - The `--plot-histograms` command will output figures to the directory specified in the `config.yml`.
  - This can load a `dataset` artifact or local `.sdf.bz2` file.
  - If the `--metadata` or `--remote-metadata` flags are passed, the bins from a previous run will be loaded and used to assign the molecules specified in the input.
  - Else the bins will be computed based on **every instance passed**.
  - This can output a `vectors` artifact or directory containing:
    - `features.npy`: Feature matrix
    - `labels.npy`: Label matrix. Currently, `[:, 0]` is the electronic (scf) energy and `[:, 1]` is the free energy, both in `kcal/mol`.
    - `identities.npy`: Column vector mapping each row of the feature matrix to an `id` in the dataset.
    - `metadata.yml`: Snapshot of the `config.yml` used to create this run, as well as the ordering and boundaries of the bins generated (only created if metadata is loaded from another file).
## Train
  - The `molmagic.ml` API provides hooks to create a weights and biases run, fetch or log `dataset`, `vector` and `model` artifacts
  - Exampe usage is in `model_training/MLP_prototype.py`.
- ## Evaluate
  - Again, the `molmagic.ml` API should be used in this case - model dependencies can be properly understood this way.
  - Evaluation sets can be generated relatively easily using the `parse`, `filter` and `vectorize` utilities (specifically using the `metadata` arguments of `vectorize`).


# Tool configuration with `config.yml`
Much of the configuration for feature generation is controlled via `config.yml`. The properties are:
## Extraction
### Filter configuration
- `output-format`: The format to save instances as internally in the output bz2 archive. This should not be changed from `sdf` as nothing else is implemented.
`bond-min-distance`: The minimum distance between all atoms. If a molecule contains a bond shorter than this, it is filtered out under the rule `long bonds`.
- `bond-max-distance`: The maximum distance between all atoms. See above
- `min-heavy-atoms`: The minimum number of heavy atoms allowed in a molecule (inclusive). Heavy atoms count as any atom with an atomic number greater than 1.
- `max-heavy-atoms`: The maximum number of heavy atoms allowed in a molecule (inclusive). Heavy atoms count as any atom with an atomic number greater than 1.
## Extraction configuration
- `hbond-atoms`: The atom types (given as atomic numbers) which can take part in hydrogen bonding. Default behaviour is Nitrogen (7) and Oxygen (8)
- `hbond-min-distance`: The minimum distance (in angstrom) between a hydrogen donor and hbond acceptor for a bond to be considered a hydrogen bond
- `hbond-max-distance`: The maximum distance (in angstrom) between a hydrogen donor and hbond acceptor for a bond to be considered a hydrogen bond
- `substructures`: A list of SMARTS strings. This list gets counted for each molecule and added to the total feature vector. See [here](https://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html) for some common examples of functional groups which can be addressed in this way.
## Aggregation
- `resolution`: The number of samples to use when sampling the KDE for computing minima.
- `bandwidth`: The bandwidth to use when generating histograms. Can either be a positive scalar, 'scott' for the Scott method, 'silverman' for the Silverman method of automatic bandwidth selection or empty for the default (Scott method).
- `use-minmax`: Should the KDE estimation use the min and max of the data supplied when generating the histograms, or take a guess at sensible values using some preset defintions?
- `weighted-bins`: If this is enabeld, given bin boundaries of [10, 20] and
- `atom-types`: Each atom (atomic number) listed here will be counted and included as part of the representation. By default this is `[1, 6, 7, 8]` meaning `[Hydrogen, Carbon, Nitrogen, Oxygen]`. The elemnents `[0:4]` of the feature will therefore correspond to the frequency of those atoms in the order given. Also controlls the behaviour of filtering if that is enabled.
- `amine-types`: The degrees of amine which are counted as part of the representation. By default, primary (`1`), secondary (`2`) and tertiary (`3`) amines are counted.
- `feature-types`: Which features to use when generating the histograms. Allowed values are `bonds`, `angles`, `dihedrals` and `hbonds`. More may be added in the future.
## Plotting
- `save-dir`: The directory to write figures to. This path will be created if it does not exist.
- `show-dist`: Plot the histogram shape (`sns.histplot`).
- `show-kde`: Draw a lineplot of the sampled kernel density estiamte.
- `plot-style`: Styling parameter passed to `sns.set_style`.
# Error messages
The following Openbabel error messages can be ignored:
- `Failed to kekulize aromatic bonds in OBMol::PerceiveBondOrders`
- `Failed to set stereochemistry as unable to find an available bond`

# Substructure SMARTS strings
This is a brief list of some structures which might be useful in analysing the distribution of errors across molecules.
Taken from [daylight](https://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html#N)
- Alkyl carbon: `[CX4]`
- Arene carbon: `c`
- Arene nitrogen: `n`
- Ortho-substituted ring: `*-!:aa-!:*`
- Meta-substituted ring: `*-!:aaa-!:*`
- Para-substituted ring: `*-!:aaaa-!:*`
- Generic hydroxyl: `[OX2H]`
- Phenol: `[OX2H][cX3]:[c]`
- Ketone: `[#6][CX3](=O)[#6]`
- Aldehyde: `[CX3H1](=O)[#6]`
- Carboxylic acid: `[CX3](=O)[OX2H1]`
- Imine: `[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]`
- Enamine: `[NX3][CX3]=[CX3]`
- Nitro: `[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]`
- Nitrile: `[NX1]#[CX2]`
- Peroxide: `[OX2,OX1-][OX2,OX1-]`
- sp2 Aromatic carbon: `[$([cX3](:*):*),$([cX2+](:*):*)]`
- sp2 carbon: `[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)]`
- sp2 Nitrogen: `[$([nX3](:*):*),$([nX2](:*):*),$([#7X2]=*),$([NX3](=*)=*),$([#7X3+](-*)=*),$([#7X3+H]=*)]`
- sp3 Nitrogen: `[$([NX4+]),$([NX3]);!$(*=*)&!$(*:*)]`
- Amide: `[NX3][CX3](=[OX1])[#6]`


# Legacy code information
## Script Information
- `clean_database.py` takes roughly 4 minutes
- `create_features.py` takes roughly 10 minutes.
- `NN.py` takes on order of 5-10 hours.
## Openbabel
`openbabel` used in the project seems to be version `2.x.x` however this can't be verified due to no requirements file being included. Instead of using this it seems better to migrate to version `3.1.1` and moving forward take advantage of the `pybabel` API rather than the auto-generated C++ bindings. Some corrections which appear to be related to the version change are being made to the original code.

## Pipeline
1. Parse a train set of molecules into an sdf archive (write out to artifact)
2. Parse a test set of molecules into an sdf archive (same applies)
3. Vectorize the train data and generate metadata (artifact again)
4. Vectorize the test data based on the rules established by train data
5. (optionally) Search for optimal model paramteres
6. Train the model on the training data, holding some out for validation (model is an artifact)
   - Save the final per-example errors by molecule id
7. Test the model on the testing dataset, generating a set of errors for the mols in the test data
8. Analyse the error distribution across the train and test sets


## Cloud instance setup
1. Collect your WandB API key
2. Collect a github personal access token
3. Provision a google cloud instance either online in the console or via `gcloud compute instances create wandb-compute ` (must setup `gcloud` first)
4. Log in via SSH with `gcloud compute ssh wandb-compute`
5. Install conda with
```
curl -o script.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod u+x script.sh
./script.sh
```
And follow the prompts
6. Run `source ~/.bashrc` and conda should be working
7. Install git with `sudo apt install git` and clone the reop, using the command `git clone https://<username>:<access-token>@github.com/LukeRaw/molecular-magic.git`
8. Change into the directory with `cd molecular-magic`
9. Install the conda environment with `conda env create -f environment.yml`
10. Install the magic module with `pip install -e .`
11. Reinstall `protobuf` by running `pip uninstall protobuf` and then `conda install protobuf`
12. Log into weights and biases with `wandb login` and paste your API key when asked
13. The installation is now complete

<!-- ```
curl -o script.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod u+x script.sh
./script.sh
source ~/.bashrc
sudo apt install -y git
git clone https://<username>:<access-token>@github.com/LukeRaw/molecular-magic.git
conda env create -f environment.yml
pip install -e .
pip uninstall protobuf
conda install -y protobuf
``` -->

<!-- - Install Git LFS (https://git-lfs.github.com/) and run the following commands in the local git folder:
  - `git lfs install`
  - `git lfs fetch`
  - `git lfs pull` --> -->

<!-- ## Visualisation of Results
- Tensorboard can be installed to the conda environment using the following command:
  - `conda install -c conda-forge tensorboard`
- Tensorboard logs can be visualised using the following command:
  - `tensorboard --logdir=./`
- NN logs are store in `./static_data/NN_rewrite`.
- TO-DO: Define weights and biases setup and usage