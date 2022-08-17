# Molecular Magic

Working repository for the rewriting of the code in: https://github.com/sanha0213/MolE8

Work is performed as part of the University of Nottingham Machine Learning in Science MSc project of Luke Rawlence and Ellis Lunnon.
****
</br>

## How to run the code
Much of the configuration for feature generation is controlled via `config.yml`. The properties are:
- ### Extraction
  - #### Filter configuration
  - `use-filters`: Should molecules be filtered based on the MolE8 filtering rules (configured below). Note filtering is applied at the **vectorizing step** not the parsing step, so archives will still retain all instances and do not need to be recompiled every time a rule changes.
  - `output-format`: The format to save instances as internally in the output bz2 archive. This should not be changed from `sdf` as nothing else is implemented.
  `bond-min-distance`: The minimum distance between all atoms. If a molecule contains a bond shorter than this, it is filtered out under the rule `long bonds`.
  - `bond-max-distance`: The maximum distance between all atoms. See above
  - `min-heavy-atoms`: The minimum number of heavy atoms allowed in a molecule (exclusive). Heavy atoms count as any atom with an atomic number greater than 1.
  - `max-heavy-atoms`: The maximum number of heavy atoms allowed in a molecule (exclusive). Heavy atoms count as any atom with an atomic number greater than 1.
  - #### Extraction configuration
  - `hbond-atoms`: The atom types (given as atomic numbers) which can take part in hydrogen bonding. Default behaviour is Nitrogen (7) and Oxygen (8)
  - `hbond-min-distance`: The minimum distance (in angstrom) between a hydrogen donor and hbond acceptor for a bond to be considered a hydrogen bond
  - `hbond-max-distance`: The maximum distance (in angstrom) between a hydrogen donor and hbond acceptor for a bond to be considered a hydrogen bond
  - `substructures`: A list of SMARTS strings. This list gets counted for each molecule and added to the total feature vector. See [here](https://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html) for some common examples of functional groups which can be addressed in this way.
- ### Aggregation
  - `resolution`: The number of samples to use when sampling the KDE for computing minima.
  - `bandwidth`: The bandwidth to use when generating histograms. Can either be a positive scalar, 'scott' for the Scott method, 'silverman' for the Silverman method of automatic bandwidth selection or empty for the default (Scott method).
  - `label_name`: The property to extract as the target for each molecule (can be `scf_energy`, `free_energy` or `net_charge`)
  - `atom-types`: Each atom (atomic number) listed here will be counted and included as part of the representation. By default this is `[1, 6, 7, 8]` meaning `[Hydrogen, Carbon, Nitrogen, Oxygen]`. The elemnents `[0:4]` of the feature will therefore correspond to the frequency of those atoms in the order given. Also controlls the behaviour of filtering if that is enabled.
  - `amine-types`: The degrees of amine which are counted as part of the representation. By default, primary (`1`), secondary (`2`) and tertiary (`3`) amines are counted.
  - `feature-types`: Which features to use when generating the histograms. Allowed values are `bonds`, `angles`, `dihedrals` and `hbonds`. More may be added in the future.
- ### Plotting
  - `save-dir`: The directory to write figures to. This path will be created if it does not exist.
  - `show-dist`: Plot the histogram shape (`sns.histplot`).
  - `show-kde`: Draw a lineplot of the sampled kernel density estiamte.
  - `plot-style`: Styling parameter passed to `sns.set_style`.

### 0. Set up the environment
- Clone this repo and install the anaconda environment with `conda env create -f environment.yml`. This downlads the correct version of python and all its dependencies.
  - Activate the environment with `conda activate molmagic`
- Once this is done, install the `molmagic` module locally using pip. Run the command `pip install -e .` to install the script endpoints in your environment.
  - Running the command `magic` in your terminal should print out the help message for the tool
- Download the dataset zip files from the [University of Nottingham repository](https://unow.nottingham.ac.uk/handle/internal/9356?show=full).
  - Create a folder called `moldata` in the repo folder and extract the files there.
### 1. Create cleaned annotated sdf files
- Run `magic parser <path_to_input> -o <path_to_output_file>`,
  - Input can be:
    - Directory of `g09` formatted Gaussian frequency calcualtion files (the parser will glob for `*f.out`) files.
    - `.tar` archive of files. The format of the compressed files is extracted from the file name (which should be of the form `name.format.tar`). Currently, `xyz` files are supported (of the formt designated by QM9).
  - If a directory is specified the parser will recursively search for files.
  - Right now we use `pathlib.Path.glob`, so there is no order-consistency guarantee.
  - `-o` is an optional argument,
    - if provided, a **bz2 compressed** `sdf` formatted file is written to the destination file specified.
    - else, if the `-a/--artifact` flag is supplied then a tempoaray file will be created in order to log the dataset and then unlinked.
    <!-- - else, the **uncompressed** `sdf` formatted data will be written to `stdout`. -->
    - The suggesed file extension for the output of this command is `.sdf.bz2` for clarity.
  - Running the command `magic parser ./moldata -o ./cleaned_data.sdf.bz2` will read all `g09` frequency files from `moldata` and writing atom positions and computed properties to a **compressed** `sdf` file `./cleaned_data.sdf.bz2`.
  - Note that filtering occurs at this stage
  <!-- - Running the command `magic parser ./moldata > output.sdf` will write an **uncompressed** sdf file `output.sdf` as we are making use of the shell redirection tricks. -->
### 2. Generate npy feature vectors
- Run `magic vectorizer (-i <path_to_annotated_sdf_file> OR -l <name_of_wandb_artifact>) -o <output_dir>`
- The utility `magic vectorizer` can be used to convert a `.sdf.bz2` archive into saved numpy vectors (`.npx` files).
- To generate a representation based on the molecules in the archive, do not pass the `-m`/`--metadata` flag.
  - There will be four new files created in that directory specified, `features.npy`, `labels.npy`, `identity.npy` and `metadata.yml`.
  - `identity.npy` is a $1\times N$ vector mapping each row of the generated matrix to a molecule ID which can be accessed on `pybel.Molecule` instances as `pybel.Molecule.data['id']` (note only supported in archives generated by `parser >= 2.0.0`).
  - `metadata.yml` is used later on to enable converting molecules into this vector scheme using the same histograms.
- To generate a representation based on the results of a previous run, pass the `-m`/`--metadata` flag with the path to the `metadata.yml` file which was created for the run you wish to use.
- The flag `--plot-histograms` can be used when generating a new representation, and will output png files to the directory specified in `config.yml:plotting:save-dir`. These are to help visualise exactly what the algorithm is doing
- The `-a/--artifact <name>` flag will write the output as an artifact with name `<name>` to weights and biases.
- The labels in `labels.npy` will be extracted such that `labels[:, 0]` corresponds to the electronic energy and `labels[:, 1]` corresponds to the free energy. The funtion `molmagic.ml.get_label_type` allows for extraction of the desired property by name.
### 3. Machine Learning Code...

----
</br>

## Additional Information
<!-- ### `molmagic` usage
- Run `molmagic -h` for a list of subcommands.
- The database cleaning can be invoked with `molmagic parser -i <input directory> -o <output archive>`. -->
Note the order is guaranteed after generating the sdf file - `read_sdf_archive` is order-guaranteed.
### Error messages
- The following Openbabel error messages can be ignored: 
  - `Failed to kekulize aromatic bonds in OBMol::PerceiveBondOrders`
  - `Failed to set stereochemistry as unable to find an available bond`

<!-- ### Tests
- Tests can be run with pytest (`python3 -m pytest`). -->

### Weights and biases
- Weights and biases is used to record experiments. In order to perform and track an experimient, please sign into weights and biases locally (see [documentation](https://docs.wandb.ai/)).
The workspace which contains runs and information is hosted [here](https://wandb.ai/molecular-magicians/MolecularMagic).

</br></br></br>

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