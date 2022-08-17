"""
Utils for parsing g09 formatted output files into an annotated
and compressed SDF file.

When invoked, converts all the files from `input` into filtered output in
`output`. Always uses the more advanced frequncy calculation (`*f.out` files).

input:
    Directory containing all the files with specified format
output:
    File to write the archive into.
fmt:
    File format of the output

Depends on `cclib` and `bz2`.
"""
from argparse import ArgumentParser, Namespace
from tarfile import is_tarfile
import tarfile

import oyaml as yaml
from tqdm import tqdm
from molmagic.config import extraction as cfg_ext, qm9_exclude, plotting as cfg_plot
from molmagic import parser
from molmagic import vectorizer
from molmagic.aggregator import autobin_mols, bin_mols
from molmagic.rules import FilteredMols, filter_mols
from molmagic import config
import numpy as np
from pathlib import Path
import sys
import wandb


def parse(args: Namespace) -> None:
    """Convert all the files from basepath into filtered output in outpath

    Always uses the more advanced frequncy calculation

    There is an error between converged frequency and geometry files,
    with a cumulative value of: ~0.004eV on the entire part1 dataset.
    This can be considered neglegable.

    basepath:
        Directory containing all the files with specified format
    outpath:
        Directory to write all the output files to
    """

    basepath: Path = args.input
    outpath: Path = args.output

    # Check the path exists
    if not basepath.exists():
        print(f"{basepath} does not exist.")

    # Detect if this is a tar archive to extract
    if basepath.is_file():
        # Check the file is readable
        if not is_tarfile(basepath):
            raise tarfile.ReadError("Tarfile is not readable")
        # Autodetect the internal format from the filename
        fmt = basepath.name.split(".")[-2]
        # Get the number of entries in the archive
        with tarfile.open(basepath) as archive:
            n_instances = sum(1 for member in archive if member.isreg())
        mols = parser.parse_tar_archive(basepath, fmt, exclude=qm9_exclude)

    # Detect if this is a directory
    elif basepath.is_dir():
        # Walk the basepath directory and discover all the
        # g09 formatted output files
        matched_paths = list(basepath.glob("./**/*f.out"))
        mols = parser.parse_files(matched_paths)
        n_instances = len(matched_paths)

    else:
        raise NotImplementedError("Cannot handle parsing this kind of structure.")

    # If we are not given a file, write this to stdout **uncompressed**
    if not outpath:
        for mol in mols:
            sys.stdout.write(mol.write(format=config.extraction["output-format"]))
        return 0

    # Check the ouptut directory exists, and create if it does not
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if not outpath.name.endswith(".sdf.bz2"):
        outpath = outpath.with_suffix(".sdf.bz2")

    # Write the archive out
    n_mols = parser.write_compressed_sdf(mols, outpath, n_instances)
    print(f"Written {n_mols} instances out to {outpath}")

    # Write this archive to a wandb archive (if asked to)
    ...


def vectorize(args: Namespace) -> None:
    """Computes a vector representation of a file of molecules, based on a previous
    configuration setup by <aggregate>
    """
    # If no archive is specified load an artefact
    ...

    # Get our molecule set
    molecules = parser.read_sdf_archive(args.input)

    # Filter
    if cfg_ext["use-filters"]:
        molecules = list(
            tqdm(
                filter(filter_mols, molecules), leave=False, desc="Filtering molecules"
            )
        )

        print(f"Filtered {FilteredMols.get_total()} instances:")
        print(FilteredMols.get_breakdown())

    # Extract the molecular properties
    # Note here that the substructure search will return different results if the
    # original and current `config.yml` files are not identical for that field
    molecules = list(
        tqdm(
            map(vectorizer.calculate_mol_data, molecules),
            leave=False,
            desc="Extracting molecular properties",
            total=len(molecules) if isinstance(molecules, list) else None,
        )
    )

    if args.metadata:
        # Load the original configuration
        with args.metadata.open("r") as f:
            constructor = yaml.load(f, Loader=yaml.CLoader)
        data = constructor["data"]
        loaded_metadata = constructor["metadata"]

        # Bin the molecules according to the data and metadata
        feature_vector, target_vector = bin_mols(molecules, data, loaded_metadata)
    else:
        # Compute and bin the molecules which have been extracted
        feature_vector, target_vector, calculated_metadata = autobin_mols(
            molecules, args.plot_histograms
        )

    print(
        f"Vectorized {feature_vector.shape[0]} instances into {feature_vector.shape[1]} features."
    )

    # Get the molecule id's
    id_vector = np.array([mol.data["id"] for mol in molecules]).astype(np.int32)

    # Check the output path exists
    if not args.output:
        args.output = Path("/tmp/")
    else:
        args.output.mkdir(exist_ok=True)

    # Define the paths
    features_output = args.output / "features.npy"
    labels_output = args.output / "labels.npy"
    identities_output = args.output / "identities.npy"
    metadata_output = args.output / "metadata.yml"

    # Save the files
    np.save(features_output, feature_vector)
    np.save(labels_output, target_vector)
    np.save(identities_output, id_vector)

    # If we are not loading metadata it means we have created some
    if not args.metadata:
        with (metadata_output).open("w") as metadata_file:
            yaml.dump(calculated_metadata, metadata_file)

    # Create an artifact if we are asked to do so
    if args.artifact:
        wandb.init(
            project="MolecularMagic",
            entity="molecular-magicians",
        )
        artifact = wandb.Artifact(
            name=args.artifact,
            type="dataset",
            description="Output of molmagic.cli.vectorize for the {args.artifact} dataset",
        )
        # Upload the files
        artifact.add_file(features_output, name="features.npy")
        artifact.add_file(labels_output, name="labels.npy")
        artifact.add_file(identities_output, name="identities.npy")
        artifact.add_file(metadata_output, name="metadata.yml")

        # Check if we need to log figures
        if args.plot_histograms:
            artifact.add_dir(cfg_plot["save-dir"])

        wandb.log_artifact(artifact)

    # Unlink the created files if we are not saving output
    if not args.output:
        features_output.unlink()
        labels_output.unlink()
        identities_output.unlink()
        if not args.metadata:
            metadata_output.unlink()

    return 0


def main(argv=sys.argv):
    base_parser = ArgumentParser()

    def show_help(*args, **kwargs):
        base_parser.print_help()

    base_parser.set_defaults(func=show_help)

    # Create a parser option
    subparsers = base_parser.add_subparsers(title="subcommands")
    parser = subparsers.add_parser(
        name="parser",
        help="""Utility for converting a directory of gaussian
        frequency files into a compressed SDF output archive (bz2 compression)
        """,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="""Pass the directory which contains the input files. Note this
        can be a folder of folders; any `*f.out` files processed by `g09` in
        the subtree will be discovered.""",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        type=Path,
        help="""The output SDF archive. Instances have a sdf_energy
        key which contains the extracted energy. File is saved as a
        bz2-compressed archive, which can be recovered using methods
        implemented in magic.parser""",
    )
    parser.set_defaults(func=parse)

    # Create a vectorizer option
    vectorizer = subparsers.add_parser(
        name="vectorizer",
        description="""Compute the feature vector
        from a given archive file.""",
    )
    vectorizer.add_argument(
        "input",
        type=Path,
        help="""The bz2 archive of SDF structures annotated by the
        parser utility.""",
    )
    vectorizer.add_argument(
        "-m",
        "--metadata",
        required=False,
        type=Path,
        help="""The metadata which was generated by aggregator when creating
        feature bins in a previous run. If this is not supplied, new bins
        will be computed based on the input file.""",
    )
    vectorizer.add_argument(
        "-o",
        "--output",
        required=False,
        type=Path,
        help="""The directory to output numpy arrays and metadata to.
        If it does not exist, it will be created. If not specified, no output
        will be given.""",
    )
    vectorizer.add_argument(
        "--plot-histograms",
        action="store_true",
        help="""Save histograms for this run""",
    )
    vectorizer.add_argument(
        "-a",
        "--artifact",
        required=False,
        type=str,
        help="""The name of this artifact in weights and biases (default not saved)""",
        default=None,
    )
    vectorizer.set_defaults(func=vectorize)

    args = base_parser.parse_args(argv[1:])
    args.func(args)


if __name__ == "__main__":
    main(sys.argv)
