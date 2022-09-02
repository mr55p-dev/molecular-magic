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
import sys
import tarfile
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tarfile import is_tarfile
from typing import Iterable

import numpy as np
import oyaml as yaml
from openbabel import pybel as pb
from tqdm import tqdm

from molmagic import ml, parser, vectorizer
from molmagic.aggregator import autobin_mols, bin_mols
from molmagic.config import extraction as cfg_ext
from molmagic.config import qm9_exclude
from molmagic.rules import FilteredMols, global_filters, local_filters


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

    input_path: Path = args.input
    output_path: Path = args.output

    # Check the path exists
    if not input_path.exists():
        print(f"{input_path} does not exist.")

    # Detect if this is a tar archive to extract
    if input_path.is_file():
        molecules, n_instances = _parse_tar_dir(input_path)
    # Detect if this is a directory
    elif input_path.is_dir():
        molecules, n_instances = _parse_g09_dir(input_path)
    else:
        raise NotImplementedError("Cannot handle parsing this kind of structure.")

    # Check the ouptut directory exists, and create if it does not
    output_path = Path(output_path or "/tmp/archive.sdf.bz2")

    # Filter
    molecules = list(
        tqdm(
            filter(global_filters, molecules),
            leave=False,
            desc="Applying global filters",
        )
    )
    print(
        f"Filtered {FilteredMols.disjoint_structure} instances due to non-viable structure."
    )

    # Write the archive out
    n_molecules = parser.write_compressed_sdf(molecules, output_path, n_instances)
    print(f"Written {n_molecules} instances out to {output_path}")

    # Write this archive to a wandb archive (if asked to)
    if args.artifact:
        ml.log_parser_artifact(args.artifact, output_path, n_molecules)
        output_path.unlink()


def molecule_filter(args: Namespace) -> None:
    """Filter an sdf archive"""
    # Load the input file or artifact
    if args.input_file:
        infile = args.input_file
    else:
        ml.run_controller.use_run("filter")
        infile = ml.get_dataset_artifact(args.input_artifact)

    # Read the archive into memory and filter using the local filters
    molecules = parser.read_sdf_archive(infile)
    molecules = list(
        tqdm(
            filter(local_filters, molecules),
            leave=False,
            desc="Filtering molecules",
        )
    )
    print(f"Filtered {FilteredMols.get_total()} instances:")
    print(FilteredMols.get_breakdown())
    n_molecules = len(molecules)

    # Output a file to either destination or tmpdir
    if args.output_file:
        output_path = args.output_path
    else:
        output_path = Path("/tmp/archive.sdf.bz2")
    parser.write_compressed_sdf(molecules, output_path, n_molecules)

    # Save artifact if asked
    if args.output_artifact:
        ml.log_parser_artifact(args.output_artifact, output_path, n_molecules)
        # Delete tempoary archive if asked
        if not args.output_file:
            output_path.unlink()


def _parse_g09_dir(input_path: Path) -> tuple[Iterable[pb.Molecule], int]:
    # Walk the basepath directory and discover all the
    # g09 formatted output files
    matched_paths = list(input_path.glob("./**/*f.out"))
    molecules = parser.parse_files(matched_paths)
    n_instances = len(matched_paths)
    return molecules, n_instances


def _parse_tar_dir(input_path: Path):
    # Check the file is readable
    if not is_tarfile(input_path):
        raise tarfile.ReadError("Tarfile is not readable")
    # Autodetect the internal format from the filename
    fmt = input_path.name.split(".")[-2]
    # Get the number of entries in the archive
    with tarfile.open(input_path) as archive:
        n_instances = sum(1 for member in archive if member.isreg())
    return parser.parse_tar_archive(input_path, fmt, exclude=qm9_exclude), n_instances


def vectorize(args: Namespace) -> None:
    """Computes a vector representation of a file of molecules, based on a previous
    configuration setup by <aggregate>
    """
    # If no archive is specified load an artifact
    if args.load:
        ml.run_controller.use_run("vectorizer")
        input_path = ml.get_dataset_artifact(args.load)
    else:
        input_path = args.input

    # Get our molecule set
    molecules = parser.read_sdf_archive(input_path)

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

    metadata_file = None
    if args.metadata:
        metadata_file = args.metadata
    elif args.remote_metadata:
        # Define the run context
        run_dir = ml.get_vector_artifact(args.remote_metadata)
        metadata_file = run_dir / "metadata.yml"

    if metadata_file:
        # Load the original configuration
        with metadata_file.open("r") as f:
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
    if not (args.metadata or args.remote_metadata):
        with (metadata_output).open("w") as metadata_file:
            yaml.dump(calculated_metadata, metadata_file)

    # Create an artifact if we are asked to do so
    if args.artifact:
        ml.log_vector_artifact(
            args,
            feature_vector,
            features_output,
            labels_output,
            identities_output,
            metadata_output,
        )

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
    parser.add_argument(
        "-a",
        "--artifact",
        type=str,
        help="The name of this artifact in weights and biases (default not saved)",
    )
    parser.set_defaults(func=parse)

    # Create a filter option
    filterer = subparsers.add_parser(
        name="filter",
        description="Filters out instances based on the rules in config.yml",
    )
    filter_input_group = filterer.add_mutually_exclusive_group()
    filter_input_group.add_argument(
        "-i", "--input-file", type=Path, help="The input sdf file location"
    )
    filter_input_group.add_argument(
        "-l", "--input-artifact", type=str, help="The input artifact name"
    )
    filterer.add_argument(
        "-o", "--output-file", type=Path, help="The output file destiation."
    )
    filterer.add_argument(
        "-a", "--output-artifact", type=str, help="The output artifact name."
    )
    filterer.set_defaults(func=molecule_filter)

    # Create a vectorizer option
    vectorizer = subparsers.add_parser(
        name="vectorizer",
        description="""Compute the feature vector
        from a given archive file.""",
    )
    vectorizer_input = vectorizer.add_mutually_exclusive_group(required=True)
    vectorizer_input.add_argument(
        "-i",
        "--input",
        type=Path,
        help="""The bz2 archive of SDF structures annotated by the
        parser utility.""",
    )
    vectorizer_input.add_argument(
        "-l", "--load", type=str, help="""Name of the artifact to load from WandB"""
    )
    vectorizer_metadata = vectorizer.add_mutually_exclusive_group()
    vectorizer_metadata.add_argument(
        "-m",
        "--metadata",
        required=False,
        type=Path,
        help="""The metadata which was generated by aggregator when creating
        feature bins in a previous run. If this is not supplied, new bins
        will be computed based on the input file.""",
    )
    vectorizer_metadata.add_argument(
        "-r",
        "--remote-metadata",
        required=False,
        type=str,
        help="""The name and tag of a wandb run containing a metadata file to
        use as a vectorizing basis.""",
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
