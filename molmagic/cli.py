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
import bz2

import oyaml as yaml
from tqdm import tqdm
from molmagic import parser
from molmagic.rules import filter_mols
from molmagic import vectorizer
from molmagic.aggregator import autobin_mols, bin_mols
from molmagic import config
import numpy as np
from pathlib import Path
import sys


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

    basepath = args.input
    outpath = args.output

    # Walk the basepath directory and discover all the
    # g09 formatted output files
    # TODO: #70 use shell globbing and take a list of paths as args.input
    matched_paths = list(basepath.glob("./**/*f.out"))

    # Read those files and extract geometries and scf energies
    mol = map(parser.read_dft_frequency, matched_paths)

    # Filter this list to remove any bad objects
    mol_subset = filter(filter_mols, mol)

    # If we are not given a file, write this to stdout **uncompressed**
    if not outpath:
        for mol in mol_subset:
            sys.stdout.write(mol.write(format=config.extraction["output-format"]))
        return 0

    # Check the ouptut directory exists, and create if it does not
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if not outpath.name.endswith(".sdf.bz2"):
        outpath = outpath.with_suffix(".sdf.bz2")

    # Create a compression object
    compressor = bz2.BZ2Compressor()

    # Write appropriate objects into outpath under the same filename
    with outpath.open("wb") as buffer:
        # Iterate the molecules
        for mol in tqdm(mol_subset, total=len(matched_paths)):
            # Pybel returns a string if no output file is provided
            raw_output: str = mol.write(format=config.extraction["output-format"])
            # Encode the string to utf8 bytes
            bytes_output = raw_output.encode("utf-8")
            # Compress those bytes
            compressed_output = compressor.compress(bytes_output)
            # Stream them into the output file
            buffer.write(compressed_output)

        # Make sure nothing gets left behind in the compressor
        buffer.write(compressor.flush())


def vectorize(args: Namespace) -> None:
    """Computes a vector representation of a file of molecules, based on a previous
    configuration setup by <aggregate>
    """
    # Get our molecule set
    mols = parser.read_sdf_archive(args.input)

    # Extract the molecular properties
    # Note here that the substructure search will return different results if the
    # original and current `config.yml` files are not identical for that field
    mols = list(
        tqdm(
            map(vectorizer.calculate_mol_data, mols),
            leave=False,
            desc="Extracting molecular properties",
        )
    )

    if args.metadata:
        # Load the original configuration
        with args.metadata.open("r") as f:
            constructor = yaml.load(f, Loader=yaml.CLoader)
        data = constructor["data"]
        loaded_metadata = constructor["metadata"]

        # Bin the molecules according to the data and metadata
        feature_vector, target_vector = bin_mols(mols, data, loaded_metadata)
    else:
        # Compute and bin the molecules which have been extracted
        feature_vector, target_vector, calculated_metadata = autobin_mols(
            mols, args.plot_histograms
        )

    # Exit if we are not saving the output
    print(
        f"Vectorized {feature_vector.shape[0]} instances into {feature_vector.shape[1]} features."
    )
    if not args.output:
        return 0

    # Check the output path exists
    args.output.mkdir(exist_ok=True)

    # Save the files
    np.save(args.output / "features", feature_vector)
    np.save(args.output / "labels", target_vector)

    # If we are not loading metadata it means we have created some
    if not args.metadata:
        with (args.output / "metadata.yaml").open("w") as metadata_file:
            yaml.dump(calculated_metadata, metadata_file)

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
    vectorizer.set_defaults(func=vectorize)

    args = base_parser.parse_args(argv[1:])
    args.func(args)


if __name__ == "__main__":
    main(sys.argv)
