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

from tqdm import tqdm
from molmagic import parser
from molmagic.rules import filter_mols
from molmagic import vectorizer
from molmagic.aggregator import compute_and_bin_mols
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
    matched_paths = list(basepath.glob("./**/*f.out"))

    # Read those files and extract geometries and scf energies
    mol = map(parser.read_dft_frequency, matched_paths)

    # Filter this list to remove any bad objects
    mol_subset = filter(filter_mols, mol)

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


def aggregate(args: Namespace) -> None:
    """Reads an SDF archive, computes feature vectors based on histograms
    and writes out features and labels to numpy arrays."""
    # Get our molecule set
    mols = parser.read_sdf_archive(args.input)

    # Extract the molecular properties
    mols = list(
        tqdm(
            map(vectorizer.calculate_mol_data, mols),
            leave=False,
            desc="Extracting molecular properties",
        )
    )

    # Compute and bin the molecules which have been extracted
    target_vector, feature_vector = compute_and_bin_mols(mols, args.plot_histograms)

    # Check the output path exists
    args.output.mkdir(exist_ok=True)

    # Save the files
    np.save(args.output / "features", feature_vector)
    np.save(args.output / "labels", target_vector)

    print(
        f"""Generated a feature matrix with {feature_vector.shape[0]} instances
        and {feature_vector.shape[1]} features."""
    )


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
        "-i",
        "--input",
        required=True,
        type=Path,
        help="""Pass the directory which contains the input files. Note this
        can be a folder of folders; any `*f.out` files processed by `g09` in
        the subtree will be discovered.""",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
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
        "-i",
        "--input",
        required=True,
        type=Path,
        help="""The bz2 archive of SDF structures annotated by the
        parser utility.""",
    )
    vectorizer.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="""The directory to output numpy arrays to.
        If it does not exist, it will be created.""",
    )
    vectorizer.add_argument(
        "--plot-histograms",
        action="store_true",
        help="""Save histograms for this run""",
    )
    vectorizer.set_defaults(func=aggregate)

    args = base_parser.parse_args(argv[1:])
    args.func(args)


if __name__ == "__main__":
    main(sys.argv)
