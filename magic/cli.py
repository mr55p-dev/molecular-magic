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
from argparse import ArgumentParser
from magic.parser import convert_tree
from pathlib import Path
import sys


def main(argv=sys.argv):
    base_parser = ArgumentParser()

    # Create a parser option
    subparsers = base_parser.add_subparsers(
        title="subcommands", description="utilities defined in the module"
    )
    parser = subparsers.add_parser(
        name="parser",
        description="""Utility for converting a directory of gaussian
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
    parser.set_defaults(func=convert_tree)

    args = base_parser.parse_args(argv[1:])
    args.func(args)


if __name__ == "__main__":
    main(sys.argv)
