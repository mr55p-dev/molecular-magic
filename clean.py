from magic.parser import convert_tree
from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=Path, help="Directory containing input files")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Directory to write output files")
    # parser.add_argument("--in-format", type=str, choices=["g09"], help="Input file format")
    # parser.add_argument("--out-format", type=str, choices=["sdf"], help="Output file format")


    args = parser.parse_args()
    convert_tree(args.input, args.output)
