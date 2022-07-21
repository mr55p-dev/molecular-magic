"""
Utils for parsing g09 formatted output files into an annotated
SDF file.

Requires cclib and bz2 to be installed
"""
from pathlib import Path
from typing import Generator
import openbabel.pybel as pb
import cclib
import bz2
from tqdm import tqdm
from magic.rules import filter_mols


def check_convergence(path: Path) -> bool:
    """Open a file and check that its converged.

    This must be done on a submitted geometry file, as the frequency files do not
    modify the geometry in their steps"""
    with path.open("r") as f:
        stat = cclib.io.ccread(f).optdone

    return stat


def read_dft_frequency(path: Path) -> pb.Molecule:
    """Read a gaussian output file into an OBMol object with scf energy annotated

    Should only be used on converged files - this is NOT CHECKED here

    path:
        Path to the gaussian16 (g09) encoded file object
    """
    with path.open("r", encoding="latin-1") as file:
        # Extract converged geometry with pybel
        mol = pb.readstring(format="g09", string=file.read())

        # Extract compute outputs with cclib
        ccdata = cclib.io.ccread(file)

    # Check molecule is read properly
    # Check cclib has succeeded
    if ccdata is None:
        raise ValueError(f"cclib could not parse data for {path}")

    if not hasattr(ccdata, "scfenergies"):
        raise ValueError("cclib could not extract energies")

    # Always use the latest scf energy computed in the file (assuming these are in order)
    scf_ev = ccdata.scfenergies[-1]

    # TODO: #25 Check openbabel has worked its magic
    assert mol is not None

    # Convert the energy reported from eV to Hartrees (as per original spec)
    scf_energy = scf_ev * 0.036749322176

    # Set the converged energy as an attribute on the OBMol object
    mol.data.update({"scf_energy": scf_energy})

    return mol


def read_sdf_archive(archive_path: Path) -> Generator[pb.Molecule, None, None]:
    """Open a bz2 archive containing SDF-formatted molecules and return an iterator over
    pybel molecules"""

    # Create an array to hold relevant lines
    linebuffer = []

    # Open the archive and start reading lines
    with bz2.BZ2File(archive_path, mode="rb") as archive:
        for line in archive.readlines():
            linebuffer.append(line)
            if b"$$$$" in line:
                # Decode bytes array into string
                sdf_string = b"".join(linebuffer)
                sdf_string = sdf_string.decode("utf-8")

                # Flush the buffer if we reach the termination line
                linebuffer.clear()

                # Construct an pb.Molecule
                yield pb.readstring(format="sdf", string=sdf_string)


def convert_tree(basepath: Path, outpath: Path, fmt="sdf") -> None:
    """Convert all the files from basepath into filtered output in outpath

    Always uses the more advanced frequncy calculation

    Might need to use geometry files to check for convergence, look into this further
    Energies should be taken from frequency files, assuming these are always identical though.
    This needs to be checked also.
    UPDATE the energies in the frequency and geometry files are consistent
        There is an error, with a cumulative value of: ~0.004eV on the entire part1 dataset
        This can be considered neglegable

    basepath:
        Directory containing all the files with specified format
    outpath:
        Directory to write all the output files to
    fmt:
        File format of the output
    """

    # Walk the basepath directory and discover all the
    # g09 formatted output files
    matched_paths = list(basepath.glob("./**/*f.out"))

    # Read those files and extract geometries and scf energies
    mol = map(read_dft_frequency, matched_paths)

    # Filter this list to remove any bad objects
    mol_subset = filter(filter_mols, mol)

    # Check the ouptut directory exists, and create if it does not
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Create a compression object
    compressor = bz2.BZ2Compressor()

    # Write appropriate objects into outpath under the same filename
    with outpath.open("wb") as buffer:
        # Iterate the molecules
        for mol in tqdm(mol_subset, total=len(matched_paths)):
            # Pybel returns a string in the case that no output file is provided
            raw_output: str = mol.write(format=fmt)
            # Encode the string to utf8 bytes
            bytes_output = raw_output.encode("utf-8")
            # Compress those bytes
            compressed_output = compressor.compress(bytes_output)
            # Stream them into the output file
            buffer.write(compressed_output)

        # Make sure nothing gets left behind in the compressor
        buffer.write(compressor.flush())
