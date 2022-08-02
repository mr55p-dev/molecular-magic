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


def check_convergence(path: Path) -> bool:
    """Open a file and check that its converged.

    This must be done on a submitted geometry file, as the frequency
    files do not modify the geometry in their steps"""
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

    # Always use the latest scf energy computed in the file
    # (assuming these are in order)
    scf_ev = ccdata.scfenergies[-1]

    # TODO: #25 Check openbabel has worked its magic
    assert mol is not None

    # Unit conversions from http://wild.life.nctu.edu.tw/class/common/energy-unit-conv-table.html
    # Convert the energy reported from eV to Kcal mol^-1 (as per original spec)
    scf_energy = scf_ev * 23.0609

    # Convert hartree / particle to kcal / mol
    free_energy = ccdata.freeenergy * ...

    # Set the converged energy as an attribute on the OBMol object
    mol.data.update(
        {
            "scf_energy": scf_energy,
            "net_charge": ccdata.charge,
            "free_energy": free_energy,
        }
    )

    return mol


def read_sdf_archive(archive_path: Path) -> Generator[pb.Molecule, None, None]:
    """Open a bz2 archive containing SDF-formatted molecules and return an
    iterator over pybel molecules"""

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
