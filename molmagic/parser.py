"""
Utils for parsing g09 formatted output files into an annotated
SDF file.

Requires cclib and bz2 to be installed
"""
from os import PathLike
from pathlib import Path
from typing import Iterable, Union
import openbabel.pybel as pb
import cclib
import bz2
from molmagic.config import extraction as cfg
import tarfile
from molmagic.config import qm9_exclude
from tqdm import tqdm
from rdkit import Chem


def parse_files(paths: list[Path]) -> Iterable[pb.Molecule]:
    """Iterate over a sequence of dft file paths
    and return each one parsed in the order given"""

    # Read those files and extract geometries and scf energies
    indices = range(len(paths))
    mol = map(read_dft_frequency, paths, indices)

    return mol


def parse_tar_archive(
    tarball: PathLike, format: str, exclude: list[int] = None
) -> Iterable[pb.Molecule]:
    """Open a tar archive molecules and return an iterator over pybel
    molecules"""
    if format != "xyz":
        raise NotImplementedError(
            f"{format} not yet implemented for parsing a tar archive."
        )
    # Get files in tarball
    with tarfile.open(tarball) as tar:
        for member in tar.getmembers():
            # Extract the file
            file = tar.extractfile(member)
            contents = file.read()

            # Extract props
            props = contents.splitlines()[1]
            fields = props.split(b"\t")

            # Get the file index
            identity = int(fields[0].split(b" ")[1])  # ID specified by QM9
            if exclude and (identity in exclude):
                continue

            # Construct openbabel
            mol = pb.readstring(format="xyz", string=contents.decode("utf-8"))
            mol.OBMol.AddHydrogens()

            # Calculate the label properties
            scf_energy = float(fields[12]) * 627.503  # U @ 298K (kcal/mol)
            free_energy = float(fields[14]) * 627.503  # G @ 298K (kcal/mol)

            # Save this to the molecule
            mol.data.update(
                {"id": identity, "scf_energy": scf_energy, "free_energy": free_energy}
            )

            yield mol


def parse_g09_dir(input_path: Path) -> tuple[Iterable[pb.Molecule], int]:
    # Walk the basepath directory and discover all the
    # g09 formatted output files
    matched_paths = list(input_path.glob("./**/*f.out"))
    molecules = parse_files(matched_paths)
    n_instances = len(matched_paths)
    return molecules, n_instances


def parse_tar_dir(input_path: Path):
    # Check the file is readable
    if not tarfile.is_tarfile(input_path):
        raise tarfile.ReadError("Tarfile is not readable")
    # Autodetect the internal format from the filename
    fmt = input_path.name.split(".")[-2]
    # Get the number of entries in the archive
    with tarfile.open(input_path) as archive:
        n_instances = sum(1 for member in archive if member.isreg())
    return parse_tar_archive(input_path, fmt, exclude=qm9_exclude), n_instances


def parse_sdf_archive(input_path: Path):
    # Iterate to get the number of instances
    with bz2.BZ2File(input_path, mode="rb") as archive:
        n_instances = sum(1 for line in archive.readlines() if line == b"$$$$")

    return read_sdf_archive(input_path), n_instances


def check_convergence(path: Path) -> bool:
    """Open a file and check that its converged.

    This must be done on a submitted geometry file, as the frequency
    files do not modify the geometry in their steps"""
    with path.open("r") as f:
        stat = cclib.io.ccread(f).optdone

    return stat


def read_dft_frequency(path: Path, idx: int) -> pb.Molecule:
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
    free_energy = ccdata.freeenergy * 627.503

    # Set the converged energy as an attribute on the OBMol object
    mol.data.update(
        {
            "id": idx,
            "scf_energy": scf_energy,
            "net_charge": ccdata.charge,
            "free_energy": free_energy,
        }
    )

    return mol


def read_sdf_archive(archive_path: Path) -> Iterable[pb.Molecule]:
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


def convert_pb_to_rdkit(
    molecules: Union[pb.Molecule, list[pb.Molecule]]
) -> Chem.rdmolfiles.SDMolSupplier:
    """Convert a pybel molecule(s) into an rdkit supplier (list of molecules) using SDF strings as an intermediary"""
    # Write to SDF
    if isinstance(molecules, pb.Molecule):
        molecules = [molecules]
    sdf_data = "".join(map(lambda x: x.write("sdf"), molecules))

    # Read from SDF
    supplier = Chem.rdmolfiles.SDMolSupplier()
    supplier.SetData(sdf_data)

    for mol in enumerate(supplier):
        if not mol:
            print("Epic fail")

    return supplier


def write_compressed_sdf(
    mol_subset: list[pb.Molecule], outpath: PathLike, n_instances: int = None
) -> int:
    # Create a compression object
    compressor = bz2.BZ2Compressor()

    # Write appropriate objects into outpath under the same filename
    n_mols = 0
    with outpath.open("wb") as buffer:
        # Iterate the molecules
        for mol in tqdm(
            mol_subset, total=n_instances if n_instances else None, leave=False
        ):
            # Pybel returns a string if no output file is provided
            raw_output: str = mol.write(format=cfg["output-format"])
            # Encode the string to utf8 bytes
            bytes_output = raw_output.encode("utf-8")
            # Compress those bytes
            compressed_output = compressor.compress(bytes_output)
            # Stream them into the output file
            buffer.write(compressed_output)
            # Increment the counter
            n_mols += 1

        # Make sure nothing gets left behind in the compressor
        buffer.write(compressor.flush())
    return n_mols
