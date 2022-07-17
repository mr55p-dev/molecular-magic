"""
Utils for parsing g09 formatted output files into an annotated
SDF file.

Requires cclib to be installed
"""

from pathlib import Path
import openbabel.pybel as pb
import cclib
from tqdm import tqdm


def read_geom(path: Path) -> pb.Molecule:
    """Read a gaussian output file into an OBMol object with scf energy annotated

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

    assert hasattr(ccdata, "scfenergies")
    scf_energies = ccdata.scfenergies
    assert scf_energies is not None

    # Always use the latest scf energy computed in the file (assuming these are in order)
    scf_ev = ccdata.scfenergies[-1]

    # TODO: Check openbabel has worked its magic
    assert mol

    # Convert the energy reported from eV to Hartrees (as per original spec)
    scf_energy = scf_ev * 0.036749322176

    # Set the converged energy as an attribute on the OBMol object
    mol.data.update({"scf_energy": scf_energy})

    return mol


def filter_mols(molecule: pb.Molecule) -> bool:
    """Defines filtering rules to eliminate molecules from the dataset"""

    return True


def convert_tree(basepath: Path, outpath: Path, fmt="sdf") -> None:
    """Convert all the files from basepath into filtered output in outpath

    Always uses the more advanced frequncy calculation

    Might need to use geometry files to check for convergence, look into this further
    Energies should be taken from frequency files, assuming these are always identical though.
    This needs to be checked also.
    UPDATE the energies in the frequency and geometry files are consistent
        There is an error, with a cumulative value of: eV


    TODO: #23 Look into open babel kekulerize warning

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
    mol = map(read_geom, matched_paths)

    # Filter this list to remove any bad objects
    mol_subset = filter(filter_mols, mol)

    # Check the ouptut directory exists, and create if it does not
    outpath.mkdir(parents=True, exist_ok=True)

    # Write appropriate objects into outpath under the same filename
    # Ideally we write to just one output file, but this seems easier for now
    # TODO: #24 Write the converted output to a single sdf file
    # TQDM total is approximate total
    for idx, mol in tqdm(enumerate(mol_subset), total=len(matched_paths)):
        output_file = outpath / f"{idx}.out"
        mol.write(format=fmt, filename=str(output_file), overwrite=True)