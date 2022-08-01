"""
Utilities to extract properties from molecules in a structured and repeatable way
"""

from dataclasses import dataclass
from typing import Any, DefaultDict, Generator, Iterator, TypeVar
from openbabel import pybel as pb
from openbabel import openbabel as ob
from collections import Counter, defaultdict, namedtuple
from magic.config import extraction as cfg


# Types
HistogramData = TypeVar("HistogramData", bound=DefaultDict[str, list[float]])
DonorPair = namedtuple("DonorPair", ["proton", "bonded_atom"])


# Data types
class HBondInteraction:
    """Store the interacting bodies in a hydrogen bonding situation"""

    __slots__ = ["bonded_atom", "proton", "acceptor", "distance"]

    def __init__(self, bonded_atom: ob.OBAtom, proton: ob.OBAtom, acceptor: ob.OBAtom):
        """Store the"""
        self.bonded_atom = bonded_atom
        self.proton = proton
        self.acceptor = acceptor
        self.distance = self.compute_distance(proton, acceptor)

    @staticmethod
    def compute_distance(left: ob.OBAtom, right: ob.OBAtom) -> float:
        """Compute and cache the distance between the proton and acceptor"""
        return left.GetDistance(right)


# If this is slow we can bump the version to 3.10 and unlock slots on the dataclass
# @dataclass(eq=False, slots=True)
@dataclass(eq=False)
class MoleculeData:
    # Intrinsic properties
    attributes: dict

    # Properties which are just simple counts
    atoms: Counter
    amines: Counter

    # Properties which require a shitton of compute
    bonds: HistogramData
    angles: HistogramData
    dihedrals: HistogramData
    hbonds: HistogramData


def calculate_mol_data(molecule: pb.Molecule) -> MoleculeData:
    """Calculate a MoleculeData instance for a single molecule, using hbond criteria given in
    hbond_distance.

    molecule: pb.Molecule
        Pybel molecule object
    hbond_distance: tuple[low: float, high: float]
        Tuple containing the lower- and upper-bounds of allowed distances in hydrogen-bonding
        interactions.
    """
    # Get the counts of different atoms straight off
    atoms = Counter([i.atomicnum for i in molecule.atoms])
    amines = Counter(_get_amine_counts(molecule.OBMol))

    # Extract the more complicated features of a molecule
    bonds = _get_bonds_data(molecule.OBMol)
    angles = _get_angles_data(molecule.OBMol)
    dihedrals = _get_dihedrals_data(molecule.OBMol)
    hbonds = _get_hbond_data(molecule.OBMol)

    # Copy the SDF attributes into MoleculeData
    return MoleculeData(molecule.data, atoms, amines, bonds, angles, dihedrals, hbonds)


# Util functions
def _should_reverse(arr: list[Any]) -> bool:
    """Check if we should reverse a representation to ensure
    order is consistent"""
    # Find the midpoint of our list
    middle = len(arr) // 2

    # Iterate from the edges to the center
    for left, right in zip(arr[:middle], reversed(arr[middle:])):
        # If there are a pair such that right > left then we will instruct
        # reversal to occur
        if left < right:
            return True
    return False


def _sort_atoms(atoms: tuple[ob.OBAtom]) -> tuple[ob.OBAtom]:
    """Ensure atoms is always in the order heaviest to smallest."""
    # Convert the atoms into atomic numbers
    # Remove unnecessary calls to GetAtomicNum for torsion checks
    atom_nums = [i.GetAtomicNum() for i in atoms]
    return atoms[::-1] if _should_reverse(atom_nums) else atoms


def _create_dict_key(atoms: list[ob.OBAtom]) -> tuple[int]:
    """Convert atom sequence into a list of atomic numbers"""
    return tuple(i.GetAtomicNum() for i in atoms)


def _collect_neighbours(proton: ob.OBAtom) -> DonorPair:
    """Protons should have only one neighbour"""
    for neighbour in ob.OBAtomAtomIter(proton):
        return DonorPair(proton, neighbour)


def _get_combinations(
    donor_set: list[DonorPair], acceptor_set: list[ob.OBAtom]
) -> Generator[HBondInteraction, None, None]:
    """Yields an iterator over every possible donor-acceptor interaction
    in the system."""
    for donor in donor_set:
        for acceptor in acceptor_set:
            yield HBondInteraction(
                proton=donor.proton,
                bonded_atom=donor.bonded_atom,
                acceptor=acceptor,
            )


def _proton_is_enabled(proton: pb.Atom, enablers: tuple[int]) -> bool:
    """Check if the proton has a neighbour in the enablers list
    Enablers should be a list of atomic numbers"""
    for neighbour in ob.OBAtomAtomIter(proton.OBAtom):
        if neighbour.GetAtomicNum() in enablers:
            return True

    return False


def _get_amine_counts(molecule: ob.OBMol) -> Iterator[int]:
    """Return an iterable of amine degrees present in the molecule

    Currently the definition of an amine is strict - it requires a nitrogen
    with 3 single covalent bonds. There is no detection of further functionalisation
    and so amides will be identified as pure amines.
    """
    # Get all the nitrogen atoms (this will include imines and nitriles)
    nitrogen_centers = [
        atom for atom in ob.OBMolAtomIter(molecule) if atom.GetAtomicNum() == 7
    ]

    # More traditional amine defintion (excludes imines, nitriles)
    # Without this, imines get classified as primary amines
    nitrogen_centers = filter(
        lambda atom: atom.CountBondsOfOrder(1) == 3, nitrogen_centers
    )

    # Classify each as primary, secondary or tertiary
    return map(
        lambda atom: len(
            [None for i in ob.OBAtomAtomIter(atom) if i.GetAtomicNum() != 1]
        ),
        nitrogen_centers,
    )


def _get_bonds_data(molecule: ob.OBMol) -> HistogramData:
    # Get all the bond pairs around
    bonds = defaultdict(list)
    for ob_bond in ob.OBMolBondIter(molecule):
        # Get the participating atoms
        atoms = _sort_atoms(
            (
                ob_bond.GetBeginAtom(),
                ob_bond.GetEndAtom(),
            )
        )

        # Save the atomic numbers list
        bond_length = ob_bond.GetLength()
        bonds[_create_dict_key(atoms)].append(bond_length)

    return bonds


def _get_angles_data(molecule: ob.OBMol) -> HistogramData:
    # Get all the angles
    angles = defaultdict(list)
    for ob_angle in ob.OBMolAngleIter(molecule):
        # Need to include center-carbon valence here too
        # Get the participating atoms
        # ob_angle contains an array of atom indices (shifted by 1
        # for some reason)
        atoms = _sort_atoms(
            (
                molecule.GetAtom(ob_angle[0] + 1),
                molecule.GetAtom(ob_angle[1] + 1),
                molecule.GetAtom(ob_angle[2] + 1),
            )
        )

        # Calculate the angle
        angle_degree = molecule.GetAngle(*atoms)

        key = _create_dict_key(atoms)

        # Check if the center atom is a carbon
        if (c_atom := atoms[1]).GetAtomicNum() == 6:
            # Tack the total valence onto the end as a special requirement
            # Count the number of bonds with orders 1, 2 and 3 and sum them
            n_bonds = sum(c_atom.CountBondsOfOrder(i) for i in range(1, 4))
            key = (*key, n_bonds)
        else:
            # If we are not dealing with carbon set valence = 0
            key = (*key, 0)

        # Save the information
        angles[key].append(angle_degree)

    return angles


def _get_dihedrals_data(molecule: ob.OBMol) -> HistogramData:
    # Get all the torsional angles
    dihedrals = defaultdict(list)
    for ob_torsion in ob.OBMolTorsionIter(molecule):
        # Get the participating atoms
        atoms = _sort_atoms(
            (
                molecule.GetAtom(ob_torsion[0] + 1),
                molecule.GetAtom(ob_torsion[1] + 1),
                molecule.GetAtom(ob_torsion[2] + 1),
                molecule.GetAtom(ob_torsion[3] + 1),
            )
        )

        # Calculate the angle
        torsion_degree = molecule.GetTorsion(*atoms)

        # Save the information
        dihedrals[_create_dict_key(atoms)].append(torsion_degree)

    return dihedrals


def _get_hbond_data(molecule: ob.OBMol) -> HistogramData:
    """Calculate all the hydrogen bond interactions between atom pairs.
    Returns a key of the form (bonded_enabler, accepting_enabler).

    eg. a system N-H...N yields a key (7, 7)
                 N-H...O yields a key (7, 8)
    """
    hbonds = defaultdict(list)

    # Find every proton H_a
    protons = [atom for atom in ob.OBMolAtomIter(molecule) if atom.GetAtomicNum() == 1]

    # For each proton find its bonded atoms
    # Keep only donors which neighbour atoms in the enabler set
    donor_set = map(_collect_neighbours, protons)
    donor_set = filter(lambda x: x.bonded_atom.GetAtomicNum() in cfg["hbond-atoms"], donor_set)

    # Find all the acceptor atoms
    acceptor_set = list(
        filter(lambda x: x.GetAtomicNum() in cfg["hbond-atoms"], ob.OBMolAtomIter(molecule))
    )

    # Find the set of all donors and acceptors, keep only those within the right distance
    d_min: float = cfg["hbond-min-distance"]
    d_max: float = cfg["hbond-max-distance"]
    interaction_set = _get_combinations(donor_set, acceptor_set)
    interaction_set = filter(lambda x: d_min < x.distance < d_max, interaction_set)

    # Save the good stuff
    for interaction in interaction_set:
        key = _create_dict_key((interaction.bonded_atom, interaction.acceptor))
        hbonds[key].append(interaction.distance)

    return hbonds
