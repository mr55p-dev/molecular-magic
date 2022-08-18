"""Implementions of filtering rules used by the parser"""

from logging import Filter
from openbabel import pybel as pb, openbabel as ob
from molmagic.config import extraction as cfg, aggregation as cfg_agg

anglelimit4 = 29.0
anglelimit3A = 20.0
anglelimit3B = 35.0
anglelimit2 = 20.0
carbanion_checklist = ["C4", "H1", "O2", "N3"]


class FilteredMols:
    other_atom = 0
    heavy_atom = 0
    zero_free_energy = 0
    long_bond = 0
    strained_angle = 0
    tetravalent_nitrogen = 0
    carbanion = 0
    disjoint_structure = 0

    @staticmethod
    def get_total():
        return sum(
            [
                FilteredMols.other_atom,
                FilteredMols.heavy_atom,
                FilteredMols.zero_free_energy,
                FilteredMols.long_bond,
                FilteredMols.strained_angle,
                FilteredMols.tetravalent_nitrogen,
                FilteredMols.carbanion,
                FilteredMols.disjoint_structure,
            ]
        )

    @staticmethod
    def get_breakdown():
        filter_categories = FilteredMols.get_filter_categories()
        return "\t" + "\n\t".join(
            [f"{name.replace('_', ' ')}: {count}" for name, count in filter_categories]
        )

    @staticmethod
    def get_filter_categories():
        filter_categories = [
            (i, getattr(FilteredMols, i))
            for i in vars(FilteredMols)
            if not (callable(getattr(FilteredMols, i)) or i.startswith("_"))
        ]

        return filter_categories

    @staticmethod
    def get_dict():
        return dict(FilteredMols.get_filter_categories())


def global_filters(molecule: pb.Molecule) -> bool:
    """Defines filtering rules required for the proper function
    of the vectorizer"""
    # Disallow disconnected structures (hacky but it works)
    smiles = molecule.write("smi").split("\t")[0]
    if "." in smiles:
        FilteredMols.disjoint_structure += 1
        return False
    return True


def local_filters(molecule: pb.Molecule) -> bool:
    """Defines filtering rules to eliminate molecules from the dataset.

    If new molecules are added, it will need to check everything from the
    original paper.  This will require data from the geometry and frequency
    calculation steps.

    Can make a preprocessing step where the combined data is stored as keys
    in the sdf file and then a second script to read those sdfs and their
    properties"""
    mol = molecule.OBMol

    # Remove atoms other than HCNO
    if any(i.atomicnum not in cfg_agg["atom-types"] for i in molecule.atoms):
        FilteredMols.other_atom += 1
        return False

    # Filter out molecules of the wrong size
    min_heavy_atoms = cfg["min-heavy-atoms"]
    max_heavy_atoms = cfg["max-heavy-atoms"]
    if not (
        min_heavy_atoms
        <= len([i for i in molecule.atoms if i.atomicnum != 1])
        <= max_heavy_atoms
    ):
        FilteredMols.heavy_atom += 1
        return False

    # Filter free energy == 0
    if not float(molecule.data["free_energy"]):
        FilteredMols.zero_free_energy += 1
        return False

    # Filter bonds which are not in the limit range
    bond_min_range = cfg["bond-min-distance"]
    bond_max_range = cfg["bond-max-distance"]
    if any(
        not (bond_min_range < i.GetLength() < bond_max_range)
        for i in ob.OBMolBondIter(mol)
    ):
        FilteredMols.long_bond += 1
        return False

    # Filter angles which are classed as strained (check implementation)
    atomanglesum = dict()
    for angle in ob.OBMolAngleIter(mol):
        central_val = mol.GetAtom(angle[0] + 1).GetTotalValence()
        central_type = mol.GetAtom(angle[0] + 1).GetType()
        central_idx = mol.GetAtom(angle[0] + 1).GetIdx()
        atomangle = mol.GetAngle(
            mol.GetAtom(angle[1] + 1),
            mol.GetAtom(angle[0] + 1),
            mol.GetAtom(angle[2] + 1),
        )

        testangle = 0
        if central_val == 4:
            testangle = atomangle - 109.5

        if central_val == 3 and central_type[0] == "C":

            if central_type[0] + str(central_idx) in atomanglesum.keys():
                atomanglesum[central_type[0] + str(central_idx)].append(atomangle)
            else:
                atomanglesum[central_type[0] + str(central_idx)] = [atomangle]

        if central_val == 2:
            testangle = 180 - atomangle

        if testangle > anglelimit4 and central_type[0] == "C" and central_val == 4:
            FilteredMols.strained_angle += 1
            return False

        if testangle > anglelimit2 and central_type[0] == "C" and central_val == 2:
            FilteredMols.strained_angle += 1
            return False

    # Remove tetravalent nitrogen atoms (check usage of GetTotalValence)
    if any(
        i.OBAtom.GetTotalValence() == 4
        for i in filter(lambda x: x.atomicnum == 7, molecule.atoms)
    ):
        FilteredMols.tetravalent_nitrogen += 1
        return False

    # Remove carbanions
    for atom in ob.OBMolAtomIter(mol):
        # Check the atom is carbon and has valence of 3
        if atom.GetAtomicNum() != 6 or atom.GetTotalValence() != 3:
            continue

        atom_types = []
        for neighbour_atom in ob.OBAtomAtomIter(atom):
            neighbour_atomtype = str(neighbour_atom.GetType())
            neighbour_valence = str(neighbour_atom.GetTotalValence())
            atom_types += [neighbour_atomtype[0] + neighbour_valence]

        if all(i in carbanion_checklist for i in atom_types):
            FilteredMols.carbanion += 1
            return False

    return True
