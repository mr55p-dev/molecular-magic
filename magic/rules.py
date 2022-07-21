"""Implementions of filtering rules used by the parser"""

from openbabel import pybel as pb


def filter_mols(molecule: pb.Molecule) -> bool:
    """Defines filtering rules to eliminate molecules from the dataset.

    If new molecules are added, it will need to check everything from the
    original paper.  This will require data from the geometry and frequency
    calculation steps.

    Can make a preprocessing step where the combined data is stored as keys
    in the sdf file and then a second script to read those sdfs and their
    properties"""
    return True
