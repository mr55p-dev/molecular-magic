import openbabel.openbabel as ob
import pytest
import openbabel.pybel as pb
from magic.vectorizer import _search_substructure


def generate_mol(smiles_str) -> ob.OBMol:
    mol = pb.readstring("SMILES", smiles_str)
    mol.addh()
    mol.localopt()
    return mol.OBMol


@pytest.mark.parametrize("molsmiles,expected", [("C-C=C-C(O)=O", 1), ("C=C", 0), ("O=C(O)-C(O)-C(O)=O", 2)])
def test_carboxylic_pattern_search(molsmiles, expected):
    mol = generate_mol(molsmiles)
    count = _search_substructure(mol, "[CX3](=O)[OX2H1]")
    assert count == expected
