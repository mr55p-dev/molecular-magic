import openbabel.pybel as pb
import openbabel.openbabel as ob


def test_angle_iter():
    mol = pb.readstring("SMILES", "C=C")
    mol.addh()
    mol.localopt()
    x = [i for i in ob.OBMolAngleIter(mol.OBMol)]
    assert len(x) == 6
