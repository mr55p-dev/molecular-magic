import argparse
import openbabel.pybel as pb
from molmagic.parser import read_sdf_archive
from molmagic.cli import parse
from pathlib import Path
import pytest


test_dir = Path("dft_test_files/")
test_output = Path("dft_test_files/output.sdf.bz2")


@pytest.mark.dependency()
def test_encode():
    """Test taking a set of dft files and encoding them as SDFs"""
    assert test_dir.exists()
    test_output.unlink(missing_ok=True)

    args = argparse.Namespace(input=test_dir, output=test_output)
    parse(args)

    assert test_output.exists()
    assert test_output.stat().st_size > 0


@pytest.mark.dependency(depends=["test_encode"])
def test_decode():
    """Test decoding a compressed SDF file into an iterator of molecules"""
    # Use the load iterator function
    mol_iterator = read_sdf_archive(test_output)
    first_mol = next(mol_iterator)
    assert isinstance(first_mol, pb.Molecule)

    # Check the energy can be cast to float and is negative
    assert float(first_mol.data["scf_energy"]) < 0