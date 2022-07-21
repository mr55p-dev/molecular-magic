from magic.parser import convert_tree, read_sdf_archive
from pathlib import Path


test_dir = Path("dft_test_files/")
test_output = Path("dft_test_files/output.sdf.bz2")


def test_encode():
    """Test taking a set of dft files and encoding them as SDFs"""
    assert test_dir.exists()
    test_output.unlink(missing_ok=True)

    convert_tree(test_dir, test_output)

    assert test_output.exists()
    assert test_output.stat().st_size > 0

