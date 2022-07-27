from pathlib import Path
from tqdm import tqdm
import cclib


# This should be the path to the raw data file
base = Path("dft_test_files")


def test_freq_geom_agreement():
    freq = base.glob("./**/*f.out")

    def getname(x: Path) -> str:
        return x.stem[:-1]

    err = 0
    for f in tqdm(freq):
        g = f.parent / (getname(f) + "a.out")
        assert g.exists()

        with g.open("r") as gfi, f.open("r") as ffi:
            eg = cclib.io.ccread(gfi)
            ef = cclib.io.ccread(ffi)

        assert getname(g) == getname(f)
        err += abs(eg.scfenergies[-1] - ef.scfenergies[-1])

    assert err < 0.1
