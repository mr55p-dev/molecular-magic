from pathlib import Path
from tqdm import tqdm
import cclib


# This should be the path to the raw data file
base = Path("/Users/ellis/Downloads/Part1")
geom = base.glob("./**/*a.out")
freq = base.glob("./**/*f.out")

getname = lambda x: x.stem[:-1]
geom = sorted(geom, key=getname)
freq = sorted(freq, key=getname)

pairs = list(zip(geom, freq))

err = 0
for g, f in tqdm(pairs):
    with g.open("r") as gfi, f.open("r") as ffi:
        eg = cclib.io.ccread(gfi)
        ef = cclib.io.ccread(ffi)

    assert getname(g) == getname(f)
    err += abs(eg.scfenergies[-1] - ef.scfenergies[-1])

print(err)

