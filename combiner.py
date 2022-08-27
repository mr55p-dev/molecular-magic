from pathlib import Path
from molmagic.parser import read_sdf_archive, write_compressed_sdf
from molmagic.cli import _parse_g09_dir
from molmagic.rules import global_filters, local_filters


base = Path("./data/MolE8/mole8.sdf.bz2")
new = Path("./data/MolE8/extrapolated")

orig = list(read_sdf_archive(base))
new_mols, _ = _parse_g09_dir(new)
new_mols = list(new_mols)

mols = orig + new_mols
mols = filter(global_filters, mols)
# mols = filter(local_filters, mols)
write_compressed_sdf(mols, Path("./data/MolE8/combined.sdf.bz2"))

