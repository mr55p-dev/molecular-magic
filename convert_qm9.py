from molmagic.parser import read_qm9_dir, write_compressed_sdf
from molmagic.rules import FilteredMols, filter_mols

exclude_files = [
    21725,
    87037,
    59827,
    117523,
    128113,
    129053,
    129152,
    129158,
    130535,
    6620,
    59818,
    21725,
    59827,
    128113,
    129053,
    129152,
    130535,
    6620,
    59818,
]
mols = read_qm9_dir("data/qm9/qm9.xyz.tar", exclude=exclude_files)
n_mols = write_compressed_sdf(filter(filter_mols, mols), "./data/qm9/qm9")
print(f"Filtered {FilteredMols.get_total()} instances. Written {n_mols} instances")
print([(i, getattr(FilteredMols, i)) for i in vars(FilteredMols) if not (callable(i) or i.startswith('_'))])
