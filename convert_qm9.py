from molmagic.parser import read_qm9_dir, write_compressed_sdf

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
mols = read_qm9_dir("data/qm9/dsgdb9nsd.xyz.tar", exclude=exclude_files)
write_compressed_sdf(mols, "./data/qm9/qm9_conformer")
