import oyaml as yaml


with open("config.yml", "r") as f:
    configuration = yaml.safe_load(f)


extraction = configuration["extraction"]
aggregation = configuration["aggregation"]
plotting = configuration["plotting"]
qm9_exclude = [
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
