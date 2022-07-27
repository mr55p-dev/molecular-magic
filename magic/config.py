import yaml


with open("config.yml", "r") as f:
    configuration = yaml.safe_load(f)


extraction = configuration["extraction"]
aggregation = configuration["aggregation"]
