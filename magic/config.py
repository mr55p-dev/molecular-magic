import yaml


with open("config.yml", "r") as f:
    configuration = yaml.load(f)


extraction = configuration["extraction"]
aggregation = configuration["aggregation"]