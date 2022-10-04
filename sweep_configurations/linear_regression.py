project: MolecularMagic
name: elastic-architecture-sweep
program: "model_training/sweep_elastic_architecture.py"
method: grid
command:
  - ${env}
  - ${interpreter}
  - ${program}
metric:
  goal: minimize
  name: val_mae
parameters:
  seed:
    value: 50
  splitting_type:
    value: random
  label_name:
    value: free_energy
  elastic_alpha:
    distribution: categorical
    values:
      - 1
      - 0.5
      - 0.1
      - 0.05
      - 0.01
      - 0.005
      - 0.001
      - 0.0005
      - 0.0001
      - 0
  l1_ratio:
    distribution: categorical
    values:
      - 0
      - 0.1
      - 0.2
      - 0.3
      - 0.4
      - 0.5
      - 0.6
      - 0.7
      - 0.8
      - 0.8
      - 0.9
      - 1.0
  max_iter:
    distribution: categorical
    values:
      - 1000
      - 10000
  training_artifact:
    value: molecular-magicians/MolecularMagic/qm9-std-train-0.48-scott:latest