"""Configuration for `clean_database.py`"""

from pathlib import Path


data_basepath = Path("./static_data/")
molecule_list_path = "GDB_allgroundnoimagnocarban_Dec2021_G298_LongBondLimit1.6.pkl"
output_basepath = data_basepath / "clean_database_output"

MaxAtoms = 8
scaling_factor = 627.50960803 # Hartree to kcal/mol
Group_to_Use = 'A'
MaxBondLength = 2.0 # Angstroms
HeavyAtomLimit = 2
LongBondLimit = 1.60
testanglelimit4 = 29.0
testanglelimit3A = 20.0
testanglelimit3B = 35.0
testanglelimit2 = 20.0

unwanted_files = [
    "69104A1a.out",
    "69106A1a.out",
    "69107A1a.out",
    "69117A1a.out",
    "69118A1a.out",
    "69122A1a.out",
    "69123A1a.out",
]

PTable = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', \
          'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', \
          'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', \
          'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', \
          'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', \
          'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', \
          'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']

