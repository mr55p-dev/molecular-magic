##############################################################################
# Settings
##############################################################################

"""

Code setup definitions:

MaxAtoms            Maximum number of atoms the molecules in the database can contain
PickleFile          Pickle file to import data
plot_location       Directory to store the KDE plots
scaling_factor      Scaling factor to convert energy from hartree to a different unit
Group_to_Use
MaxBondLength       In KDE distributions, if atomic separation is longer than this length, do not consider as a bond
VersionNo           Python code version number
atomweight          How much weight to give to the atomic part of the features matrix
bondweight          How much weight to give to the bond part of the features matrix
bondkdewidth        KDE widths for bond distributions
anglekdewidth       KDE widths for angle distributions
sampleoutputnumber  Ouput features vector X upto this number

"""

from pathlib import Path

data_basepath = Path("./static_data/")
cleaned_database_path = data_basepath / "clean_database_output" / "data" / "cleaned_data.pkl"

MaxAtoms = 8
output_basepath = Path("static_data/create_features_output/")
plot_location = output_basepath / "plots/"
scaling_factor = 627.509608030593  # kcal/mol
Group_to_Use = "A"
MaxBondLength = 2.0  # Angstroms
VersionNo = "v1"
atomweight = 100
bondweight = 1
bondkdewidth = 0.07
XHbondkdewidth = 0.3
anglekdewidth = 0.07
fileswanted = ["022090A1a.out", "000779A1a.out", "45253A1a.out", "020348A1a.out"]


PTable = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
]
