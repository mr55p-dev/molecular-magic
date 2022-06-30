"""
Original name CreateFeatures_Gy_v1_withsave

This file serves the purpose of opening a pickled array of GDB9molecule objects and compiling a feature and labels vector from them.
It uses the functions defined in `create_features_function_defs` and the constant settings defined in `create_features_config`
"""

from pathlib import Path
import pickle
import sys
from typing import List

import numpy as np
from openbabel import openbabel as ob

from create_features_function_defs import (
    AtomTypeNumbers,
    BuildOBMol,
    ExtractEnergy,
    ExtractFilename,
    SelectHBonds,
    GetAtomSymbol,
    FindKDEMax,
    SingleDataDist,
    GenerateFeatures,
)
from create_features_config import (
    cleaned_database_path,
    output_basepath,
    plot_location,
    # Group_to_Use,
    VersionNo,
    atomweight,
    bondweight,
    XHbondkdewidth,
    anglekdewidth,
    bondkdewidth,
    learning_rate,
    decay_rate,
    batch_size
)
from Database import GDB9molecule


print("**********************************************************************")

print("CreateFeatures_" + VersionNo + ".py")

print("Description: Reads in Gaussian output files, test for convergence and")
print("reads in the corresponding .xyz file to create the feature vector X")
print("and target vector y for machine learning")

print("Author: Sanha Lee")
print("University of Cambridge")

print("**********************************************************************")


# -- Import data
with open(cleaned_database_path, "rb") as read_file:
    GDB_data: List[GDB9molecule] = pickle.load(read_file)

print("Read in " + str(len(GDB_data)) + " samples")
print("Number of samples " + str(len(GDB_data)))

# -- Max_Atomtypes: Returns a dictionary of maximum number of atoms of each element type
Max_Atomtypes = AtomTypeNumbers(GDB_data)
print("Max Atomtypes")
print(Max_Atomtypes)

# -- Define lists to store important data --
GDB_compounds = []
GDB_OBmols = []  # list of all molecules in the database as OBmol object
OBmol_data = []  # list of list containing bond, angle, dihedral etc for each molecules
OBmol_atoms = []  # list of atoms for each molecule in the database
RingVariants = []  # Variable to save all possible ring structures found in the dataset


print("Creating OBmol object for all input files")

y = []  # target vector for machine learning


###########################################################################################
# Feature generation step 1 summary:
# - Read in the molecule objects from the database and generate OBmol objects.
# - Iterate over all OBmol objects and extract data for bond, angle, dihedral etc
# - Generate histograms and find KDE maximas and minimas, the KDE maximas will become features
# - OBmol_data contains bond, angle, dihedral etc data for each molecule
# - BondLengths contains overall bond data for all molecules in the database
###########################################################################################

# -- These variables will be used to generate the distribution plots --
BondLengths = (
    dict()
)  # create a dictionary key for every new bond type e.g. 'CH' and append bond legnth as values for every time that particular bond is detected
AnglesTypes = (
    dict()
)  # create a dictionary key for every new angle type e.g. 'C-C-H' and append angles as values for every time that particular angle is detected
DihedralTypes = dict()  # create a dictionary key for every new dihedral type
PartCharges = (
    dict()
)  # create a dictionary key for every new atom type e.g. 'C' and append partial charge as values for every time that particular atom is detected
HbondTypes = (
    dict()
)  # dictionary to collect H bond data {'OH-O' = [value1, value2, ...], 'NH-O' = [value1, value2, ...], etc}
AtomTypeFeatures = []

molecule_count = 0
first10names = []
filenames = []

sample_outputnumber = len(GDB_data)

for mol in GDB_data:

    # Save some props from the molecules identity
    GDB_OBmols.append(BuildOBMol(mol.atoms, mol.coords))
    OBmol_atoms.append(mol.atoms)
    filenames.append(ExtractFilename(mol))

    # Save the energy
    y.append(ExtractEnergy(mol))

    # this gathers the geometry file names for the first ten molecules. This will be used at the end to print the X vector for first 10 molecules
    if molecule_count < sample_outputnumber:
        first10names += [ExtractFilename(mol)]

    molecule_count += 1

    sys.stdout.write("\r")
    sys.stdout.write(
        str(round(100.0 * float(GDB_data.index(mol)) / float(len(GDB_data)), 1)) + "%"
    )
    sys.stdout.flush()

# -- Generate target vector array and save as pickle file
y = np.array(y)

# -- Loop over all OBmol objects created for all molecules and save all the types of bond in the database
for OBmol in GDB_OBmols:
    # -- Same dictionary definition as above but these are for one particular molecule, not for the entire database
    MolBondData = dict()
    MolAngleData = dict()
    MolDihedralData = dict()
    MolAtomData = dict()
    MolHbondData = dict()
    MolNHatomData = {"NH1": 0, "NH2": 0, "NH3": 0}

    # -- Extract N atoms with one, two and three H atoms
    for OBmolatom in ob.OBMolAtomIter(OBmol):  # iterate through all atoms in OBmol
        valencedata = str(
            OBmolatom.GetTotalValence()
        )  # GetValence() is no longer bound
        atomdata = str(OBmolatom.GetType())

        if "N" in atomdata:

            number_of_neighbourH = 0

            for neighbour_atom in ob.OBAtomAtomIter(OBmolatom):
                neighbour_atomtype = str(neighbour_atom.GetType())

                if neighbour_atomtype[0] == "H":
                    number_of_neighbourH += 1

            if number_of_neighbourH > 0:
                MolNHatomData["NH" + str(number_of_neighbourH)] += 1

    # -- Extract hydrogen bonding data
    OBmol_Hbondtype, OBmol_Hbonddata = SelectHBonds(OBmol)

    for index in range(len(OBmol_Hbondtype)):

        if str(OBmol_Hbondtype[index]) in HbondTypes.keys():
            HbondTypes[str(OBmol_Hbondtype[index])].append(OBmol_Hbonddata[index])
        else:
            HbondTypes[str(OBmol_Hbondtype[index])] = [OBmol_Hbonddata[index]]

        if str(OBmol_Hbondtype[index]) in MolHbondData.keys():
            MolHbondData[str(OBmol_Hbondtype[index])].append(OBmol_Hbonddata[index])
        else:
            MolHbondData[str(OBmol_Hbondtype[index])] = [OBmol_Hbonddata[index]]

    # -- Iterate over all bonds
    for OBmolbond in ob.OBMolBondIter(OBmol):
        bondtype = str(GetAtomSymbol(OBmolbond.GetBeginAtom().GetAtomicNum())) + str(
            GetAtomSymbol(OBmolbond.GetEndAtom().GetAtomicNum())
        )

        if (
            bondtype in BondLengths.keys()
        ):  # check whether, for example, 'CH' already exists in the dictionary
            BondLengths[bondtype].append(OBmolbond.GetLength())
        elif (
            bondtype[::-1] in BondLengths.keys()
        ):  # check for example 'CH' bond vs 'HC' bond
            BondLengths[bondtype[::-1]].append(OBmolbond.GetLength())
        else:
            BondLengths[bondtype] = [
                OBmolbond.GetLength()
            ]  # add new bond, for example, 'CH' if doesn't already exist in the dictionary

        if bondtype in MolBondData.keys():
            MolBondData[bondtype].append(OBmolbond.GetLength())
        elif bondtype[::-1] in MolBondData.keys():
            MolBondData[bondtype[::-1]].append(OBmolbond.GetLength())
        else:
            if bondtype in BondLengths.keys():
                MolBondData[bondtype] = [OBmolbond.GetLength()]
            elif bondtype[::-1] in BondLengths.keys():
                MolBondData[bondtype[::-1]] = [OBmolbond.GetLength()]
            else:
                print("Bondtype does not exist in MolBondData or BondLenths")
                exit()

    # -- Iterate over all angles
    for OBmolangle in ob.OBMolAngleIter(OBmol):

        angletype = [
            GetAtomSymbol(OBmol.GetAtom(OBmolangle[0] + 1).GetAtomicNum()),
            GetAtomSymbol(OBmol.GetAtom(OBmolangle[1] + 1).GetAtomicNum()),
            GetAtomSymbol(OBmol.GetAtom(OBmolangle[2] + 1).GetAtomicNum()),
        ]  # vertex first

        # -- Test for carbon coordination:
        # create separate group depending on central carbon coordination number
        # for example CC3C means CCC bond angle where the central carbon has coordination number of 3
        if str(angletype[0]) == "C":
            C_coordno = OBmol.GetAtom(OBmolangle[0] + 1).GetTotalValence()  # CHANGED
            angletype[0] = angletype[0] + str(C_coordno)

        if (
            str(angletype[1] + angletype[0] + angletype[2]) in AnglesTypes.keys()
        ):  # check whether, for example, angle 'CCC' exists in the dictionary
            AnglesTypes[str(angletype[1] + angletype[0] + angletype[2])].append(
                OBmol.GetAngle(
                    OBmol.GetAtom(OBmolangle[1] + 1),
                    OBmol.GetAtom(OBmolangle[0] + 1),
                    OBmol.GetAtom(OBmolangle[2] + 1),
                )
            )
        elif str(angletype[2] + angletype[0] + angletype[1]) in AnglesTypes.keys():
            AnglesTypes[str(angletype[2] + angletype[0] + angletype[1])].append(
                OBmol.GetAngle(
                    OBmol.GetAtom(OBmolangle[2] + 1),
                    OBmol.GetAtom(OBmolangle[0] + 1),
                    OBmol.GetAtom(OBmolangle[1] + 1),
                )
            )
        else:
            AnglesTypes[str(angletype[1] + angletype[0] + angletype[2])] = [
                OBmol.GetAngle(
                    OBmol.GetAtom(OBmolangle[1] + 1),
                    OBmol.GetAtom(OBmolangle[0] + 1),
                    OBmol.GetAtom(OBmolangle[2] + 1),
                )
            ]

        if str(angletype[1] + angletype[0] + angletype[2]) in MolAngleData.keys():
            MolAngleData[str(angletype[1] + angletype[0] + angletype[2])].append(
                OBmol.GetAngle(
                    OBmol.GetAtom(OBmolangle[1] + 1),
                    OBmol.GetAtom(OBmolangle[0] + 1),
                    OBmol.GetAtom(OBmolangle[2] + 1),
                )
            )
        elif str(angletype[2] + angletype[0] + angletype[1]) in MolAngleData.keys():
            MolAngleData[str(angletype[2] + angletype[0] + angletype[1])].append(
                OBmol.GetAngle(
                    OBmol.GetAtom(OBmolangle[2] + 1),
                    OBmol.GetAtom(OBmolangle[0] + 1),
                    OBmol.GetAtom(OBmolangle[1] + 1),
                )
            )
        else:
            if str(angletype[1] + angletype[0] + angletype[2]) in AnglesTypes.keys():
                MolAngleData[str(angletype[1] + angletype[0] + angletype[2])] = [
                    OBmol.GetAngle(
                        OBmol.GetAtom(OBmolangle[1] + 1),
                        OBmol.GetAtom(OBmolangle[0] + 1),
                        OBmol.GetAtom(OBmolangle[2] + 1),
                    )
                ]
            elif str(angletype[2] + angletype[0] + angletype[1]) in AnglesTypes.keys():
                MolAngleData[str(angletype[2] + angletype[0] + angletype[1])] = [
                    OBmol.GetAngle(
                        OBmol.GetAtom(OBmolangle[2] + 1),
                        OBmol.GetAtom(OBmolangle[0] + 1),
                        OBmol.GetAtom(OBmolangle[1] + 1),
                    )
                ]
            else:
                print("angletype does not exist in MolAngleData or AnglesTypes")
                exit()

    # -- Iterate over all dihedrals
    for OBdihedral in ob.OBMolTorsionIter(OBmol):

        dihedraltype = [
            GetAtomSymbol(OBmol.GetAtom(OBdihedral[0] + 1).GetAtomicNum()),
            GetAtomSymbol(OBmol.GetAtom(OBdihedral[1] + 1).GetAtomicNum()),
            GetAtomSymbol(OBmol.GetAtom(OBdihedral[2] + 1).GetAtomicNum()),
            GetAtomSymbol(OBmol.GetAtom(OBdihedral[3] + 1).GetAtomicNum()),
        ]  # CHANGED

        # print(dihedraltype)
        # -- Test for carbon coordination:
        # create separate group for linear CC bonds

        # if str(dihedraltype[1]) == 'C' and str(dihedraltype[2]) == 'C':
        #    C1_coordno = str(OBmol.GetAtom(OBdihedral[1]+1).GetValence())
        #    C2_coordno = str(OBmol.GetAtom(OBdihedral[2]+1).GetValence())

        # print(C1_coordno)
        # print(C2_coordno)

        #    if C1_coordno == '2' and C2_coordno == '2': # select bonds where C1 and C2 have two neighbours
        #        dihedraltype[1] = dihedraltype[1] + str(C1_coordno)
        #        dihedraltype[2] = dihedraltype[2] + str(C2_coordno)

        if (
            str(dihedraltype[0] + dihedraltype[1] + dihedraltype[2] + dihedraltype[3])
            in DihedralTypes.keys()
        ):
            DihedralTypes[
                str(
                    dihedraltype[0]
                    + dihedraltype[1]
                    + dihedraltype[2]
                    + dihedraltype[3]
                )
            ].append(
                abs(
                    OBmol.GetTorsion(
                        OBmol.GetAtom(OBdihedral[0] + 1),
                        OBmol.GetAtom(OBdihedral[1] + 1),
                        OBmol.GetAtom(OBdihedral[2] + 1),
                        OBmol.GetAtom(OBdihedral[3] + 1),
                    )
                )
            )
        elif (
            str(dihedraltype[3] + dihedraltype[2] + dihedraltype[1] + dihedraltype[0])
            in DihedralTypes.keys()
        ):
            DihedralTypes[
                str(
                    dihedraltype[3]
                    + dihedraltype[2]
                    + dihedraltype[1]
                    + dihedraltype[0]
                )
            ].append(
                abs(
                    OBmol.GetTorsion(
                        OBmol.GetAtom(OBdihedral[3] + 1),
                        OBmol.GetAtom(OBdihedral[2] + 1),
                        OBmol.GetAtom(OBdihedral[1] + 1),
                        OBmol.GetAtom(OBdihedral[0] + 1),
                    )
                )
            )
        else:
            DihedralTypes[
                str(
                    dihedraltype[0]
                    + dihedraltype[1]
                    + dihedraltype[2]
                    + dihedraltype[3]
                )
            ] = [
                abs(
                    OBmol.GetTorsion(
                        OBmol.GetAtom(OBdihedral[0] + 1),
                        OBmol.GetAtom(OBdihedral[1] + 1),
                        OBmol.GetAtom(OBdihedral[2] + 1),
                        OBmol.GetAtom(OBdihedral[3] + 1),
                    )
                )
            ]

        if (
            str(dihedraltype[0] + dihedraltype[1] + dihedraltype[2] + dihedraltype[3])
            in MolDihedralData.keys()
        ):
            MolDihedralData[
                str(
                    dihedraltype[0]
                    + dihedraltype[1]
                    + dihedraltype[2]
                    + dihedraltype[3]
                )
            ].append(
                abs(
                    OBmol.GetTorsion(
                        OBmol.GetAtom(OBdihedral[0] + 1),
                        OBmol.GetAtom(OBdihedral[1] + 1),
                        OBmol.GetAtom(OBdihedral[2] + 1),
                        OBmol.GetAtom(OBdihedral[3] + 1),
                    )
                )
            )
        elif (
            str(dihedraltype[3] + dihedraltype[2] + dihedraltype[1] + dihedraltype[0])
            in MolDihedralData.keys()
        ):
            MolDihedralData[
                str(
                    dihedraltype[3]
                    + dihedraltype[2]
                    + dihedraltype[1]
                    + dihedraltype[0]
                )
            ].append(
                abs(
                    OBmol.GetTorsion(
                        OBmol.GetAtom(OBdihedral[3] + 1),
                        OBmol.GetAtom(OBdihedral[2] + 1),
                        OBmol.GetAtom(OBdihedral[1] + 1),
                        OBmol.GetAtom(OBdihedral[0] + 1),
                    )
                )
            )
        else:
            if (
                str(
                    dihedraltype[0]
                    + dihedraltype[1]
                    + dihedraltype[2]
                    + dihedraltype[3]
                )
                in DihedralTypes.keys()
            ):
                MolDihedralData[
                    str(
                        dihedraltype[0]
                        + dihedraltype[1]
                        + dihedraltype[2]
                        + dihedraltype[3]
                    )
                ] = [
                    abs(
                        OBmol.GetTorsion(
                            OBmol.GetAtom(OBdihedral[0] + 1),
                            OBmol.GetAtom(OBdihedral[1] + 1),
                            OBmol.GetAtom(OBdihedral[2] + 1),
                            OBmol.GetAtom(OBdihedral[3] + 1),
                        )
                    )
                ]
            elif (
                str(
                    dihedraltype[3]
                    + dihedraltype[2]
                    + dihedraltype[1]
                    + dihedraltype[0]
                )
                in DihedralTypes.keys()
            ):
                MolDihedralData[
                    str(
                        dihedraltype[3]
                        + dihedraltype[2]
                        + dihedraltype[1]
                        + dihedraltype[0]
                    )
                ] = [
                    abs(
                        OBmol.GetTorsion(
                            OBmol.GetAtom(OBdihedral[3] + 1),
                            OBmol.GetAtom(OBdihedral[2] + 1),
                            OBmol.GetAtom(OBdihedral[1] + 1),
                            OBmol.GetAtom(OBdihedral[0] + 1),
                        )
                    )
                ]
            else:
                print("dihedraltype does not exist in MolDihedralData or DihedralTypes")
                exit()

    # -- Extract number of each atom types
    atoms_data = OBmol_atoms[GDB_OBmols.index(OBmol)]
    for atom in atoms_data:
        MolAtomData[atom] = (
            MolAtomData.get(atom, 0) + 1
        )  # '0' is the value to return if the specified key does not exist

    # print(MolAtomData)

    # -- Save all information about the molecule to OBmol_data
    OBmol_data += [
        [
            MolBondData,
            MolAngleData,
            MolDihedralData,
            MolAtomData,
            MolHbondData,
            MolNHatomData,
        ]
    ]

    # Print feature generation progress as % in console
    sys.stdout.write("\r")
    sys.stdout.write(
        str(round(100.0 * float(GDB_OBmols.index(OBmol)) / float(len(GDB_OBmols)), 1))
        + "%"
    )
    sys.stdout.flush()


###########################################################################################
# Feature generation step 2 summary:
# - The first step is to find all maxima and minima on KDE plots for all distribution.
# - BondFeaturesDef etc are dictionaries containing KDE maximas and KDE minimas.
# - These dictionaries are also stored as pickle files
###########################################################################################

# -- Define dictionaries to generate features matrix X
BondFeaturesDef = {}
AngleFeaturesDef = {}
DihedralFeaturesDef = {}
HbondFeaturesDef = {}


print("\n")
print("Following H-bonds have been detected:")
print(list(HbondTypes.keys()))


# Plot all observed Hbond distributions
for key, value in HbondTypes.items():

    print(str(key) + ": " + str(len(value)) + " number of H-bonds")

    if len(value) > 1:
        Hbond_KDEmaximas, Hbond_maximas, Hbond_minimas = FindKDEMax(
            value,
            "Hbond",
            plot_location / "hbond",
            str(key),
            XHbondkdewidth,
        )
    else:
        Hbond_maximas, Hbond_minimas = SingleDataDist(value, "./HbondPlots/", str(key))

    # print('Hbond_KDEmaxias: ',Hbond_KDEmaximas)

    HbondFeaturesDef[str(key)] = [Hbond_maximas, Hbond_minimas]


print("HbondFeaturesDef:")
print(HbondFeaturesDef)


print("Following bonds have been detected:")
print(list(BondLengths.keys()))

# -- Plot all observed bond distributions by looping over all bond types
for key, value in BondLengths.items():

    print(str(key) + ": " + str(len(value)) + " number of bonds")

    if len(value) > 1 and "H" in str(key):  # set different KDE width for XH bonds
        bond_KDEmaximas, bond_maximas, bond_minimas = FindKDEMax(
            value,
            "XHbond",
            plot_location / "bond/",
            str(key),
            XHbondkdewidth,
        )
    elif len(value) > 1 and "H" not in str(key):  # normal KDE width for all other bonds
        bond_KDEmaximas, bond_maximas, bond_minimas = FindKDEMax(
            value, "bond", plot_location / "bond/", str(key), bondkdewidth
        )
    else:
        bond_maximas, bond_minimas = SingleDataDist(
            value, plot_location / "bond/", str(key)
        )  # if only one maxima exists in the distribution

    # print('bond_KDEmaximas: ', bond_KDEmaximas)
    BondFeaturesDef[str(key)] = [bond_maximas, bond_minimas]


print("Following angles have been detected:")
print(list(AnglesTypes.keys()))

# -- Plot all observed angle distributions by looping over all angle gypes
for key, value in AnglesTypes.items():

    print(str(key) + ": " + str(len(value)) + " number of angles")

    if len(value) > 1:
        angle_KDEmaximas, angle_maximas, angle_minimas = FindKDEMax(
            value,
            "angle",
            plot_location / "angle/",
            str(key),
            anglekdewidth,
        )
    else:
        angle_maximas, angle_minimas = SingleDataDist(
            value, plot_location / "angle/", str(key)
        )

    # print('angle_KDEmaximas: ', angle_KDEmaximas)
    AngleFeaturesDef[str(key)] = [angle_maximas, angle_minimas]


print("Following dihedrals have been detected:")
print(list(DihedralTypes.keys()))

# -- Plot all observed dihedral distributions by looping over all dihedral types
for key, value in DihedralTypes.items():

    print(str(key) + ": " + str(len(value)) + " number of dihedrals")

    if len(value) > 1:
        dihedral_KDEmaximas, dihedral_maximas, dihedral_minimas = FindKDEMax(
            value,
            "dihedral",
            plot_location / "dihedral/",
            str(key),
            anglekdewidth,
        )
    else:
        dihedral_maximas, dihedral_minimas = SingleDataDist(
            value, plot_location / "dihedral/", str(key)
        )

    # print('dihedral_KDEmaximas: ', dihedral_KDEmaximas)
    DihedralFeaturesDef[str(key)] = [dihedral_maximas, dihedral_minimas]


print("Max_Atomtypes: ", Max_Atomtypes)


print("Saving features dictionaries - use them to generate features on a new dataset")

# BondFeaturesDefFile = open(
#     "./" + plot_location + "/" + plot_location + "_BondFeaturesDef.plk", "wb"
# )
# pickle.dump(BondFeaturesDef, BondFeaturesDefFile)
# BondFeaturesDefFile.close()
# print("Saved BondFeaturesDef")
# print(BondFeaturesDef)

# AngleFeaturesDefFile = open(
#     "./" + plot_location + "/" + plot_location + "_AngleFeaturesDef.plk", "wb"
# )
# pickle.dump(AngleFeaturesDef, AngleFeaturesDefFile)
# AngleFeaturesDefFile.close()
# print("Saved AngleFeaturesDef")
# print(AngleFeaturesDef)

# DihedralFeaturesDefFile = open(
#     "./" + plot_location + "/" + plot_location + "_DihedralFeaturesDef.plk", "wb"
# )
# pickle.dump(DihedralFeaturesDef, DihedralFeaturesDefFile)
# DihedralFeaturesDefFile.close()
# print("Saved DihedralFeaturesDef")
# print(DihedralFeaturesDef)

# MaxAtomTypesDefFile = open(
#     "./" + plot_location + "/" + plot_location + "_MaxAtomTypesDef.plk", "wb"
# )
# pickle.dump(Max_Atomtypes, MaxAtomTypesDefFile)
# MaxAtomTypesDefFile.close()
# print("Saved MaxAtomTypesDef")
# print(Max_Atomtypes)

# HbondFeaturesDefFile = open(
#     "./" + plot_location + "/" + plot_location + "_HbondFeaturesDef.plk", "wb"
# )
# pickle.dump(HbondFeaturesDef, HbondFeaturesDefFile)
# MaxAtomTypesDefFile.close()
# print("Saved HbondFeaturesDef")
# print(HbondFeaturesDef)


###########################################################################################
# Feature generation step 3 summary:
# - This step creates features matrix X. The maximas on KDE plots for bonds, angles, dihedrals etc becomes the feature components.
# - The algorithm calculates number of times a particular feature component exists for each molecules, for example, how many 'CC' bonds does the molecule contain?
# - This is then added to the matrix X for each molecule.
###########################################################################################

print("The algorithm will use the following weights:")
print("bondweight: ", atomweight)
print("angleweight: ", bondweight)

# Features vector

print("Generating features matrix X for all molecules")

X, read_X = GenerateFeatures(
    OBmol_data,
    BondFeaturesDef,
    AngleFeaturesDef,
    DihedralFeaturesDef,
    Max_Atomtypes,
    HbondFeaturesDef,
    filenames,
)


print("Matrix X preview:")
print(X)

print("Vector y preview:")
print(y)

print("Readable matrix X:")
print(read_X)


list_X = X.tolist()

with open(output_basepath / "first_10.txt", "w") as first10file:
    for vector in list_X[0:sample_outputnumber]:
        string_vector = " ".join(str(component) for component in vector)
        first10file.write(str(list_X.index(vector) + 1) + "\n")
        first10file.write(first10names[list_X.index(vector)] + "\n")
        first10file.write(string_vector + "\n")

with open(
    output_basepath / "readable_X.txt",
    "w",
) as readableXfile:
    for index in range(len(read_X)):
        readableXfile.write(read_X[index] + " ")


# -- Save key vectors
data_output_dir = output_basepath / "data"
data_output_dir.mkdir(parents=True, exist_ok=True)

np.save(
    data_output_dir / "features.npy",
    X,
    allow_pickle=True,
)
np.save(
    data_output_dir / "labels.npy",
    y,
    allow_pickle=True,
)


print("There are " + str(len(X[0])) + " features")
print("Successfully created feature vector X and target vector y")
