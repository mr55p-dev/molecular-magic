import pickle
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import openbabel as ob
from scipy.stats import gaussian_kde


"""Original name CreateFeatures_Gy_v1_withsave"""

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


MaxAtoms = 8
PickleFile = "GDB_allgroundnoimagnocarban_Dec2021_G298_LongBondLimit1.6.plk"
database_loc = "."
plot_location = "CreateFeatures_v20_fAng_fNH_B0p07_A0p07_G298_2"
scaling_factor = 627.509608030593  # kcal/mol
Group_to_Use = "A"
MaxBondLength = 2.0  # Angstroms
VersionNo = "v1"
atomweight = 100
bondweight = 1
bondkdewidth = 0.07
XHbondkdewidth = 0.3
anglekdewidth = 0.07


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


##############################################################################
# Functions
##############################################################################


def LoadDBFromPickle(pklfile):
    from databases import Database
    with open(database_loc + "/" + pklfile, "rb") as read_file:
        return pickle.load(read_file)


def AtomTypeNumbers(GDBclass):
    """
    Returns a dictionary of maximum number of atoms of each element type

    """
    print("")
    print("Calculating maximum number of atoms of each element type")

    element_type = dict()

    for index in range(1, len(GDBclass)):
        file_dict = Counter(GDBclass[index].atoms)

        # check whether the element type in molecule exists in overall element type dictionary
        for key in file_dict:

            if key in element_type.keys():
                if file_dict[str(key)] > element_type[str(key)]:
                    element_type[str(key)] = file_dict[str(key)]
                else:
                    pass
            else:
                element_type[str(key)] = file_dict[str(key)]

    return element_type


def BuildOBMol(atoms, coords):
    """
    Generate OBmol object

    """

    mol = ob.OBMol()
    for anum, acoords in zip(atoms, coords):
        atom = ob.OBAtom()
        atom.thisown = False
        atom.SetAtomicNum(GetAtomNum(anum))
        atom.SetVector(acoords[0], acoords[1], acoords[2])
        mol.AddAtom(atom)

    # Restore the bonds
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()

    # mol.Kekulize()

    return mol


def ExtractEnergy(GDBmolecule):
    """
    Returns energy of GDB molecule
    Change the scaling factor to the desired units

    """

    energy = float(GDBmolecule.G298) * scaling_factor

    return energy


def GetAtomSymbol(AtomNum):

    if AtomNum > 0 and AtomNum < len(PTable):
        return PTable[AtomNum - 1]
    else:
        print("No such element with atomic number " + str(AtomNum))
        return 0


def GetAtomNum(AtomSymbol):

    if AtomSymbol in PTable:
        return PTable.index(AtomSymbol) + 1
    else:
        print("No such element with symbol " + str(AtomSymbol))
        return 0


def FindKDEMax(distribution_data, distribution_type, location, name, kde_width):

    KDE_maximas = []

    kde = gaussian_kde(distribution_data, bw_method=kde_width)

    if distribution_type == "bond":
        x_data = np.linspace(0.8, MaxBondLength, 10000)
    elif distribution_type == "XHbond":
        x_data = np.linspace(0.8, MaxBondLength, 10000)
    elif distribution_type == "angle" or distribution_type == "dihedral":
        x_data = np.linspace(0, 200, 10000)
    elif distribution_type == "Hbond":
        x_data = np.linspace(1.2, 3.0, 10000)

    y_data = kde.evaluate(x_data)

    dydx = np.diff(y_data, 1)

    KDE_maxima = np.where(
        (y_data > np.roll(y_data, 1)) & (y_data > np.roll(y_data, -1))
    )  # Returns Tupule
    KDE_minima = np.where(
        (y_data < np.roll(y_data, 1)) & (y_data < np.roll(y_data, -1))
    )
    KDE_maxima = KDE_maxima[0]
    KDE_minima = KDE_minima[0]
    KDE_maximas += [KDE_maxima]
    # WARNING: check the y_data to see whether it starts from 0.0 : change KDE_minima list accordingly
    # print('y_data: ', y_data[:10])

    if distribution_type == "Hbond":
        KDE_minima = KDE_minima[1:]

    y_forKDEmaxima = [0] * len(list(KDE_maxima))
    y_forKDEminima = [0] * len(list(KDE_minima))

    x_minimas = list([x_data[i] for i in KDE_minima])
    x_maximas = list([x_data[i] for i in KDE_maxima])

    plt.figure()

    ax = plt.gca()
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)

    plt.tick_params(axis="both", labelsize=14)

    if distribution_type == "bond":
        plt.xlabel("Bond Length / $\AA$", fontsize=14, fontweight="bold")
        plt.xlim([round(min(x_maximas) - 0.2, 1), round(max(x_maximas) + 0.2, 1)])
        plt.hist(
            distribution_data,
            bins=100,
            density=True,
            color="#CACFD2",
            label=name + " hist",
        )
        dydx = dydx / np.max(dydx)

    elif distribution_type == "XHbond":
        plt.xlabel("Bond Length / $\AA$", fontsize=14, fontweight="bold")
        plt.xlim([round(min(x_maximas) - 0.05, 1), round(max(x_maximas) + 0.05, 1)])
        plt.hist(
            distribution_data,
            bins=100,
            density=True,
            color="#CACFD2",
            label=name + " hist",
        )
        dydx = dydx / np.max(dydx)

    elif distribution_type == "angle" or distribution_type == "dihedral":
        plt.xlabel("Angle / deg", fontsize=14, fontweight="bold")
        plt.xlim([round(min(x_maximas) - 5.0), round(max(x_maximas) + 5.0)])
        plt.hist(
            distribution_data,
            bins=100,
            density=True,
            color="#CACFD2",
            label=name + " hist",
        )
        dydx = (dydx / np.max(dydx)) * 0.1

    elif distribution_type == "Hbond":
        plt.xlabel("Bond Length / $\AA$", fontsize=14, fontweight="bold")
        plt.xlim([1.1, 3.0])
        plt.hist(
            distribution_data,
            bins=100,
            density=True,
            color="#CACFD2",
            label=name + " hist",
        )
        dydx = dydx / np.max(dydx)

    elif len(distribution_data) <= 1:
        pass

    plt.ylabel("Freq.", fontsize=14, fontweight="bold")

    plt.plot(x_data, y_data, label="KDE fit", color="#17202A")
    plt.plot(x_data[1:], dydx, color="#A93226", label="KDE grad")
    plt.plot(
        x_data[KDE_maxima],
        y_forKDEmaxima,
        "o",
        color="#A93226",
        label="maxima",
        markersize=3,
    )
    plt.plot(
        x_data[KDE_minima],
        y_forKDEminima,
        "^",
        color="#A93226",
        label="minima",
        markersize=3,
    )

    # plt.ylim([-1.2,1.2])

    plt.legend()
    plt.savefig(location + name + ".png", format="png", dpi=500)
    plt.close()

    np.save(location + name, distribution_data, allow_pickle=True)

    x_maximas_outputfile = open(location + name + ".txt", "w")
    for item in x_maximas:
        x_maximas_outputfile.write(str(item) + ",")
    x_maximas_outputfile.close()

    return KDE_maximas, x_maximas, x_minimas


def SingleDataDist(distribution_data, location, name):

    x_maximas = distribution_data
    x_minimas = []

    return x_maximas, x_minimas


def GenerateFeatures(
    OBmol_data, bonddict, angledict, dihedraldict, maxatomtypes, hbonddict, FileNames
):
    """

    Creates fecture matrix X for all OBmol database from the bond, angle,
    dihedral dictionaries

    bondict is the list of maximas and minimas found in the bond distribution
    for all bonds found in the entire dataset. Likewise for angledict and
    dihedraldict.

    Note: the bonddict, angledict, dihedraldict dictionary values are in the
    format [[maximas], [minimas]]

    OBmol_data += [[MolBondData, MolAngleData, MolDihedralData, MolAtomData, MolHbondData, MolNHatomData]]

    """

    X = []
    readable_X = []

    # Generate readable X vector for the output:

    for bonddict_key, bonddict_values in bonddict.items():
        feat_vec = [bonddict_key] * len(bonddict_values[0])
        readable_X += feat_vec

    for angledict_key, angledict_values in angledict.items():
        feat_vec = [angledict_key] * len(angledict_values[0])
        readable_X += feat_vec

    for dihedraldict_key, dihedraldict_values in dihedraldict.items():
        feat_vec = [dihedraldict_key] * len(dihedraldict_values[0])
        readable_X += feat_vec

    for atom_key, atom_values in maxatomtypes.items():
        feat_vec = [atom_key]
        readable_X += feat_vec

    for Hbond_key, Hbond_values in hbonddict.items():
        feat_vec = [Hbond_key] * len(Hbond_values[0])
        readable_X += feat_vec

    feat_vec = ["NH1", "NH2", "NH3"]
    readable_X += feat_vec

    # -- Now generate actual feature vector X
    for OBmol in OBmol_data:

        # -- Extract all data for OBmol
        MolBondDict = OBmol[0]
        MolBondKeys = MolBondDict.keys()
        MolAngleDict = OBmol[1]
        MolAngleKeys = MolAngleDict.keys()
        MolDihedralDict = OBmol[2]
        MolDihedralKeys = MolDihedralDict.keys()
        MolAtomDict = OBmol[3]
        MolAtomKeys = list(MolAtomDict.keys())
        MolHbondDict = OBmol[4]
        MolHbondKeys = list(MolHbondDict.keys())
        MolNHatomDict = OBmol[5]
        MolNHatomKeys = list(MolNHatomDict.keys())
        MolNHatomValues = list(MolNHatomDict.values())
        Mol_X = []

        FileName = FileNames[OBmol_data.index(OBmol)]

        # -- Bond Features Vector Generation
        # -- (use 'print' statements to test the code is working properly)

        if FileName in fileswanted:

            print("MolBondDict:")
            print(MolBondDict)

        for bonddict_key, bonddict_values in bonddict.items():

            if FileName in fileswanted:
                print(FileName)
                # bonddict_key in format 'CC', 'CH', etc for total dictionary
                # bonddict_values in format [[maximas], [minimas]]

                print("bonddict_key: ", bonddict_key)
                print("maximas: ", bonddict_values[0])
                print("minimas: ", bonddict_values[1])

            feat_vec = [0] * len(
                bonddict_values[0]
            )  # create list of zeros, same size as list of maximas

            if (
                bonddict_key in MolBondKeys
            ):  # check if particular bond type from total dictionary exists in the molecule dictionary

                # print('molbonddict: ', MolBondDict[bonddict_key])

                for mol_bondvalue in MolBondDict[
                    bonddict_key
                ]:  # for each bondlength for particular bond type...
                    if all(
                        item <= mol_bondvalue for item in bonddict_values[1]
                    ):  # check whether the bond length is greater than all values in the list of minimas
                        feat_vec[-1] += (
                            1 * bondweight
                        )  # append one to the last element of the features vec (corresponds to the maxima of greatest bond length)
                    else:
                        # MOST IMPORTANT PART
                        # returns the first index of the minima list when the total dict minima value gets larger than the molecule dict bond value
                        feat_idx = next(
                            index
                            for index, bond_dictvalue in enumerate(bonddict_values[1])
                            if bond_dictvalue > mol_bondvalue
                        )
                        # else find the index of the minima where the bond length under consideration becomes larger
                        feat_vec[feat_idx] += 1 * bondweight

                if FileName in fileswanted:
                    print("bond feat_vecc:")
                    print(feat_vec)

                Mol_X += feat_vec

            else:
                # print('no such bond exist')
                Mol_X += feat_vec

            # print(bonddict_key, bonddict_values)
            # print(feat_vec)

            # print(feat_vec)
        # if FileName in fileswanted:
        #    print('Mol_X: ', Mol_X)

        # print(MolAngleDict)
        # -- Angle Features Vector Generation
        for angledict_key, angledict_values in angledict.items():

            # print('angledict_key: ', angledict_key)
            # print('maximas: ', angledict_values[0])

            feat_vec = [0] * len(angledict_values[0])

            if angledict_key in MolAngleKeys:

                for mol_anglevalue in MolAngleDict[angledict_key]:
                    if all(item <= mol_anglevalue for item in angledict_values[1]):
                        feat_vec[
                            -1
                        ] += 1  # append one to the last element of the feature vec
                    else:
                        feat_idx = next(
                            index
                            for index, angle_dictvalue in enumerate(angledict_values[1])
                            if angle_dictvalue > mol_anglevalue
                        )
                        feat_vec[feat_idx] += 1

                Mol_X += feat_vec

            else:
                Mol_X += feat_vec

            # print(angledict_key, angledict_values)
            # print(feat_vec)

        # print('Mol_X: ', Mol_X)

        # -- Dihedral Features Vector Generation

        # print(MolDihedralDict)
        for dihedraldict_key, dihedraldict_values in dihedraldict.items():

            # print('dihedraldict_key: ', dihedraldict_key)
            # print('maximas: ', dihedraldict_values[0])

            feat_vec = [0] * len(dihedraldict_values[0])

            if dihedraldict_key in MolDihedralKeys:

                for mol_dihedralvalue in MolDihedralDict[dihedraldict_key]:
                    if all(
                        item <= mol_dihedralvalue for item in dihedraldict_values[1]
                    ):
                        feat_vec[-1] += 1
                    else:
                        feat_idx = next(
                            index
                            for index, dihedral_dictvalue in enumerate(
                                dihedraldict_values[1]
                            )
                            if dihedral_dictvalue > mol_dihedralvalue
                        )
                        feat_vec[feat_idx] += 1

                Mol_X += feat_vec

            else:
                Mol_X += feat_vec

            # print(dihedraldict_key, dihedraldict_values)
            # print(feat_vec)

        # print('Mol_X: ', Mol_X)

        # -- Atom Feature Vector Generation
        feat_vec = [0] * len(maxatomtypes)

        for atom_key, atom_values in maxatomtypes.items():

            # print('atom_key: ', atom_key)

            maxatomkeys = list(maxatomtypes.keys())

            if atom_key in MolAtomKeys:
                feat_vec[maxatomkeys.index(atom_key)] = (
                    MolAtomDict[atom_key] * atomweight
                )

        Mol_X += feat_vec

        # -- Hbond Feature Vector Generation
        for hbonddict_key, hbonddict_values in hbonddict.items():

            # print('hbonddict_key: ',hbonddict_key)
            # print('maximas: ', hbonddict_values[0])

            feat_vec = [0] * len(hbonddict_values[0])

            if hbonddict_key in MolHbondKeys:

                for mol_hbondvalue in MolHbondDict[hbonddict_key]:
                    if all(item <= mol_hbondvalue for item in hbonddict_values[1]):
                        feat_vec[-1] += 1
                    else:
                        feat_idx = next(
                            index
                            for index, hbond_dictvalue in enumerate(hbonddict_values[1])
                            if hbond_dictvalue > mol_hbondvalue
                        )
                        feat_vec[feat_idx] += 1

                Mol_X += feat_vec

            else:
                Mol_X += feat_vec

        # -- NHatom Feature Vector Generation
        feat_vec = [0] * len(MolNHatomKeys)

        for index in range(len(MolNHatomKeys)):

            feat_vec[index] = MolNHatomValues[index]

        Mol_X += feat_vec

        # print('MaxAtomTypes: ', maxatomtypes)
        # print('MolAtomDict: ', MolAtomDict)
        # print('MolAtomKeys: ', MolAtomKeys)
        # print('feat_vec: ', feat_vec)

        X += [Mol_X]

        if FileName in fileswanted:
            print("Mol_X:")
            print(Mol_X)

        sys.stdout.write("\r")
        sys.stdout.write(
            str(
                round(
                    100.0 * float(OBmol_data.index(OBmol)) / float(len(OBmol_data)), 1
                )
            )
            + "% : "
            + str(OBmol_data.index(OBmol))
            + " out of "
            + str(len(OBmol_data))
            + " molecules"
        )
        sys.stdout.flush()

    return np.array(X), readable_X


def GetDistance(mol, atoms):
    """
    Returns distance of two specific atoms in OBmol

    """

    a1 = mol.GetAtom(atoms[0])

    return a1.GetDistance(atoms[1])


# Hbonders are atoms the SelectHBonds will detect as H-bonding atoms
Hbonders = ["O", "N"]


def SelectHBonds(mol):
    """
    Function to detect H bonds in OBmol and generate features
    Edit 'Hbonders' above to change the H-bonding atoms

    """
    natoms = mol.NumAtoms()
    AcidHs = []
    ENatoms = []

    # Go through all atoms
    # For all protons, check if covalently bonded to an EN atom (defined by Hbonders list)
    # Make a list of all acidic H atoms
    for a in range(1, natoms + 1):
        atom = mol.GetAtom(a)
        if atom.GetAtomicNum() != 1:
            continue
        else:
            for NbrAtom in ob.OBAtomAtomIter(
                atom
            ):  # iterate over all neighbouring atoms
                if GetAtomSymbol(NbrAtom.GetAtomicNum()) in Hbonders:
                    AcidHs.append(a)

    # Make a list of all EN atoms (defined by Hbonders list) in molecule
    ENatoms = []
    for a in range(1, natoms + 1):
        atom = mol.GetAtom(a)
        if GetAtomSymbol(atom.GetAtomicNum()) in Hbonders:
            ENatoms.append(a)

    # We take all the detected X-H...Y distances and classify them
    # as either covalent, H-bonded or non-bonded
    ActiveHBonds = []
    ActiveHDists = []
    for (
        Ha
    ) in (
        AcidHs
    ):  # for all Hs attached to e-neg atoms [[H1-E1, H1-E2, ...], [H2-E1, H2-E2, ...], etc]

        for ENa in ENatoms:  # append distances to each e-neg attoms
            H_dist = GetDistance(mol, [Ha, ENa])

            if H_dist > 1.3 and H_dist < 2.6:

                ActiveHDists += [H_dist]

                for NbrAtom in ob.OBAtomAtomIter(mol.GetAtom(Ha)):
                    bond = (
                        GetAtomSymbol(NbrAtom.GetAtomicNum())
                        + GetAtomSymbol(mol.GetAtom(Ha).GetAtomicNum())
                        + "-"
                        + GetAtomSymbol(mol.GetAtom(ENa).GetAtomicNum())
                    )
                    ActiveHBonds.append(bond)

    return ActiveHBonds, ActiveHDists


def ExtractFilename(GDBmolecule):
    """
    Returns the output filename of GDB molecule

    """
    outputfilename = GDBmolecule.geomfilename
    outputfilenamelist = outputfilename.split("/")

    return outputfilenamelist[-1]


###############################################################################
# Main Code
###############################################################################

print("")
print("**********************************************************************")
print("")
print("CreateFeatures_" + VersionNo + ".py")
print("")
print("Description: Reads in Gaussian output files, test for convergence and")
print("reads in the corresponding .xyz file to create the feature vector X")
print("and target vector y for machine learning")
print("")
print("Author: Sanha Lee")
print("University of Cambridge")
print("")
print("**********************************************************************")
print("")

# -- Import data
print("Loading pickle file " + PickleFile)
GDB_data = LoadDBFromPickle(PickleFile)
print("Read in " + str(len(GDB_data)) + " samples")
print("")

print("Number of samples " + str(len(GDB_data)))

# -- Max_Atomtypes: Returns a dictionary of maximum number of atoms of each element type
Max_Atomtypes = AtomTypeNumbers(GDB_data)

print("")
print("Max Atomtypes")
print(Max_Atomtypes)

fileswanted = ["022090A1a.out", "000779A1a.out", "45253A1a.out", "020348A1a.out"]

# -- Define lists to store important data --
GDB_compounds = []
GDB_OBmols = []  # list of all molecules in the database as OBmol object
OBmol_data = (
    []
)  # list of list containing bond, angle, dihedral etc data for each molecules
OBmol_atoms = []  # list of atoms for each molecule in the database
RingVariants = []  # Variable to save all possible ring structures found in the dataset

print("")
print("Creating OBmol object for all input files")

y = []  # target vector for machine learning


"""

Feature generation step 1 summary:
Read in the molecule objects from the database and generate OBmol objects.
Iterate over all OBmol objects and extract data for bond, angle, dihedral etc
Generate histograms and find KDE maximas and minimas, the KDE maximas will become features
OBmol_data contains bond, angle, dihedral etc data for each molecule
BondLengths contains overall bond data for all molecules in the database

"""


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

    GDB_OBmols += [BuildOBMol(mol.atoms, mol.coords)]
    OBmol_atoms += [mol.atoms]
    y += [ExtractEnergy(mol)]
    filenames += [ExtractFilename(mol)]

    if (
        molecule_count < sample_outputnumber
    ):  # this gathers the geometry file names for the first ten molecules. This will be used at the end to print the X vector for first 10 molecules
        first10names += [ExtractFilename(mol)]

    molecule_count += 1

    sys.stdout.write("\r")
    sys.stdout.write(
        str(round(100.0 * float(GDB_data.index(mol)) / float(len(GDB_data)), 1)) + "%"
    )
    sys.stdout.flush()

# -- Generate target vector array and save as pickle file
y = np.array(y)


# -- Iterate over all OBMol object created for all molecules

print("")
print("Reading bonds, angles and torsions for all OBmol objects")


# -- Loop over all OBmol objects created for all molecules
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
        valencedata = str(OBmolatom.GetValence())
        atomdata = str(OBmolatom.GetType())

        if "N" in atomdata:

            number_of_neighbourH = 0

            for neighbour_atom in ob.OBAtomAtomIter(OBmolatom):
                neighbour_atomtype = str(neighbour_atom.GetType())

                if neighbour_atomtype[0] == "H":
                    number_of_neighbourH += 1

            if number_of_neighbourH > 0:
                MolNHatomData["NH" + str(number_of_neighbourH)] += 1

    # print(MolNHatomData)

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
            C_coordno = OBmol.GetAtom(OBmolangle[0] + 1).GetValence()  # CHANGED
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


"""

Feature generation step 2 summary:
The first step is to find all maxima and minima on KDE plots for all distribution.
BondFeaturesDef etc are dictionaries containing KDE maximas and KDE minimas.
These dictionaries are also stored as pickle files
    
"""


# -- Define dictionaries to generate features matrix X
BondFeaturesDef = dict()
AngleFeaturesDef = dict()
DihedralFeaturesDef = dict()
HbondFeaturesDef = dict()


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
            "./" + plot_location + "/HbondPlots/",
            str(key),
            XHbondkdewidth,
        )
    else:
        Hbond_maximas, Hbond_minimas = SingleDataDist(value, "./HbondPlots/", str(key))

    # print('Hbond_KDEmaxias: ',Hbond_KDEmaximas)

    HbondFeaturesDef[str(key)] = [Hbond_maximas, Hbond_minimas]

print("")
print("HbondFeaturesDef:")
print(HbondFeaturesDef)

print("")
print("Following bonds have been detected:")
print(list(BondLengths.keys()))

# -- Plot all observed bond distributions by looping over all bond types
for key, value in BondLengths.items():

    print(str(key) + ": " + str(len(value)) + " number of bonds")

    if len(value) > 1 and "H" in str(key):  # set different KDE width for XH bonds
        bond_KDEmaximas, bond_maximas, bond_minimas = FindKDEMax(
            value,
            "XHbond",
            "./" + plot_location + "/BondPlots/",
            str(key),
            XHbondkdewidth,
        )
    elif len(value) > 1 and "H" not in str(key):  # normal KDE width for all other bonds
        bond_KDEmaximas, bond_maximas, bond_minimas = FindKDEMax(
            value, "bond", "./" + plot_location + "/BondPlots/", str(key), bondkdewidth
        )
    else:
        bond_maximas, bond_minimas = SingleDataDist(
            value, "./" + plot_location + "/BondPlots/", str(key)
        )  # if only one maxima exists in the distribution

    # print('bond_KDEmaximas: ', bond_KDEmaximas)
    BondFeaturesDef[str(key)] = [bond_maximas, bond_minimas]


print("")
print("Following angles have been detected:")
print(list(AnglesTypes.keys()))

# -- Plot all observed angle distributions by looping over all angle gypes
for key, value in AnglesTypes.items():

    print(str(key) + ": " + str(len(value)) + " number of angles")

    if len(value) > 1:
        angle_KDEmaximas, angle_maximas, angle_minimas = FindKDEMax(
            value,
            "angle",
            "./" + plot_location + "/AnglePlots/",
            str(key),
            anglekdewidth,
        )
    else:
        angle_maximas, angle_minimas = SingleDataDist(
            value, "./" + plot_location + "/AnglePlots/", str(key)
        )

    # print('angle_KDEmaximas: ', angle_KDEmaximas)
    AngleFeaturesDef[str(key)] = [angle_maximas, angle_minimas]


print("")
print("Following dihedrals have been detected:")
print(list(DihedralTypes.keys()))

# -- Plot all observed dihedral distributions by looping over all dihedral types
for key, value in DihedralTypes.items():

    print(str(key) + ": " + str(len(value)) + " number of dihedrals")

    if len(value) > 1:
        dihedral_KDEmaximas, dihedral_maximas, dihedral_minimas = FindKDEMax(
            value,
            "dihedral",
            "./" + plot_location + "/DihedralPlots/",
            str(key),
            anglekdewidth,
        )
    else:
        dihedral_maximas, dihedral_minimas = SingleDataDist(
            value, "./" + plot_location + "/DihedralPlots/", str(key)
        )

    # print('dihedral_KDEmaximas: ', dihedral_KDEmaximas)
    DihedralFeaturesDef[str(key)] = [dihedral_maximas, dihedral_minimas]


print("Max_Atomtypes: ", Max_Atomtypes)

print("")
print("Saving features dictionaries - use them to generate features on a new dataset")

BondFeaturesDefFile = open(
    "./" + plot_location + "/" + plot_location + "_BondFeaturesDef.plk", "wb"
)
pickle.dump(BondFeaturesDef, BondFeaturesDefFile)
BondFeaturesDefFile.close()
print("Saved BondFeaturesDef")
print(BondFeaturesDef)

AngleFeaturesDefFile = open(
    "./" + plot_location + "/" + plot_location + "_AngleFeaturesDef.plk", "wb"
)
pickle.dump(AngleFeaturesDef, AngleFeaturesDefFile)
AngleFeaturesDefFile.close()
print("Saved AngleFeaturesDef")
print(AngleFeaturesDef)

DihedralFeaturesDefFile = open(
    "./" + plot_location + "/" + plot_location + "_DihedralFeaturesDef.plk", "wb"
)
pickle.dump(DihedralFeaturesDef, DihedralFeaturesDefFile)
DihedralFeaturesDefFile.close()
print("Saved DihedralFeaturesDef")
print(DihedralFeaturesDef)

MaxAtomTypesDefFile = open(
    "./" + plot_location + "/" + plot_location + "_MaxAtomTypesDef.plk", "wb"
)
pickle.dump(Max_Atomtypes, MaxAtomTypesDefFile)
MaxAtomTypesDefFile.close()
print("Saved MaxAtomTypesDef")
print(Max_Atomtypes)

HbondFeaturesDefFile = open(
    "./" + plot_location + "/" + plot_location + "_HbondFeaturesDef.plk", "wb"
)
pickle.dump(HbondFeaturesDef, HbondFeaturesDefFile)
MaxAtomTypesDefFile.close()
print("Saved HbondFeaturesDef")
print(HbondFeaturesDef)


"""

Feature generation step 3 summary:
This step creates features matrix X. The maximas on KDE plots for bonds, angles, dihedrals etc becomes the feature components. 
The algorithm calculates number of times a particular feature component exists for each molecules, for example, how many 'CC' bonds does the molecule contain?
This is then added to the matrix X for each molecule.

"""

print("")
print("The algorithm will use the following weights:")
print("bondweight: ", atomweight)
print("angleweight: ", bondweight)

# Features vector
print("")
print("Generating features matrix X for all molecules")
print("")
X, read_X = GenerateFeatures(
    OBmol_data,
    BondFeaturesDef,
    AngleFeaturesDef,
    DihedralFeaturesDef,
    Max_Atomtypes,
    HbondFeaturesDef,
    filenames,
)

print("")
print("Matrix X preview:")
print(X)
print("")
print("Vector y preview:")
print(y)
print("")
print("Readable matrix X:")
print(read_X)


list_X = X.tolist()

first10file = open(
    "./"
    + plot_location
    + "/GDB"
    + str(Group_to_Use)
    + "_"
    + plot_location
    + "_"
    + VersionNo
    + "_f"
    + str(len(X[0]))
    + "_first10"
    + ".txt",
    "w",
)

for vector in list_X[0:sample_outputnumber]:
    string_vector = " ".join(str(component) for component in vector)
    first10file.write(str(list_X.index(vector) + 1) + "\n")
    first10file.write(first10names[list_X.index(vector)] + "\n")
    first10file.write(string_vector + "\n")

first10file.close()

readableXfile = open(
    "./"
    + plot_location
    + "/GDB"
    + str(Group_to_Use)
    + "_"
    + plot_location
    + "_"
    + VersionNo
    + "_f"
    + str(len(X[0]))
    + "_readableX"
    + ".txt",
    "w",
)
for index in range(len(read_X)):
    readableXfile.write(read_X[index] + " ")
readableXfile.close()


# -- Save key vectors

np.save(
    "./"
    + plot_location
    + "/GDB"
    + str(Group_to_Use)
    + "_"
    + plot_location
    + "_"
    + VersionNo
    + "_f"
    + str(len(X[0]))
    + "_X",
    X,
    allow_pickle=True,
)
np.save(
    "./"
    + plot_location
    + "/GDB"
    + str(Group_to_Use)
    + "_"
    + plot_location
    + "_"
    + VersionNo
    + "_f"
    + str(len(X[0]))
    + "_yG",
    y,
    allow_pickle=True,
)

print("")
print("There are " + str(len(X[0])) + " number of features")
print("")
print("Successfully created feature vector X and target vector y")
print("")
