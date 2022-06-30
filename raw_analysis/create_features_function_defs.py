"""
Function definitions for manipulating the molecules defined in Database.py
"""

from pathlib import Path
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from openbabel import openbabel as ob
from scipy.stats import gaussian_kde

from create_features_config import (
    scaling_factor,
    MaxBondLength,
    PTable,
    bondweight,
    atomweight,
    fileswanted,
)


##############################################################################
# Functions
##############################################################################


def AtomTypeNumbers(GDBclass) -> dict:
    """
    Returns a dictionary of maximum number of atoms of each element type

    """
    print("")
    print("Calculating maximum number of atoms of each element type")

    element_type = {}

    for index in range(1, len(GDBclass)):
        file_dict = Counter(GDBclass[index].atoms)

        # check whether the element type in molecule exists in overall element type dictionary
        for key in file_dict:

            if key in element_type:
                if file_dict[str(key)] > element_type[str(key)]:
                    element_type[str(key)] = file_dict[str(key)]
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


def FindKDEMax(distribution_data, distribution_type, location: Path, name, kde_width):
    """MODIFIED TO NOT SAVE FILES"""
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

    # Ensure the target dir exists
    location.mkdir(parents=True, exist_ok=True)

    plt.legend()
    plt.savefig(location / (name + "_kde_plot" + ".png"), format="png", dpi=500)
    plt.close()

    np.save(location / name, distribution_data, allow_pickle=True)

    x_maximas_outputfile = open(location / (name + "_kde_maximas" + ".txt"), "w")
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

            print('atom_key: ', atom_key)

            maxatomkeys = list(maxatomtypes.keys())

            if atom_key in MolAtomKeys:
                feat_vec[maxatomkeys.index(atom_key)] = (
                    MolAtomDict[atom_key] * atomweight
                )

        Mol_X += feat_vec # [714:718]

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
