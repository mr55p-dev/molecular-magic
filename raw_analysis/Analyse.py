#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for analysing GDB9 data

Created on 12/07/2019

@author: ke291

"""

import os
import glob
from openbabel import *
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit import RDLogger

import Monitor
import subprocess


def GetInchifromGaus(GOutpFileRoot):

    outfilelist = []
    print(GOutpFileRoot + '*A*a.out')
    outfilelist.extend(glob.glob(GOutpFileRoot + '*A*a.out'))
    inchis = []
    smiles = []

    for GOutpFile in outfilelist:
        print(GOutpFile)
        atoms, coords, charge, multiplicity, steps, scfenergy = ReadGausOutp(GOutpFile, -1)
        #print("Charge = " + str(charge) + ', Multiplicity = ' + str(multiplicity))
        obmol = BuildOBMol(atoms, coords)
        obmol.SetTotalCharge(charge)
        obmol.SetTotalSpinMultiplicity(multiplicity)

        obConversion = OBConversion()
        obConversion.SetOutFormat("mol")

        MolSDFstring = obConversion.WriteString(obmol)

        rdmol = Chem.MolFromMolBlock(MolSDFstring, sanitize = False)
        #rdmol.UpdatePropertyCache()
        #print(rdmol)

        """if (charge != 0) or (multiplicity != 1):
            atomid = GetOBOffValenceAtom(obmol)
            if atomid != -1:
                FixRDChargeSpin(rdmol, atomid, charge, multiplicity)
        """
        #Chem.AssignRadicals(rdmol)
        Chem.RemoveHs(rdmol, sanitize = False)
        #print('NRadicalElectrons: ' + str(Descriptors.NumRadicalElectrons(rdmol)))
        #Chem.Compute2DCoords(rdmol)
        #Chem.SanitizeMol(rdmol, )
        try:
            newinchi = Chem.inchi.MolToInchi(rdmol, logLevel=None, treatWarningAsError=False)
        except ValueError:
            newinchi = ''
        inchis.append(newinchi)
        smiles.append(Chem.MolToSmiles(rdmol, allHsExplicit = False))

    return inchis, smiles, outfilelist

def GetReact8Stats():

    # Get GDB folders
    GDBroot = '/scratch/ke291/GDB9React'
    dirlist = os.listdir(GDBroot)
    GDBfolders = []
    for name in dirlist:
        if os.path.isdir(os.path.join(GDBroot, name)):
            GDBfolders.append(os.path.join(GDBroot, name))

    for folder in GDBfolders:
        print('Entering folder ' + str(folder))

    Inputs, Outputs, Completed, Converged = Monitor.CheckGDB9()
    pass

def ReadGausOutp(GOutpFile, gindex=-1):

    gausfile = open(GOutpFile, 'r')
    GOutp = gausfile.readlines()
    gausfile.close()

    atoms = []
    coords = []
    chindex = -1
    stindex = -1
    scfindex = -1

    # Find the last geometry section
    for index in range(len(GOutp)):
        if ('Input orientation:' in GOutp[index]) or ("Standard orientation:" in GOutp[index]):
            gindex = index + 5
        if ('Multiplicity' in GOutp[index]):
            chindex = index
        if ('Step number' in GOutp[index]):
            stindex = index
        if ('SCF Done' in GOutp[index]):
            scfindex = index

    if chindex < 0:
        print('Error: No charge and/or multiplicity found in file ' + GOutpFile)
        return [], [], -1000, -1000, 0, 0

    if gindex < 0:
        print('Error: No geometry found in file ' + GOutpFile)
        return [], [], -1000, -1000, 0, 0

    # Read geometry
    for line in GOutp[gindex:]:
        if '--------------' in line:
            break
        else:
            data = [_f for _f in line[:-1].split(' ') if _f]
            atoms.append(GetAtomSymbol(int(data[1])))
            coords.append([float(x) for x in data[3:]])

    # Read charge and multiplicity
    data = [_f for _f in GOutp[chindex][:-1].split(' ') if _f]
    charge = int(data[2])
    multiplicity = int(data[5])

    # Read Step count
    if stindex > 0:
        data = [_f for _f in GOutp[stindex][:-1].split(' ') if _f]
        steps = int(data[2])
    else:
        steps = 0

    # Read SCF energy
    if scfindex > 0:
        data = [_f for _f in GOutp[scfindex][:-1].split(' ') if _f]
        scfenergy = float(data[4])
    else:
        scfenergy = 0

    return atoms, coords, charge, multiplicity, steps, scfenergy


def RadicalTest():
    mols = []
    mols.append(Chem.MolFromSmiles('C1=CN[C]=C1', sanitize = False))
    Chem.AssignRadicals(mols[-1])
    mols[-1].UpdatePropertyCache()
    print('NRadicalElectrons: ' + str(Descriptors.NumRadicalElectrons(mols[-1])))
    mols.append(Chem.MolFromSmiles('C1=C[O]C=C1'))
    mols[-1].UpdatePropertyCache()
    print('NRadicalElectrons: ' + str(Descriptors.NumRadicalElectrons(mols[-1])))
    mols.append(Chem.MolFromSmiles('CC[O]CCC'))
    mols[-1].UpdatePropertyCache()
    print('NRadicalElectrons: ' + str(Descriptors.NumRadicalElectrons(mols[-1])))
    #Draw.MolToFile(mols[0], 'img/_test.png')
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 300))
    img.save('img/_test.png')


def FixRDChargeSpin(rdmol, atomid, charge, spin):

    atom = rdmol.GetAtomWithIdx(atomid-1)
    atom.SetFormalCharge(charge)
    if spin == 2:
        atom.SetNumRadicalElectrons(1)
    if spin == 1:
        atom.SetNumRadicalElectrons(0)


def GetOBOffValenceAtom(obmol):

    for atom in OBMolAtomIter(obmol):
        idx = atom.GetIdx()

        valence = 0
        for bond in OBAtomBondIter(atom):
            valence += bond.GetBO()

        if (atom.GetAtomicNum() == 6) and (valence != 4):
            return idx
        if (atom.GetAtomicNum() == 7) and (valence != 3):
            return idx
        if (atom.GetAtomicNum() == 8) and (valence != 2):
            return idx

    return -1


def FixMolString(molstr, charge, multiplicity):
    if charge == 0:
        return molstr
    else:
        stringlines = molstr.split('\n')
        data = stringlines[-3].split('  ')
        stringlines[-3] = 'M  CHG  ' + data[2] + '  ' + data[3] + '  ' + '{0:2}'.format(charge)
        return '\n'.join(stringlines)


def BuildOBMol(atoms, coords):

    mol = OBMol()
    for anum, acoords in zip(atoms, coords):
        atom = OBAtom()
        atom.thisown = False
        atom.SetAtomicNum(GetAtomNum(anum))
        atom.SetVector(float(acoords[0]),float(acoords[1]), float(acoords[2]))
        mol.AddAtom(atom)

    #Restore the bonds
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()

    #mol.Kekulize()

    return mol


def GetGDB9StepCounts():

    #Get GDB folders
    GDBroot = '/scratch/ke291/GDB9React'
    dirlist = os.listdir(GDBroot)
    GDBfolders = []
    for name in dirlist:
        if os.path.isdir(os.path.join(GDBroot, name)):
            GDBfolders.append(os.path.join(GDBroot, name))

    StepCounts = []
    Nfiles = 0

    for folder in GDBfolders:
        outfilelist = glob.glob(folder + '/*a.out')
        for f in outfilelist:
            StepCounts.append(GetStepCount(f))
            Nfiles += 1
            if Nfiles%1000 == 0:
                print('{:d} files analysed'.format(Nfiles))

    return StepCounts


def StepHistogram():

    import matplotlib.pyplot as plt

    # Generate a normal distribution, center at x=0 and y=5
    StepCounts = GetGDB9StepCounts()

    # We can set the number of bins with the `bins` kwarg
    plt.hist(StepCounts, bins=50)


    plt.show()

    return StepCounts


PTable = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', \
          'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', \
          'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', \
          'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', \
          'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', \
          'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', \
          'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']


def GetAtomSymbol(AtomNum):

    if AtomNum > 0 and AtomNum < len(PTable):
        return PTable[AtomNum-1]
    else:
        print("No such element with atomic number " + str(AtomNum))
        return 0


def GetAtomNum(AtomSymbol):

    if AtomSymbol in PTable:
        return PTable.index(AtomSymbol)+1
    else:
        print("No such element with symbol " + str(AtomSymbol))
        return 0