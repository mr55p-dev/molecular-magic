#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for generating indexes for the React8 database,
searching the database

Created on 5/01/2021

@author: ke291

"""

import os
import psutil
import glob
import pickle
import numpy as np
import tarfile
import shutil
import MultiMol
import Monitor
import gc


from openbabel import openbabel as ob
from openbabel.openbabel import OBMol, OBAtom, OBConversion, OBMolBondIter, OBMolAtomIter
from rdkit.Chem import AllChem as Chem
from rdkit import RDLogger

lg = RDLogger.logger()

lg.setLevel(RDLogger.ERROR)

MEMLIM = 10.0    # Memory limit in GB, some functions will check memory and exit, if this is exceeded

GDBroot = '/media/kristaps/SCRATCH/Data/GDB9React_Nov2020'

folderindex = [['GDB1', 1, 1000], ['GDB2', 1001, 2000], ['GDB3', 2001, 3000], ['GDB4', 3001, 4000], ['GDB5', 4001, 5000],
        ['GDB6', 5001, 6000], ['GDB7', 6001, 7000], ['GDB8', 7001, 8000], ['GDB9', 8001, 9000], ['GDB10', 9001, 10000],
        ['GDB11', 10001, 11000], ['GDB12', 11001, 12000], ['GDB13', 12001, 13000], ['GDB14', 13001, 14000], ['GDB15', 14001, 15000],
        ['GDB16', 15001, 16000], ['GDB17', 16001, 17000], ['GDB18', 17001, 18000], ['GDB19', 18001, 19000], ['GDB20', 19001, 20000],
        ['GDB21', 20001, 21000], ['GDB22', 21001, 22000], ['GDB23', 22001, 23000], ['GDB30', 30000, 30999],
        ['GDB31', 31000, 31999], ['GDB32', 32000, 32999], ['GDB33', 33000, 33999], ['GDB34', 34000, 34999], ['GDB35', 35000, 35999],
        ['GDB36', 36000, 36999], ['GDB37', 37000, 37999], ['GDB38', 38000, 38999], ['GDB39', 39000, 39999], ['GDB40', 40000, 40999],
        ['GDB41', 41000, 41999], ['GDB42', 42000, 42999], ['GDB43', 43000, 43999], ['GDB44', 44000, 44999], ['GDB45', 45000, 45999],
        ['GDB46', 46000, 46999], ['GDB47', 47000, 47999], ['GDB48', 48000, 48999], ['GDB49', 49000, 49999], ['GDB50', 50000, 50999],
        ['GDB51', 51000, 51999], ['GDB52', 52000, 52999], ['GDB53', 53000, 53999], ['GDB54', 54000, 54999], ['GDB55', 55000, 55999],
        ['GDB56', 56000, 56999], ['GDB57', 57000, 57999], ['GDB58', 58000, 58999], ['GDB59', 59000, 59999], ['GDB60', 60000, 60999],
        ['GDB61', 61000, 61999], ['GDB62', 62000, 62999], ['GDB63', 63000, 63999], ['GDB64', 64000, 64999], ['GDB65', 65000, 65999],
        ['GDB66', 66000, 66999], ['GDB67', 67000, 67999], ['GDB68', 68000, 68999], ['GDB69', 69000, 69095]]


# Function for generating the fingerprint index, useful for substructure search
def GenerateFPIndex():
    fpindex = []
    limit = 100
    defmol = Chem.MolFromSmiles('C')
    #deffp = Chem.RDKFingerprint(defmol)
    deffp = Chem.PatternFingerprint(defmol)

    for pklfile in folderindex:
        process = psutil.Process()
        print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')
        print('Analyzing ' + pklfile[0] + '.pkl')
        pkldata = LoadFolderFromPickle(os.path.join(GDBroot, pklfile[0] + '.pkl'))
        for i, mol in enumerate(pkldata):
            #Generate fingerprint
            #print(mol.geomfilename.split('/')[-1][:-5])
            charge = mol.charge
            mult = mol.multiplicity
            if mol.converged == False:
                continue
            inchi = MolToInchi(mol.atoms, mol.coords, mol.charge, mol.multiplicity).rstrip()
            #print(inchi)
            rdmol = Chem.MolFromInchi(inchi, sanitize=False, removeHs=False)
            if rdmol == None:
                print('No go')
                fpindex.append([deffp, pklfile[0], i, 0, 1])
                continue
            #fp = Chem.RDKFingerprint(rdmol)
            fp = Chem.PatternFingerprint(rdmol)
            #print(Chem.MolToSmiles(rdmol))
            #print(DataStructs.cDataStructs.BitVectToText(fp))
            fpindex.append([fp, pklfile[0], i, charge, mult])

    print('Writing FPIndex.pkl')
    #with open(os.path.join(Database.GDBroot, "FPIndex.pkl"), "wb") as write_file:
    with open(os.path.join(GDBroot, "PatternFPIndex.pkl"), "wb") as write_file:
        pickle.dump(fpindex, write_file)


# Function for generating the inchi index by pkl file
def GeneratePklInchiIndex():

    pklinchidict = {}

    for pklfile in folderindex:
        process = psutil.Process()
        print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')
        print('Analyzing ' + pklfile[0] + '.pkl')
        pkldata = LoadFolderFromPickle(os.path.join(GDBroot, pklfile[0] + '.pkl'))
        pklinchis = []
        for mol in pkldata:
            inchi = MolToInchi(mol.atoms, mol.coords, mol.charge, mol.multiplicity).rstrip()
            pklinchis.append(inchi)
        pklinchidict[pklfile[0]] = pklinchis

    print('Writing PklInchiIndex.pkl')
    with open(os.path.join(GDBroot, "PklInchiIndex.pkl"), "wb") as write_file:
        pickle.dump(pklinchidict, write_file)


# Function for generating the stoichiometry index
def GenerateStoichInchiIndex():

    knownstoichs = []
    stoichcounts = []
    inchiindex = []

    for pklfile in folderindex:
        process = psutil.Process()
        print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')
        print('Analyzing ' + pklfile[0] + '.pkl')
        pkldata = LoadFolderFromPickle(os.path.join(GDBroot, pklfile[0] + '.pkl'))
        for i, mol in enumerate(pkldata):
            Ccount = mol.atoms.count('C')
            Hcount = mol.atoms.count('H')
            Ocount = mol.atoms.count('O')
            Ncount = mol.atoms.count('N')
            charge = mol.charge
            mult = mol.multiplicity
            molname = mol.geomfilename.split('/')[-1][:-5]
            stoich = [Ccount, Hcount, Ocount, Ncount, charge, mult]
            inchi = MolToInchi(mol.atoms, mol.coords, mol.charge, mol.multiplicity).rstrip()
            if stoich in knownstoichs:
                stid = knownstoichs.index(stoich)
                stoichcounts[stid] += 1
                inchiindex[stid].append([inchi, pklfile[0], i, molname])
            else:
                knownstoichs.append(stoich)
                stoichcounts.append(1)
                inchiindex.append([])
                inchiindex[-1].append([inchi, pklfile[0], i, molname])

    print(len(knownstoichs))
    print(max(stoichcounts))

    print('Writing StoichInchiIndex.pkl')
    with open(os.path.join(GDBroot, "StoichInchiIndex.pkl"), "wb") as write_file:
        pickle.dump((knownstoichs, inchiindex), write_file)

    return knownstoichs, stoichcounts


def SearchSMARTS(smarts, charge, mult, mods):
    patternmol = Chem.MolFromSmarts(smarts)
    return SearchPattern(patternmol, charge, mult, mods)


def SearchSubStructure(smiles, charge, mult, mods):
    patternmol = Chem.MolFromSmiles(smiles)
    return SearchPattern(patternmol, charge, mult, mods)


def SearchPattern(patternmol, charge, mult, mods):
    startime = time.time()
    resultlimit = 1000
    hits = 0

    # patternmol = Chem.AddHs(patternmol)
    # fpindex = LoadFolderFromPickle(os.path.join(GDBroot, "FPIndex.pkl"))
    gc.disable()
    fpindex = LoadFolderFromPickle(os.path.join(GDBroot, "PatternFPIndex.pkl"))
    gc.enable()
    print('FPIndex.pkl loaded')
    print('Mods: ' + str(mods))
    # targetfp = Chem.RDKFingerprint(patternmol)
    if patternmol != None:
        targetfp = Chem.PatternFingerprint(patternmol)
    else:
        return
    candidates = []

    for [fp, pklfile, i, molch, molmult] in fpindex:
        if (DataStructs.cDataStructs.AllProbeBitsMatch(targetfp, fp) == True) and \
                (molch in charge) and (molmult in mult):
            candidates.append([pklfile, i])
    print(str(len(candidates)) + ' candidates from fingerprints found')

    gc.disable()
    pklinchidict = LoadFolderFromPickle(os.path.join(GDBroot, "PklInchiIndex.pkl"))
    gc.enable()
    print('PklInchiIndex.pkl loaded')
    newcandidates = []
    for pklfile, i in candidates:
        candmol = Chem.MolFromInchi(pklinchidict[pklfile][i], sanitize=False, removeHs=False)
        if candmol == None:
            print("RDMol generation failed, molecule " + str(i) + " from " + pklfile + " skipped")
            continue
        if candmol.HasSubstructMatch(patternmol):
            newcandidates.append([pklfile, i])
    candidates = newcandidates

    print(str(len(candidates)) + ' candidates from pkl inchi index found')

    results = []
    currbasename = ''
    currpkl = ''
    for pklfile, i in candidates:
        if pklfile != currpkl:
            gc.disable()
            pkldata = LoadFolderFromPickle(os.path.join(GDBroot, pklfile + '.pkl'))
            gc.enable()
            print('Searching in ' + pklfile + '.pkl')
            currpkl = pklfile

        if (pkldata[i].charge in charge) and (pkldata[i].multiplicity in mult) and (pkldata[i].geomfilename[-7] in mods):
            basename = pkldata[i].geomfilename.split('/')[-1][:-7]
            if basename != currbasename:
                currbasename = basename
                results.append([])
            results[-1].append(pkldata[i])
            hits += 1
    print('Search done in {:.1f} seconds, {:d} hits found'.format(time.time() - startime, hits))
    for i in range(len(results)):
        for j in range(len(results[i])):
            results[i][j].molname = results[i][j].geomfilename.split('/')[-1][:-5]

    return results


def MolToInchi(atoms, coords, charge, multiplicity):

    obmol = ob.OBMol()
    for anum, acoords in zip(atoms, coords):
        atom = ob.OBAtom()
        atom.thisown = False
        atom.SetAtomicNum(GetAtomNum(anum))
        atom.SetVector(acoords[0], acoords[1], acoords[2])
        obmol.AddAtom(atom)

    # Restore the bonds
    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()
    obmol.SetTotalCharge(charge)
    obmol.SetTotalSpinMultiplicity(multiplicity)

    obConversion = ob.OBConversion()
    obConversion.SetOutFormat("inchi")

    inchistring = obConversion.WriteString(obmol)

    return inchistring


def LoadFolderFromPickle(pklfile):
    with open(pklfile, "rb") as read_file:
        return pickle.load(read_file)


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