#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for enumerating and generating React8 structures

Created on 2/10/2019

@author: ke291

"""
import glob
import sys
import os
import Analyse
#import xlsxwriter
import random

from rdkit.Chem import AllChem as Chem
import rdkit

MaxHeavyAtoms = 8
SkelRoot1 = '/scratch/ke291/GDB9/ch/desat/'
SkelRoot2 = '/scratch/ke291/GDB9/ch/graphs/'

OrigRoot = '/scratch/ke291/GDB9/GDB9from13/'
#OrigRoot = '/scratch/ke291/GDB9/GDB9orig/'

React8Root = '/scratch/ke291/GDB9React/'

GDB8inchis = []

# Read all smi files in in_folder, randomly select nmols number of smiles from the list
# Generate 3D geometries for these and write to the out_folder
def GenerateTestMols(in_folder, out_folder, nmols, suffix):

    #get file list
    smifilelist = glob.glob(in_folder + '*.smi')

    allsmiles = []

    for f in smifilelist:
        smifile = open(f, 'r')
        smidata = smifile.readlines()
        smifile.close()

        allsmiles.extend([x[:-1] for x in smidata])

    print(str(len(allsmiles)) + ' total number of smiles read')
    print('First smiles: ' + allsmiles[0])
    print('Last smiles: ' + allsmiles[0])

    #set seed explicitly for repeatability
    random.seed(30470)
    testsmiles = random.sample(allsmiles, nmols)

    print(testsmiles)

    for i, smiles in enumerate(testsmiles):
        Gen3D(smiles, out_folder + 'T' + suffix + '_{:04d}'.format(i))


def CountMols():

    allsmiles = ReadSkeletalSmiles()
    counts = [0 for x in range(MaxHeavyAtoms)]

    for smi in allsmiles:
        c = smi.count('C')
        counts[c-1] += 1

    print(counts)


def ReadSkeletalSmiles():

    #get file list

    smifilelist = []
    smifilelist.extend(glob.glob(SkelRoot1 + '*.smi'))
    smifilelist.extend(glob.glob(SkelRoot2 + '*.smi'))

    temp = []

    for f in smifilelist:
        if int(f.split('/')[-1].split('.')[0]) <= MaxHeavyAtoms:
            temp.append(f)

    smifilelist = temp
    allsmiles = []

    for f in smifilelist:
        smifile = open(f, 'r')
        smidata = smifile.readlines()
        smifile.close()

        print(f + ': ' + str(len(smidata)))
        allsmiles.extend([x[:-1] for x in smidata])

    return allsmiles


def Gen3D(smiles, outfile):

    #input is a txt file with inchi string written on first line

    #output is output path

    m = Chem.rdmolfiles.MolFromSmiles(smiles)

    m = Chem.AddHs(m, addCoords=True)

    Chem.EmbedMolecule(m)

    save3d = Chem.SDWriter(outfile + ".sdf")

    save3d.write(m)


def GetGDB8inchis(directory='/scratch/ke291/GDB9/GDB9orig/'):

    #smifilelist = sorted(glob.glob(OrigRoot + '*.smi'), key=lambda x: int(x.split('/')[-1][:-4]))
    smifilelist = glob.glob(directory + '*.smi')
    print(directory + '*.smi')
    print(smifilelist)
    inchis = []
    molwts = []
    #input is a txt file with inchi string written on first line

    #output is output path
    allsmiles = []

    for f in smifilelist:
        smifile = open(f, 'r')
        smidata = smifile.readlines()
        smifile.close()

        print(f + ': ' + str(len(smidata)))
        allsmiles.extend([x[:-1] for x in smidata])

    for smiles in allsmiles:
        m = Chem.rdmolfiles.MolFromSmiles(smiles)

        m = Chem.AddHs(m, addCoords=True)

        inchis.append(Chem.inchi.MolToInchi(m))
        molwts.append(rdkit.Chem.Descriptors.MolWt(m))

    print('Total number of inchis: ' + str(len(inchis)))
    print('Total size in bytes: ' + str(sys.getsizeof(inchis)))
    return inchis, allsmiles, molwts


def MatchCalc2GDBInchis():
    GDB8inchis = GetGDB8inchis()
    tempinchis = []
    for i in GDB8inchis:
        tmp = i.split('/')
        tempinchis.append('/'.join(tmp[:4]))
    GDB8inchis = tempinchis
    dirlist = sorted([x for x in os.listdir(React8Root) if os.path.isdir(React8Root + x)])
    print(React8Root+os.listdir(React8Root)[0])
    print(dirlist)

    CalcInchis = Analyse.GetInchifromGaus(React8Root + dirlist[1] + '/')

    RedCalcInchis = []

    for i in CalcInchis:
        tmp = i.split('/')
        RedCalcInchis.append('/'.join(tmp[:4]))
    nfound = 0
    for i in RedCalcInchis:
        if i in GDB8inchis:
            nfound += 1

    print('{:d}/{:d} ({:.1f}%) of calculated found in GDB8'.format(nfound, len(RedCalcInchis), nfound * 100 / len(RedCalcInchis)))

def Generate3DsdfsToRun():

    torun = GenerateToRunList()

    os.chdir(os.path.join(React8Root, 'GDB30'))
    for i, (smi, molwt) in enumerate(torun):
        Gen3D(smi, str(30000 + i))


def GenerateToRunList():
    # Get original inchis
    inchis13, smiles13, molwts13 = GetGDB8inchis('/scratch/ke291/GDB9/GDB9orig/')
    inchis17, smiles17, molwts17 = GetGDB8inchis('/scratch/ke291/GDB9/GDB9from17/')

    dirlist = sorted([x for x in os.listdir(React8Root) if os.path.isdir(React8Root + x)])
    print(React8Root + os.listdir(React8Root)[0])
    print(dirlist)
    calcinchis = []
    calcsmiles = []
    calcoutfiles = []

    for dir in dirlist:
        tmpinchis, tmpsmiles, tmpoutfiles = Analyse.GetInchifromGaus(React8Root + dir + '/')
        calcinchis.extend(tmpinchis)
        calcsmiles.extend(tmpsmiles)
        calcoutfiles.extend(tmpoutfiles)

    redcalcinchis = []

    for i in calcinchis:
        tmp = i.split('/')
        redcalcinchis.append('/'.join(tmp[:4]))

    f = open('ToRun.smi', 'w')
    torun = []
    print('Picking the uncalculated molecules')
    for ginchi, gsmiles, molwt in zip(inchis13, smiles13, molwts13):
        if (ginchi not in redcalcinchis) and OnlyCHNO(gsmiles):
            torun.append([gsmiles,molwt])

    for ginchi, gsmiles, molwt in zip(inchis17, smiles17, molwts17):
        if (ginchi not in redcalcinchis) and (ginchi not in inchis13) and OnlyCHNO(gsmiles):
            torun.append([gsmiles, molwt])

    print('Writing the runlist')
    torun = sorted(torun, key= lambda x: x[1])
    for smi, molwt in torun:
        f.write(smi + '\n')

    #Filter out non-CNO elements containing compounds

    f.close()
    #return torun


def OnlyCHNO(smiles):

    CHNO = [6, 1, 7, 8]
    for atom in rdkit.Chem.rdmolfiles.MolFromSmiles(smiles).GetAtoms():
        if atom.GetAtomicNum() not in CHNO:
            return False
    return True


def MakeMatchingTable(datafile):
    f = open(datafile, 'w')
    #ID - row number, GDB8 compound number
    #smiles
    #inchi

    # Get original inchis
    inchis, smiles = GetGDB8inchis()

    dirlist = sorted([x for x in os.listdir(React8Root) if os.path.isdir(React8Root + x)])
    print(React8Root + os.listdir(React8Root)[0])
    print(dirlist)

    calcinchis, calcsmiles, calcoutfiles = Analyse.GetInchifromGaus(React8Root + dirlist[0] + '/')
    redcalcinchis = []

    for i in calcinchis:
        tmp = i.split('/')
        redcalcinchis.append('/'.join(tmp[:4]))

    tmp = zip(calcoutfiles, calcsmiles, redcalcinchis)
    tmp = sorted(tmp, key = lambda x: int(x[0].split('/')[-1][:-7]))
    lines = []
    for id, smile, inchi, calc in zip(range(len(smiles)), smiles, inchis, tmp):
        f.write(';'.join([str(id), smile, inchi] + [''] + list(calc)) + '\n')
    f.close()

    print(calcoutfiles)


def MakeMatchingXLSX(tablename):

    workbook = xlsxwriter.Workbook(tablename + '.xlsx')
    worksheet = workbook.add_worksheet()

    # ID - row number, GDB8 compound number
    # smiles
    # inchi

    # Get original inchis
    #gdbinchis, gdbsmiles = GetGDB8inchis()
    #gdbinchis, gdbsmiles, molwts13 = GetGDB8inchis('/scratch/ke291/GDB9/GDB9orig/')
    gdbinchis, gdbsmiles, molwts17 = GetGDB8inchis('/scratch/ke291/GDB9/GDB9from17/')

    #dirlist = sorted([x for x in os.listdir(React8Root) if os.path.isdir(React8Root + x)])
    dirlist = ['GDB1', 'GDB10', 'GDB11', 'GDB12', 'GDB13', 'GDB14', 'GDB15', 'GDB16', 'GDB17', 'GDB18', 'GDB19', 'GDB2', 'GDB20',
     'GDB21', 'GDB22', 'GDB23', 'GDB3', 'GDB4', 'GDB5', 'GDB6', 'GDB7', 'GDB8', 'GDB9']
    print(React8Root + os.listdir(React8Root)[0])
    print(dirlist)
    calcinchis = []
    calcsmiles = []
    calcoutfiles = []

    for dir in dirlist:
        tmpinchis, tmpsmiles, tmpoutfiles = Analyse.GetInchifromGaus(React8Root + dir + '/')
        calcinchis.extend(tmpinchis)
        calcsmiles.extend(tmpsmiles)
        calcoutfiles.extend(tmpoutfiles)

    redcalcinchis = []

    for i in calcinchis:
        tmp = i.split('/')
        redcalcinchis.append('/'.join(tmp[:4]))

    cell_format = workbook.add_format()

    cell_format.set_pattern(1)  # This is optional when using a solid fill.
    cell_format.set_bg_color('#88CC88')

    tmp = zip(calcoutfiles, calcsmiles, redcalcinchis)
    tmp = sorted(tmp, key=lambda x: int(x[0].split('/')[-1][:-7]))

    for id, smile, inchi, in zip(range(len(gdbsmiles)), gdbsmiles, gdbinchis):
        if inchi in redcalcinchis:
            worksheet.write_row(id, 0, [id, smile, inchi], cell_format=cell_format)
        else:
            worksheet.write_row(id, 0, [id, smile, inchi])

    for id, calc in enumerate(tmp):
        if calc[2] in gdbinchis:
            worksheet.write_row(id, 5, list(calc), cell_format=cell_format)
        else:
            worksheet.write_row(id, 5, list(calc))

    workbook.close()

    print(calcoutfiles)