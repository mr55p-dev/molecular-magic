#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for reading Gaussian files and generating the React8 database,
Also doing data cleaning and deduplication, extracting calculation error stats

Created on 2/10/2019

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

class GDB9molecule:
    """Class to represent a molecule and its properties, as calculated through Gaussian16"""
    def __init__(self, gausfile = '', strings = True):
        if gausfile == '':
            self.inputfilename = ''     # gaussian input filename containing initial geometry
            self.geomfilename = ''      # gaussian filename containing final geometry
            self.noptsteps = 0          # total number of optimization steps taken
            self.freqfilename = ''      # gaussian filename containing frequencies
            self.molid = 0              # molecular id in this database
            self.inchi = ''
            self.inchikey = ''
            self.smiles = ''
            self.charge = -1000
            self.multiplicity = -1000
            self.natoms = 0             # number of atoms
            self.nhatoms = 0            # number of heavy atoms
            self.atoms = []             # list of atom elements
            self.mulcharges = []        # partial charges for all atoms
            self.coords = []            # list of cartesian x,y,z coordinates for each atom
            self.startcoords = []
            self.bonds = []             # list of atom number pairs
            self.bondorders = []        # list of bond orders, corresponding to bonds previously
            self.fragments = 0          # number of unconnected fragments found, usually 1
            self.topologychanged = False   # has the molecular topology changed during optimisation
            self.cleantermination = False # has the gaussian job terminated cleanly
            self.converged = False      # has the molecule converged
            self.unstable = False       # True or False
            self.unstableto = None      # if its unstable during optimization, what molecule does it optimize to
            self.frequencies = []       # list of frequencies for this molecule
            self.imaginaryfreqs = False # are there imaginary frequencies indicating an unstable geometry?
            self.SCFenergy = 0          # dft electronic energy in hartrees
            self.ZPVE = 0               # ZPVE - Zero-point correction in hartrees
            self.E0 = 0                 # E0 - Sum of electronic and zero-point Energies in hartrees
            self.E0_298 = 0             # E0_298 - Thermal correction to Energy in hartrees
            self.E298 = 0               # E298 - Sum of electronic and thermal Energies in hartrees
            self.H298 = 0               # H298 - Sum of electronic and thermal Enthalpies in hartrees
            self.G298 = 0               # G298 - Sum of electronic and thermal Free Energies, total free energy in hartrees
            self.CorrectedFreeEnergy = 0  # corrected free energy in hartrees
            self.reactions = []         # elementary reaction connecting to a different molecule
        else:
            self.MolFromGaus(gausfile, strings)


    def MolFromGaus(self, gausfile, strings = True):
        """Method to parse a set of Gaussian16 output files

        Checke frequencies, then reads the final geometry/energy calculations, then thermal calculations"""

        self.inputfilename = gausfile
        self.geomfilename = gausfile

        # Check the frequencies
        if os.path.exists(gausfile[:-5] + 'f.out'):
            self.freqfilename = gausfile[:-5] + 'f.out'
            self.frequencies = self.ReadFrequencies(self.freqfilename)
            if all([freq>=0 for freq in self.frequencies]):
                self.imaginaryfreqs = False
            else:
                self.imaginaryfreqs = True
                print('Imaginary freqs in ' + self.freqfilename)
        else:
            self.freqfilename = ''
            self.frequencies = []
            self.imaginaryfreqs = False

        # Open the final optimized geometry and energy
        self.ReadGausOutp(gausfile, -1)

        # Open the thermal calculations
        self.ReadThermo(gausfile)

        # Count the atoms and heavy atoms
        self.natoms = len(self.atoms)
        self.nhatoms = self.atoms.count('C') + self.atoms.count('N') + self.atoms.count('O')

        if self.natoms <= 0:
            # This should probably raise an exception?!
            return self


        # Build an OBMol representation
        obmol = self.BuildOBMol(self.atoms, self.coords)
        obmol.SetTotalCharge(self.charge)
        obmol.SetTotalSpinMultiplicity(self.multiplicity)

        self.bonds = []
        self.bondorders = []

        for bond in OBMolBondIter(obmol):
            # Create a list of every pairwise bond in the molecule (edge list)
            self.bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            # Keep track of the bond order also
            self.bondorders.append(bond.GetBondOrder())

        # Are we calculating SMILES strings also?
        if strings:
            obConversion = OBConversion()
            obConversion.SetOutFormat("mol")

            MolSDFstring = obConversion.WriteString(obmol)

            rdmol = Chem.MolFromMolBlock(MolSDFstring, sanitize=False)

            Chem.RemoveHs(rdmol, sanitize=False)

            try:
                self.inchi = Chem.inchi.MolToInchi(rdmol, logLevel=None, treatWarningAsError=False)
            except ValueError:
                self.inchi = ''

            self.smiles = Chem.MolToSmiles(rdmol, allHsExplicit=False)
        else:
            self.inchi = ''
            self.smiles = ''

            # Apparently we need to do this entire step again?
            obmol = self.BuildOBMol(self.atoms, self.coords)
            obmol.SetTotalCharge(self.charge)
            obmol.SetTotalSpinMultiplicity(self.multiplicity)

            self.bonds = []
            self.bondorders = []

            # Why do we do this twice?
            for bond in OBMolBondIter(obmol):
                self.bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                self.bondorders.append(bond.GetBondOrder())

        self.topologychanged = TopologyChanged(self)

        # Check if there are multiple fragments in the setting
        self.fragments = len(MultiMol.GetMolParts(obmol))
        #if self.fragments > 1:
        #    print('Multiple fragments: ' + self.geomfilename)


    def AddStrings(self):
        """Apparently not referenced, but it seems to set the SMILES attribute"""
        obmol = self.BuildOBMol(self.atoms, self.coords)
        obmol.SetTotalCharge(self.charge)
        obmol.SetTotalSpinMultiplicity(self.multiplicity)

        self.bonds = []
        self.bondorders = []

        for bond in OBMolBondIter(obmol):
            self.bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            self.bondorders.append(bond.GetBondOrder())

        obConversion = OBConversion()
        obConversion.SetOutFormat("mol")

        MolSDFstring = obConversion.WriteString(obmol)

        rdmol = Chem.MolFromMolBlock(MolSDFstring, sanitize=False)

        Chem.RemoveHs(rdmol, sanitize=False)

        try:
            self.inchi = Chem.inchi.MolToInchi(rdmol, logLevel=None, treatWarningAsError=False)
        except ValueError:
            self.inchi = ''

        self.smiles = Chem.MolToSmiles(rdmol, allHsExplicit=False)


    def ReadGausOutp(self, GOutpFile, geom=-1):
        """Parses the Gaussian16 text file for the energy and geometry"""
        gausfile = open(GOutpFile, 'r')
        GOutp = gausfile.readlines()
        gausfile.close()

        self.atoms = []
        self.startcoords = []
        self.coords = []
        self.error = ''
        self.cleantermination = False
        self.converged = False
        self.mulcharges = []  # partial charges for all atoms
        gindices = []       # list of geometry line indices
        chindex = -1        # index of line containing charge and multiplicity
        mulchindex = -1     # index of last line starting mulliken partial charge data
        stindex = -1        # index of last line containing step count
        scfindex = -1       # index of last line containing SCF energy

        # Find the last geometry section
        for index in range(len(GOutp)):
            if ('Input orientation:' in GOutp[index]) or ("Standard orientation:" in GOutp[index]):
                gindices.append(index + 5)
            if ('Multiplicity' in GOutp[index]):
                chindex = index
            if ('Mulliken charges:' in GOutp[index]):
                mulchindex = index
            if ('Step number' in GOutp[index]):
                stindex = index
            if ('SCF Done' in GOutp[index]):
                scfindex = index
            if ('cpu' in GOutp[index]):
                self.cleantermination = True
            if ('Stationary' in GOutp[index]):
                self.converged = True
            if ('Error termination' in GOutp[index]):
                self.error = GOutp[index]


        if chindex < 0:
            print('Error: No charge and/or multiplicity found in file ' + GOutpFile)
            self.atoms = []
            self.coords = []
            self.charge = -1000
            self.multiplicity = 1000
            self.noptsteps = 0
            self.SCFenergy = 0
            return 0

        if gindices == []:
            print('Error: No geometry found in file ' + GOutpFile)
            self.atoms = []
            self.coords = []
            self.startcoords = []
            self.charge = -1000
            self.multiplicity = 1000
            self.noptsteps = 0
            self.SCFenergy = 0
            return 0

        # Read start geometry
        for line in GOutp[gindices[0]:]:
            if '--------------' in line:
                break
            else:
                data = [_f for _f in line[:-1].split(' ') if _f]
                self.startcoords.append([float(x) for x in data[3:]])

        # Read end geometry
        for line in GOutp[gindices[-1]:]:
            if '--------------' in line:
                break
            else:
                data = [_f for _f in line[:-1].split(' ') if _f]
                self.atoms.append(GetAtomSymbol(int(data[1])))
                self.coords.append([float(x) for x in data[3:]])

        # Read Mulliken charges
        if mulchindex > 0:
            for line in GOutp[mulchindex+2:]:
                if 'Sum of Mulliken charges' in line:
                    break
                else:
                    data = [_f for _f in line[:-1].split(' ') if _f]
                    self.mulcharges.append(float(data[2]))

        # Read charge and multiplicity
        data = [_f for _f in GOutp[chindex][:-1].split(' ') if _f]
        self.charge = int(data[2])
        self.multiplicity = int(data[5])

        # Read Step count
        if stindex > 0:
            data = [_f for _f in GOutp[stindex][:-1].split(' ') if _f]
            self.noptsteps = int(data[2])
        else:
            self.noptsteps = 0

        # Read SCF energy
        if scfindex > 0:
            data = [_f for _f in GOutp[scfindex][:-1].split(' ') if _f]
            self.SCFenergy = float(data[4])
        else:
            self.SCFenergy = 0


    def BuildOBMol(self, atoms, coords):
        """Creates an OBMol representation of the atom

        Takes in a set of atoms and coordinates, and assigns them to a vector space
        OBMol then connects the atoms automagically and infers bond orders"""
        mol = OBMol()
        for anum, acoords in zip(atoms, coords):
            atom = OBAtom()
            atom.thisown = False
            atom.SetAtomicNum(GetAtomNum(anum))
            atom.SetVector(acoords[0], acoords[1], acoords[2])
            mol.AddAtom(atom)

        # Restore the bonds
        mol.ConnectTheDots()
        mol.PerceiveBondOrders()

        # mol.Kekulize()

        return mol


    def ReadFrequencies(self, GOutpFile):
        """Reads a gaussian16 Frequency calculation file"""
        gausfile = open(GOutpFile, 'r')
        GOutp = gausfile.readlines()
        gausfile.close()

        freqs = []
        for line in GOutp:
            if 'Frequencies --' in line:
                data = [_f for _f in line[17:-1].split(' ') if _f]
                freqs.extend([float(x) for x in data])

        return freqs

    def ReadThermo(self, GOutpFile):
        """Reads a Gaussian16 thermal calculation file"""
        # All quantities in hartrees/particle
        # ZPVE - Zero-point correction=
        self.ZPVE = 0
        # E0 - Sum of electronic and zero-point Energies=
        self.E0 = 0
        # E0_298 - Thermal correction to Energy=
        self.E0_298 = 0
        # E298 - Sum of electronic and thermal Energies=
        self.E298 = 0
        # H298 - Sum of electronic and thermal Enthalpies=
        self.H298 = 0
        # G298 - Sum of electronic and thermal Free Energies=
        self.G298 = 0

        if os.path.exists(GOutpFile[:-5] + 'f.out'):
            gausfile = open(GOutpFile[:-5] + 'f.out', 'r')
            GOutp = gausfile.readlines()
            gausfile.close()
        else:
            return 0

        for line in GOutp:
            if 'Zero-point correction=' in line:
                self.ZPVE = float([_f for _f in line[:-1].split(' ') if _f][-2])

            if 'Sum of electronic and zero-point Energies=' in line:
                self.E0 = float([_f for _f in line[:-1].split(' ') if _f][-1])

            if 'Thermal correction to Energy=' in line:
                self.E0_298 = float([_f for _f in line[:-1].split(' ') if _f][-1])

            if 'Sum of electronic and thermal Energies=' in line:
                self.E298 = float([_f for _f in line[:-1].split(' ') if _f][-1])

            if 'Sum of electronic and thermal Enthalpies=' in line:
                self.H298 = float([_f for _f in line[:-1].split(' ') if _f][-1])

            if 'Sum of electronic and thermal Free Energies=' in line:
                self.G298 = float([_f for _f in line[:-1].split(' ') if _f][-1])

        #print('self.ZPVE=' + str(self.ZPVE))
        #print('self.E0=' + str(self.E0))
        #print('self.E0_298=' + str(self.E0_298))
        #print('self.E298=' + str(self.E298))
        #print('self.H298=' + str(self.H298))
        #print('self.G298=' + str(self.G298))

def MolToInchi(atoms, coords, charge, multiplicity) -> str:
    """Convert a set of atoms, coordinates, charges and multiplicities into an InChi string"""
    obmol = OBMol()
    for anum, acoords in zip(atoms, coords):
        atom = OBAtom()
        atom.thisown = False
        atom.SetAtomicNum(GetAtomNum(anum))
        atom.SetVector(acoords[0], acoords[1], acoords[2])
        obmol.AddAtom(atom)

    # Restore the bonds
    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()
    obmol.SetTotalCharge(charge)
    obmol.SetTotalSpinMultiplicity(multiplicity)

    obConversion = OBConversion()
    obConversion.SetOutFormat("mol")

    MolSDFstring = obConversion.WriteString(obmol)

    rdmol = Chem.MolFromMolBlock(MolSDFstring, sanitize=False)

    Chem.RemoveHs(rdmol, sanitize=False)

    try:
        inchi = Chem.inchi.MolToInchi(rdmol, logLevel=None, treatWarningAsError=False)
    except ValueError:
        inchi = ''

    return inchi


def key(filename):
    """Macro to sort a list of filenames"""
    dict = 'ABCDEFGH'
    tmp = filename.split('/')[-1]
    tmp = tmp.replace(tmp[-7], str(dict.index(tmp[-7])))
    return int(tmp[:-5])


def GenerateDatabaseFromArchives(strings=True, archivelist = []):
    """Creates a database from a set of tar.gz archives"""
    # Get GDB archive list
    GDBroot = '/scratch/ke291/test/GDB9React'
    if archivelist == []:
        archivelist = glob.glob(os.path.join(GDBroot, '*.tar.gz'))

    print(archivelist)

    launchdir = os.getcwd()
    os.chdir(GDBroot)

    for archfold in archivelist:
        # Unzip a folder
        basename = os.path.basename(archfold)  # get filename
        filetar, filetarext = os.path.splitext(archfold)  # split into file.tar and .gz

        fileuntar, fileuntarext = os.path.splitext(filetar)  # split into file and .tar

        if filetarext == ".gz" and fileuntarext == ".tar":  # check if file had format .tar.gz
            tar = tarfile.open(basename)
            print('Extracting ' + archfold)
            tar.extractall(path=GDBroot)  # untar file into same directory
            tar.close()
            print('Folder extracted')

        # Extract data
        print('Looking for ' + os.path.join(GDBroot, fileuntar, '*a.out'))
        outfilelist = glob.glob(os.path.join(GDBroot, fileuntar, '*a.out'))

        # Create a reference to the current proc and calculate memory usage
        process = psutil.Process()
        print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')

        # Sort the output files according to the weird key above
        outfilelist = sorted(outfilelist, key=key)

        React8db = []

        # Parse each molecule in the file line by line
        for i, f in enumerate(outfilelist):
            if i % 1000 == 0:
                print(str(i) + 'molecules read')
                print('Reading ' + f)
                print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')
            mol = GDB9molecule(f, strings)
            if mol.natoms > 0:
                React8db.append(mol)
                if len(React8db) % 1000 == 0:
                    print(React8db[-1].geomfilename)

        # Write pkl file
        print('Writing ' + fileuntar + '.pkl')
        with open(fileuntar + ".pkl", "wb") as write_file:
            pickle.dump(React8db, write_file)
        # Writing file count stats file
        if os.path.exists(fileuntar + ".fst"):
            os.remove(fileuntar + ".fst")

        inp, outp, compl, conv = Monitor.AnalyzeGDB9Folder(fileuntar)
        f = open(os.path.join(fileuntar + ".fst"), 'w')
        f.write(','.join([str(inp), str(outp), str(compl), str(conv)]))
        f.close()

        # Delete folder
        print('Deleting ' + fileuntar + ' folder')
        shutil.rmtree(fileuntar)

        # Delete the molecule data, free memory
        print('Releasing memory')
        del React8db
        gc.collect()
        print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')

    print('Data extraction done.')

    os.chdir(launchdir)


def GenerateDatabaseFromFolders(strings=True, GDBfolders = []):
    # Get GDB folders
    GDBroot = '/media/kristaps/SCRATCH/Data/React8Test/out'
    
    dirlist = os.listdir(GDBroot)

    if GDBfolders == []:
        print('Preparing the directory file list...')
        for name in dirlist:
            if os.path.isdir(os.path.join(GDBroot, name)):
                GDBfolders.append(os.path.join(GDBroot, name))

    print(GDBfolders)

    launchdir = os.getcwd()
    os.chdir(GDBroot)

    for fold in GDBfolders:

        # Extract data
        print('Looking for ' + os.path.join(GDBroot, fold, '*a.out'))
        outfilelist = glob.glob(os.path.join(GDBroot, fold, '*a.out'))

        process = psutil.Process()
        print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')
        #print(outfilelist)
        #outfilelist = sorted(outfilelist, key=key)

        React8db = []

        for i, f in enumerate(outfilelist):
            if i % 1000 == 0:
                print(str(i) + 'molecules read')
                print('Reading ' + f)
                print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')
            mol = GDB9molecule(f, strings)
            if mol.natoms > 0:
                React8db.append(mol)
                if len(React8db) % 1000 == 0:
                    print(React8db[-1].geomfilename)

        # Write pkl file
        print('Writing ' + fold + '.pkl')
        with open(fold + ".pkl", "wb") as write_file:
            pickle.dump(React8db, write_file)
        # Writing file count stats file
        if os.path.exists(fold + ".fst"):
            os.remove(fold + ".fst")

        inp, outp, compl, conv = Monitor.AnalyzeGDB9Folder(fold)
        finp, foutp, fcompl, fconv = Monitor.AnalyzeGDB9Folder(fold, '/*f')
        f = open(os.path.join(fold + ".fst"), 'w')
        f.write(','.join([str(inp), str(outp), str(compl), str(conv), str(fcompl)]))
        f.close()

        # Delete the molecule data, free memory
        print('Releasing memory')
        del React8db
        gc.collect()
        print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')

    print('Data extraction done.')

    os.chdir(launchdir)


def GenerateDatabase(strings = True):
    # Get GDB folders
    GDBroot = '/scratch/ke291/GDB9React'
    dirlist = os.listdir(GDBroot)
    GDBfolders = []

    print('Preparing the directory file list...')
    for name in dirlist:
        if os.path.isdir(os.path.join(GDBroot, name)):
            GDBfolders.append(os.path.join(GDBroot, name))

    print(GDBfolders)
    #Get all ground state output files
    outfilelist = []
    print('Preparing the output file list...')
    for folder in GDBfolders:
        #outfilelist.extend(glob.glob(os.path.join(folder, '*A*a.out')))
        outfilelist.extend(glob.glob(os.path.join(folder, '*a.out')))

    process = psutil.Process()
    print(str(process.memory_info().rss/1000000000) + 'GB memory used ')

    #Sort output files
    #outfilelist = sorted(outfilelist, key = lambda x: int(x.split('/')[-1][:-7]))
    outfilelist = sorted(outfilelist, key=key)

    #print(outfilelist)

    React8db = []

    for i, f in enumerate(outfilelist):
        if i % 1000 == 0:
            print(str(i) + 'molecules read')
            print('Reading ' + f)
            print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')
        mol = GDB9molecule(f, strings)
        if mol.natoms > 0:
            React8db.append(mol)
            if len(React8db)%1000 == 0:
                print(React8db[-1].geomfilename)

    with open("React8all.pkl", "wb") as write_file:
        pickle.dump(React8db, write_file)


def LoadDBFromPickle(pklfile):
    with open(pklfile, "rb") as read_file:
        return pickle.load(read_file)


def GetFSTTotals(folder):
    fstfiles = glob.glob(os.path.join(folder, '*.fst'))

    totinp = 0
    totoutp = 0
    totcompl = 0
    totconv = 0
    totfreq = 0
    for filename in fstfiles:
        inp, outp, compl, conv, freq = 0, 0, 0, 0, 0
        print(filename)
        f = open(filename, 'r')
        data = f.readlines()
        f.close()
        for line in data:
            if ',' in line:
                tmp = line.split(',')
                if len(tmp) == 5:
                    [inp, outp, compl, conv, freq] = [int(x) for x in tmp]
                    print([inp, outp, compl, conv, freq])
                elif len(tmp) == 4:
                    [inp, outp, compl, conv] = [int(x) for x in tmp]
                    print([inp, outp, compl, conv, freq])
                else:
                    print('ERROR: Unexpected filestats data!')
        totinp += inp
        totoutp += outp
        totcompl += compl
        totconv += conv
        totfreq += freq

    print("Total inputs: " + str(totinp))
    print("Total outputs: " + str(totoutp))
    print("Total completed: " + str(totcompl))
    print("Total converged: " + str(totconv))
    print("Total frequencies: " + str(totfreq))


def AnalyzeDatabaseErrors(pklfile):

    data = LoadDBFromPickle(pklfile)

    print('Total molecules: ' + str(len(data)))

    wellterminated = 0
    converged = 0
    error = 0
    l9999 = 0       # Run out of cycles
    l202 = 0        # Small interatomic distances encountered
    l103 = 0        # Linear angle encountered
    l502 = 0        # SCF convergence failure
    l401 = 0        # ???
    l301 = 0        # ???
    l302 = 0        # ???
    l303 = 0        # ???
    l703 = 0        # ???
    l716 = 0        # ???
    NtrErr = 0      # file error, disk reading failure
    Lnk1e = 0       # file error, internal input file deleted

    for mol in data:
        if mol.cleantermination == True:
            wellterminated += 1
        if mol.converged == True:
            converged += 1
        if mol.error != '':
            error += 1
            if '9999' in mol.error:
                l9999 += 1
            elif 'l202' in mol.error:
                l202 += 1
            elif '103' in mol.error:
                l103 += 1
            elif 'l502' in mol.error:
                l502 += 1
            elif 'l401' in mol.error:
                l401 += 1
            elif 'l301' in mol.error:
                l301 += 1
            elif 'l302' in mol.error:
                l302 += 1
            elif 'l303' in mol.error:
                l303 += 1
            elif 'l703' in mol.error:
                l703 += 1
            elif 'l716' in mol.error:
                l716 += 1
            elif 'NtrErr' in mol.error:
                NtrErr += 1
            elif 'Error termination via Lnk1e at' in mol.error:
                Lnk1e += 1
            else:
                print(mol.error)

    print('Good termination: ' + str(wellterminated))
    print('Converged: ' + str(converged))
    print('Terminated with error: ' + str(error))
    print('\nError stats:')
    print('    l9999: ' + str(l9999))
    print('     l202: ' + str(l202))
    print('     l103: ' + str(l103))
    print('     l502: ' + str(l502))
    print('     l401: ' + str(l401))
    print('     l301: ' + str(l301))
    print('     l302: ' + str(l302))
    print('     l303: ' + str(l303))
    print('     l703: ' + str(l703))
    print('     l716: ' + str(l716))
    print('   NtrErr: ' + str(NtrErr))
    print('    Lnk1e: ' + str(Lnk1e))


def CheckDuplicates(pklfile):

    moldata = LoadDBFromPickle(pklfile)

    print('Total molecules: ' + str(len(moldata)))

    process = psutil.Process()
    print(str(process.memory_info().rss/1000000000) + 'GB memory used ')

    EnergyWindow = 0.019    # everything within 50kJ/mol or 0.019 hartrees is a candidate for duplicate
    SCFEs = np.array([x.SCFenergy for x in moldata])

    duplogf = open('duplog', 'w')
    numdups = 0
    indupsets = []          #register mols that are known duplicates

    for i in range(10000):
        #print(i, indupsets)
        if i not in indupsets:

            DupCandidates = []
            fileoutp = []
            dupmolids = []
            res = np.where(abs(SCFEs - SCFEs[i]) < EnergyWindow)[0]
            if (len(res) > 1):
                DupCandidates = list(res[res>=i])
            else:
                continue

            for molid in DupCandidates:
                moldata[molid].AddStrings()

            molid1 = DupCandidates[0]
            for molid2 in DupCandidates[1:]:
                if IsDuplicateMol(moldata[molid1], moldata[molid2]):
                    if fileoutp == []:
                        fileoutp.append(moldata[molid1].geomfilename.split('/')[-1])
                        indupsets.append(molid1)
                        dupmolids.append(molid1)
                    fileoutp.append(moldata[molid2].geomfilename.split('/')[-1])
                    indupsets.append(molid2)
                    dupmolids.append(molid2)

            if fileoutp != []:
                duplogf.write(','.join(fileoutp) + '\n')
                print('FILEOUTP: ' + ','.join(fileoutp) + '\n' + str(dupmolids))
                numdups += 1

            if i % 100 == 0:
                print(str(i+1) + ' molecules checked')
                print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')

            if process.memory_info().rss / 1000000000 > MEMLIM:
                print('Memory limit exceeded, quitting!')
                quit()
        else:
            print('Skipping as molecule ' + moldata[i].geomfilename.split('/')[-1] + ' already a duplicate, molid ' + str(i))

    duplogf.close()

    print(str(numdups) + ' duplicate sets written to the file')
    print(str(len(indupsets)/numdups) + ' average number of molecules per duplicate set')


def IsDuplicateMol(mol1, mol2):

    if mol1.charge != mol2.charge:
        return False

    if mol1.multiplicity != mol2.multiplicity:
        return False

    if (mol1.converged == False) or (mol2.converged == False):
        #print('Warning: molecule not converged')
        return False

    if mol1.inchi == '' or mol2.inchi =='':
        print('Warning: molecule missing inchi string')
        return False

    tmp = mol1.inchi.split('/')
    inchi1 = '/'.join(tmp[:4])

    tmp = mol2.inchi.split('/')
    inchi2 = '/'.join(tmp[:4])

    if inchi1 == inchi2:
        print('Duplicate found : ' + mol1.geomfilename.split('/')[-1] + ' ' + mol2.geomfilename.split('/')[-1])
        print('InChI1: ' + inchi1 + '    \nInChi2:' + inchi2)
        return True


def TopologyChanged(GDBmol):
    if GDBmol.converged:

        inchi1 = MolToInchi(GDBmol.atoms, GDBmol.startcoords, GDBmol.charge, GDBmol.multiplicity)
        inchi2 = MolToInchi(GDBmol.atoms, GDBmol.coords, GDBmol.charge, GDBmol.multiplicity)

        if (inchi1 == '') or (inchi2 == ''):
            return False

        tmp1 = inchi1.split('/')
        inchi1 = '/'.join(tmp1[:4])

        tmp2 = inchi2.split('/')
        inchi2 = '/'.join(tmp2[:4])

        if inchi1 != inchi2:
            #print('Topology change: ' + GDBmol.geomfilename)
            return True
    else:
        return False


def CheckTopologyChange(pklfile):

    moldata = LoadDBFromPickle(pklfile)

    print('Total molecules: ' + str(len(moldata)))

    process = psutil.Process()
    print(str(process.memory_info().rss/1000000000) + 'GB memory used ')

    toplogf = open('topchangelog', 'w')
    numtopchanges = 0
    numconverged = 0

    for i in range(10000):
        #print(i, indupsets)
        if moldata[i].converged:
            numconverged += 1

            inchi1 = MolToInchi(moldata[i].atoms, moldata[i].startcoords, moldata[i].charge, moldata[i].multiplicity)
            inchi2 = MolToInchi(moldata[i].atoms, moldata[i].coords, moldata[i].charge, moldata[i].multiplicity)

            if (inchi1 == '') or (inchi2 == ''):
                continue

            tmp1 = inchi1.split('/')
            inchi1 = '/'.join(tmp1[:4])

            tmp2 = inchi2.split('/')
            inchi2 = '/'.join(tmp2[:4])

            if inchi1 != inchi2:
                toplogf.write(moldata[i].geomfilename.split('/')[-1] + '\n')
                numtopchanges += 1

    if i % 1000 == 0:
        print(str(i+1) + ' molecules checked')
        print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')

    if process.memory_info().rss / 1000000000 > MEMLIM:
        print('Memory limit exceeded, quitting!')
        quit()

    toplogf.close()

    print(str(numconverged) + ' converged molecules analyzed')
    print(str(numtopchanges) + ' molecules with topology changes found')


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