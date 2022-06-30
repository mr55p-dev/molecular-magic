#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDB9 automation script

Created on 09/07/2019

@author: ke291

The main file, that should be called to start the GDB9 workflow.
Interprets the arguments and takes care of the general workflow logic.
"""

import os
import sys
import datetime
import argparse
import importlib
import XYZ
import numpy as np
from openbabel import OBMol, OBAtom, OBMolAtomIter, OBAtomAtomIter, OBAtomBondIter, OBMolBondIter
from math import sqrt

DFTpackages = [['g', 'z', 'd'],
    ['Gaussian', 'GaussianZiggy', 'GaussianDarwinQ']]

#Assigning the config default values
#Settings are defined roughly in the order they are used in the script
class Settings:
    # --- Main options ---
    DFT = 'z'       # n, j, g, z or for NWChem, Jaguar or Gaussian
    Workflow = 'gmna' # defines which steps to include in the workflow
                    # g for generate diastereomers
                    # m for molecular mechanics conformational search
                    # o for DFT optimization
                    # e for DFT single-point energies
                    # n for DFT NMR calculation
                    # s for computational and experimental NMR data extraction and stats analysis
    Solvent = ''    # solvent for DFT optimization and NMR calculation
    ScriptDir = ''  # Script directory, automatically set on launch
    InputFiles = [] # Structure input files - can be MacroModel *-out.mae or *sdf files
    Title = 'GDB9molecule'       # Title of the calculation, set to NMR file name by default on launch
    AssumeDone = False          # Assume all computations done, only read DFT output data and analyze (use for reruns)
    AssumeConverged = False     # Assume all optimizations have converged, do NMR and/or energy calcs on existing DFT geometries
    UseExistingInputs = False   # Don't regenerate DFT inputs, use existing ones. Good for restarting a failed calc
    GroundOnly = False          # Don't generate reactive variants, just optimize ground state molecules
    GOutOnly = False            # Don't generate variants from DFT-unoptimized sdf files, only read Gaussian outputs
    # --- DFT ---
    MaxDFTOptCycles = 50        # Max number of DFT geometry optimization cycles to request.
    CalcFC = False              # Calculate QM force constants before optimization
    OptStepSize = 30            # Max step Gaussian should take in geometry optimization
    charge = None               # Manually specify charge for DFT calcs
    oBasisSet = "6-31g(2df,p)"    # Basis set for geometry optimizations
    oFunctional = "b3lyp"       # Functional for geometry optimizations
    eBasisSet = "def2tzvp"      # Basis set for energy calculations
    eFunctional = "m062x"       # Functional for energy calculations

    # --- Computational clusters ---
    """ These should probably be moved to relevant *.py files as Cambridge specific """
    user = 'ke291'              # Linux user on computational clusters, not used for local calcs
    TimeLimit = 6              # Queue time limit on comp clusters
    queue = 'CLUSTER'              # Which queue to use on Ziggy
    project = 'T2-CS098-CPU' # Which project to use on Darwin
    DarwinScrDir = '/home/ke291/rds/hpc-work/'  # Which scratch directory to use on Darwin
    StartTime = ''              # Automatically set on launch, used for folder names
    nProc = 1                   # Cores used per job, must be less than node size on cluster
    DarwinNodeSize = 32         # Node size on current CSD3
    MaxConcurrentJobsZiggy = 75      # Max concurrent jobs to submit on ziggy
    MaxConcurrentJobsDarwin = 320 # Max concurrent jobs to submit on CSD3

settings = Settings()

# Data struture keeping all of isomer data in one place.
class Isomer:
    def __init__(self, InputFile='', Charge=0, Multiplicity=1, ObMol = None, Template = None, Type = 'A', Index = 1, Resub = 'a'):

        self.Charge = Charge  # externally provided charge
        self.Multiplicity = Multiplicity  # Multiplicity for the isomer
        self.OptInputFiles = []  # list of DFT input file names
        self.OptOutputFiles = []  # list of DFT output file names

        self.Type = Type
        # A - base, B - cation, C - radical, D - anion
        # E - radical cation, F - carbene, G - protonated, H - radical anion

        self.Index = Index  # Which modification of this type for this base structure
        self.Resub = Resub  # Which optimization resub for this structure
        self.MolWeight = 0  # Useful heuristic for sorting structures by comp. cost

        if (InputFile != '') and (ObMol == None):
            self.InputFile = InputFile  # Initial structure input file
            if self.InputFile[-4:] == '.sdf':
                name = self.InputFile[:-4]
                atoms = []
                coords = []

                #Check if DFT optimised geometry exists already
                # If so, read that instead
                if os.path.isfile(name + 'A1a.out'):
                    atoms, coords = self.ReadGausOutp(name + 'A1a.out')

                # in case atoms or coords are still empty, read the sdf file anyway
                if (atoms == []) or (coords == []):
                    atoms, coords, charge = self.ReadSDFGeometry(InputFile)

                self.Atoms = atoms
                self.Coords = coords
                charge = Charge
                self.MolWeight = MolWeight(atoms)
            else:
                name = self.InputFile[:-4].split('_')[1]
                self.Atoms = []             # Element labels
                self.Coords = []        # from conformational search, list of atom coordinate lists
        else:
            self.InputFile = Template.InputFile  # Initial structure input file
            if '_' in self.InputFile:
                name = self.InputFile[:-4].split('_')[1]
            elif self.InputFile[-4:] == '.sdf':
                name = self.InputFile[:-4]

            self.BaseName = name  # Basename for other files

            self.BuildFromOBMol(ObMol) # Get element labels and atom coordinates from provided ObMol

        self.BaseName = name + Type + str(Index) + Resub  # Basename for calculation files

    def GenVariants(self):

        print("Generating variants for " + self.InputFile)

        if self.SmallDistPresent():
            print("Small distances encountered, skipping...")
            return []

        baseobmol = self.BuildOBMol()

        heavyatomids = []
        heavyanums = []
        Anions = []
        Cations = []
        Radicals = []
        RadicalCats = []
        Carbenes = []
        Protonated = []
        RadicalAns = []

        for atom in OBMolAtomIter(baseobmol):
            atomid = atom.GetIdx()
            anum = atom.GetAtomicNum()
            if anum > 1:
                heavyatomids.append(atomid)
                heavyanums.append(anum)

        #print('{:d} heavy atoms found.'.format(len(heavyatomids)))
        modindex = 0
        for heavyid, heavyanum in zip(heavyatomids, heavyanums):
            ModObMol = OBMol(baseobmol)
            heavyatom = ModObMol.GetAtom(heavyid)

            for neighb in OBAtomAtomIter(heavyatom):
                if neighb.GetAtomicNum() == 1:
                    #print('Modifying atom nr {:d}'.format(heavyid))
                    modindex += 1
                    ModObMol.DeleteAtom(neighb)
                    if heavyanum == 6:
                        Cations.append(Isomer(Charge=1, ObMol=ModObMol, Template=self, Type='B', Index=modindex))
                    Radicals.append(Isomer(Multiplicity = 2, ObMol=ModObMol, Template=self, Type='C', Index=modindex))
                    Anions.append(Isomer(Charge=-1, ObMol=ModObMol, Template=self, Type='D', Index=modindex))
                    break

        #Generate radical cation
        if self.HasLonePairs(baseobmol) or self.HasPiBonds(baseobmol):
            ModObMol = OBMol(baseobmol)
            RadicalCats.append(Isomer(Charge=1, Multiplicity=2, ObMol=ModObMol, Template=self, Type='E', Index=1))

        #Generate radical anion
        if self.HasPiBonds(baseobmol):
            ModObMol = OBMol(baseobmol)
            RadicalAns.append(Isomer(Charge=-1, Multiplicity=2, ObMol=ModObMol, Template=self, Type='H', Index=1))

        # Generate carbenes - at every carbon with 2 hydrogens
        modindex = 0
        for heavyid, heavyanum in zip(heavyatomids, heavyanums):
            if heavyanum == 6:
                ModObMol = OBMol(baseobmol)
                heavyatom = ModObMol.GetAtom(heavyid)
                Hatoms = []
                for neighb in OBAtomAtomIter(heavyatom):
                    if neighb.GetAtomicNum() == 1:
                        Hatoms.append(neighb)
                if len(Hatoms) > 1:
                    modindex += 1
                    ModObMol.DeleteAtom(Hatoms[0])
                    ModObMol.DeleteAtom(Hatoms[1])
                    Carbenes.append(Isomer(Charge=0, ObMol=ModObMol, Template=self, Type='F', Index=modindex))


        #Generate protonated species
        modindex = 0
        for heavyid, heavynum in zip(heavyatomids, heavyanums):
            ModObMol = OBMol(baseobmol)
            heavyatom = ModObMol.GetAtom(heavyid)

            valence = 0
            for bond in OBAtomBondIter(heavyatom):
                valence += bond.GetBondOrder()

            LonePair = False
            if (heavyatom.GetAtomicNum() == 7) and (valence < 4):
                LonePair = True
            if (heavyatom.GetAtomicNum() == 8) and (valence < 4):
                LonePair = True

            if LonePair == False:
                continue
            else:
                atomxyz = [heavyatom.GetX(),heavyatom.GetY(),heavyatom.GetZ()]
                protvector = [0,0,0]

                # calculate the average of bond vectors
                # to determine average direction of other substituents
                for neighb in OBAtomAtomIter(heavyatom):
                    nxyz = [neighb.GetX(),neighb.GetY(),neighb.GetZ()]
                    bondvector = [x-y for x,y in zip(nxyz, atomxyz)]
                    protvector = [x + y for x,y in zip(protvector, bondvector)]

                # normalize vector to 1.05 length
                protvlength = sqrt(sum([x*x for x in protvector]))
                if protvlength == 0:
                    continue
                factor = 1.05/protvlength
                Hbondvector = [x*(-factor) for x in protvector]

                Hxyz = [x + y for x,y in zip(atomxyz, Hbondvector)]
                modindex += 1
            
                atom = OBAtom()
                atom.thisown = False
                atom.SetAtomicNum(1)
                atom.SetVector(Hxyz[0],Hxyz[1], Hxyz[2])
                ModObMol.AddAtom(atom)
                Protonated.append(Isomer(Charge=1, ObMol=ModObMol, Template=self, Type='G', Index=modindex))

        return Anions + Cations + Radicals + RadicalCats + Carbenes + Protonated + RadicalAns


    def HasPiBonds(self, obmol):
        for bond in OBMolBondIter(obmol):
            bo = bond.GetBondOrder()
            if (bo == 2) or (bo == 3) or bond.IsAromatic():
                return True
        return False


    def HasLonePairs(self, obmol):

        for atom in OBMolAtomIter(obmol):
            idx = atom.GetIdx()

            valence = 0
            for bond in OBAtomBondIter(atom):
                valence += bond.GetBondOrder()

            if (atom.GetAtomicNum() == 7) and (valence < 4):
                return True
            if (atom.GetAtomicNum() == 8) and (valence < 4):
                return True

        return False

    # utility to check atom coordinates for small distances
    def SmallDistPresent(self):

        if len(self.Coords) == 0:
            return False
        #print('Checking small distances for ' + self.BaseName)
        #print('len(self.Coords) = ' + str(len(self.Coords)))
        #print('self.Coords = ' + str(self.Coords))
        #print('self.Atoms = ' + str(self.Atoms))
        natoms = len(self.Coords)
        floatcoords = [list(map(float, acoord)) for acoord in self.Coords]
        npcoords = np.array(floatcoords)
        distmat = np.zeros((natoms, natoms))
        for i in range(natoms):
            for j in range(natoms):
                if i != j:
                    distmat[i][j] = np.linalg.norm(npcoords[i]-npcoords[j])
                else:
                    distmat[i][j] = 10000

        if np.amin(distmat) > 0.5:
            return False
        else:
            return True


    def BuildFromOBMol(self, obmol):
        coords = []
        atoms = []
        for atom in OBMolAtomIter(obmol):
            x_str = format(atom.GetX(), '.6f')
            y_str = format(atom.GetY(), '.6f')
            z_str = format(atom.GetZ(), '.6f')
            coords.append([x_str, y_str, z_str])
            atoms.append(GetAtomSymbol(atom.GetAtomicNum()))

        self.Atoms = atoms
        self.Coords = coords
        self.MolWeight = MolWeight(atoms)


    def BuildOBMol(self):
        
        atoms = self.Atoms
        coords = self.Coords

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

    def ReadGausOutp(self, GOutpFile):

        print("Reading " + GOutpFile)

        gausfile = open(GOutpFile, 'r')
        GOutp = gausfile.readlines()
        gausfile.close()

        atoms = []
        coords = []
        converged = False

        gindices = []       # list of geometry line indices
        chindex = -1        # index of line containing charge and multiplicity


        # Find the last geometry section
        for index in range(len(GOutp)):
            if ('Input orientation:' in GOutp[index]) or ("Standard orientation:" in GOutp[index]):
                gindices.append(index + 5)
            if ('Multiplicity' in GOutp[index]):
                chindex = index
            if ('Stationary' in GOutp[index]):
                converged = True

        if chindex < 0:
            print('Error: No charge and/or multiplicity found in file ' + GOutpFile)
            atoms = []
            coords = []
            return atoms, coords

        if gindices == []:
            print('Error: No geometry found in file ' + GOutpFile)
            atoms = []
            coords = []
            return atoms, coords

        # Read end geometry
        for line in GOutp[gindices[-1]:]:
            if '--------------' in line:
                break
            else:
                data = [_f for _f in line[:-1].split(' ') if _f]
                atoms.append(GetAtomSymbol(int(data[1])))
                #coords.append([float(x) for x in data[3:]])
                coords.append([x for x in data[3:]])

        return atoms, coords


    def ReadSDFGeometry(self, SDfile):

        from openbabel import OBMol, OBConversion, OBMolAtomIter

        obconversion = OBConversion()
        obconversion.SetInFormat("sdf")
        obmol = OBMol()

        print("Reading " + SDfile)

        obconversion.ReadFile(obmol, SDfile)

        obmol.ConnectTheDots()

        atoms = []
        coords = []
        charge = obmol.GetTotalCharge()

        for atom in OBMolAtomIter(obmol):
            x_str = format(atom.GetX(), '.6f')
            y_str = format(atom.GetY(), '.6f')
            z_str = format(atom.GetZ(), '.6f')
            coords.append([x_str, y_str, z_str])
            atoms.append(GetAtomSymbol(atom.GetAtomicNum()))

        return atoms, coords, charge


WTable = [1.008, 4.003, 6.941, 9.012, 10.811, 12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305, 26.982, \
          28.086, 30.974, 32.065, 35.453, 39.948]

def MolWeight(atoms):

    weight = 0

    for a in atoms:
        weight += WTable[GetAtomNum(a)-1]

    return weight

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
        quit()


def GetAtomNum(AtomSymbol):

    if AtomSymbol in PTable:
        return PTable.index(AtomSymbol)+1
    else:
        print("No such element with symbol " + str(AtomSymbol))
        quit()

def SortIsomers(Isomers):

    print('Len(Isomers): ' + str(len(Isomers)))

    Base = []
    Cations = []
    Anions = []
    Radicals = []
    RadicalCats = []
    Carbenes = []
    Protonated = []
    RadicalAns = []

    # A - base, B - cation, C - radical, D - anion
    # E - radical cation, F - carbene, G - protonated, H - radical anion

    Sorted = []

    for iso in Isomers:
        if iso.Type == 'A':
            Base.append(iso)
        elif iso.Type == 'B':
            Cations.append(iso)
        elif iso.Type == 'C':
            Radicals.append(iso)
        elif iso.Type == 'D':
            Anions.append(iso)
        elif iso.Type == 'E':
            RadicalCats.append(iso)
        elif iso.Type == 'F':
            Carbenes.append(iso)
        elif iso.Type == 'G':
            Protonated.append(iso)
        elif iso.Type == 'H':
            RadicalAns.append(iso)
        else:
            print("Sorting error, quitting...")
            quit()

    Base.sort(key=lambda a:a.MolWeight)
    Cations.sort(key=lambda a: a.MolWeight)
    Anions.sort(key=lambda a: a.MolWeight)
    Radicals.sort(key=lambda a: a.MolWeight)
    RadicalCats.sort(key=lambda a: a.MolWeight)
    Carbenes.sort(key=lambda a: a.MolWeight)
    Protonated.sort(key=lambda a: a.MolWeight)
    RadicalAns.sort(key=lambda a: a.MolWeight)

    Sorted.extend(Base)
    Sorted.extend(Cations)
    Sorted.extend(Anions)
    Sorted.extend(Radicals)
    Sorted.extend(RadicalCats)
    Sorted.extend(Carbenes)
    Sorted.extend(Protonated)
    Sorted.extend(RadicalAns)

    print('Len(RadicalAns): ' + str(len(RadicalAns)))
    print('Len(RadicalCats): ' + str(len(RadicalCats)))
    return Sorted


def main(settings):

    print("==========================")
    print("GDB9React data generation automation script")
    print("\nCopyright (c) 2019 Kristaps Ermanis")
    print("Distributed under MIT license")
    print("==========================\n\n")

    print("Initial input files: " + str(settings.InputFiles))

    # Create isomer data structures
    Isomers = []

    for f in settings.InputFiles:
        if f[-4:] == '.sdf':
            if settings.GOutOnly == True:
                # Check if DFT optimised geometry exists already
                # If so, read that instead
                if os.path.isfile(f[:-4] + 'A1a.out'):
                    print(f[:-4] + 'A1a.out found, reading preoptimized geometry')
                    Isomers.append(Isomer(f))
                else:
                    print(f[:-4] + 'A1a.out not found, skipping')
                    continue
            else:
                Isomers.append(Isomer(f))
        else:
            print('FILE!')
            Isomers.append(Isomer(f))

    Isomers = [x for x in Isomers if not x.SmallDistPresent()]

    if settings.InputFiles[0][-4:] == '.xyz':
        print('Calling XYZ.ReadInputGeometries(Isomers)')
        # Read data into isomer data structures from the xyz files
        Isomers = XYZ.ReadInputGeometries(Isomers)

    if not settings.GroundOnly:
        # Generate variants (anions, cations, radicals)
        Variants = []
        for iso in Isomers:
            Variants.extend(iso.GenVariants())

        Isomers = Isomers + Variants

        #sort by increasing complexity - base, cations, anions, radicals, then by molweight

    Isomers = SortIsomers(Isomers)

    DFT = ImportDFT(settings.DFT)

    # Run DFT optimizations

    now = datetime.datetime.now()
    settings.StartTime = now.strftime('%d%b%H%M')

    print('\nSetting up geometry optimization calculations...')
    Isomers = DFT.SetupOptCalcs(Isomers, settings)
    print('\nRunning geometry optimization calculations...')
    Isomers = DFT.RunOptCalcs(Isomers, settings)


# Selects which DFT package to import, returns imported module
def ImportDFT(dft):
    if dft in DFTpackages[0]:
        DFTindex = DFTpackages[0].index(dft)
        DFT = importlib.import_module(DFTpackages[1][DFTindex])
    else:
        print("Invalid DFT package selected")
        quit()

    return DFT


def getScriptPath():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


if __name__ == '__main__':

    #Read config file and fill in settings in from that
    #These are then overridden by any explicit parameters given through the command line

    parser = argparse.ArgumentParser(description='PyDP4 script to setup\
    and run Tinker, Gaussian (on ziggy) and DP4')
    parser.add_argument('-w', '--workflow', help="Defines which steps to include in the workflow, " +
    "can contain g for generate diastereomers, m for molecular mechanics conformational search, " +
    "o for DFT optimization, e for DFT single-point energies, n for DFT NMR calculation, " +
    "s for computational and experimental NMR data extraction and stats analysis, default is 'gmns'", default=settings.Workflow)

    parser.add_argument('-d', '--dft', help="Select DFT program, \
    g for Gaussian, z for Gaussian on ziggy, d for Gaussian on \
    Darwin, default is g", choices=DFTpackages[0], default='g')
    parser.add_argument('StructureFiles', nargs='+', default=['-'], help=
    "One or more SDF file for the structures to be verified by DP4. At least one\
    is required, if automatic diastereomer generation is used.")
    parser.add_argument("-q", "--queue", help="Specify queue for job submission\
    on ziggy", default=settings.queue)
    parser.add_argument("--TimeLimit", help="Specify job time limit for jobs\
    on ziggy or darwin", type=int)

    parser.add_argument("--nProc", help="Specify number of processor cores\
    to use for Gaussian calculations", type=int, default=1)
    parser.add_argument("--batch", help="Specify max number of jobs per batch",
    type=int, default=settings.MaxConcurrentJobsZiggy)
    parser.add_argument("--project", help="Specify project for job submission\
    on darwin", default=settings.project)

    parser.add_argument("--GroundOnly", help="No reactive variant generation, just ground state optimization",
                        action="store_true")
    parser.add_argument("--GOutOnly", help="No raw sdf file geometries, only read from Gaussian outputs",
                        action="store_true")

    parser.add_argument("--OptCycles", help="Specify max number of DFT geometry\
    optimization cycles", type=int, default=settings.MaxDFTOptCycles)
    parser.add_argument("--OptStep", help="Specify the max step size\
    Gaussian should take in optimization, default is 30", type=int, default=settings.OptStepSize)
    parser.add_argument("--FC", help="Calculate force constants before optimization", action="store_true")

    parser.add_argument('-n', '--Charge', help="Specify\
    charge of the molecule. Do not use when input files have different charges")
    parser.add_argument('-B', '--oBasisSet', help="Selects the basis set for\
    DFT NMR calculations", default=settings.oBasisSet)
    parser.add_argument('-F', '--oFunctional', help="Selects the functional for\
    DFT NMR calculations", default=settings.oFunctional)

    args = parser.parse_args()
    print(args.StructureFiles)

    settings.Title = args.StructureFiles[0]
    settings.Workflow = args.workflow

    settings.DFT = args.dft
    settings.queue = args.queue
    settings.ScriptDir = getScriptPath()
    settings.oBasisSet = args.oBasisSet
    settings.oFunctional = args.oFunctional
    settings.nProc = args.nProc
    settings.MaxConcurrentJobs = args.batch
    settings.project = args.project
    settings.MaxDFTOptCycles = args.OptCycles
    settings.OptStepSize = args.OptStep
    if args.GroundOnly:
        settings.GroundOnly = True
    if args.GOutOnly:
        settings.GOutOnly = True
    if args.FC:
        settings.CalcFC = True
    
    if args.TimeLimit:
        settings.TimeLimit = args.TimeLimit

    if args.Charge is not None:
        settings.charge = int(args.Charge)

    now = datetime.datetime.now()
    settings.StartTime = now.strftime('%d%b%H%M')

    with open('cmd.log', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    if all([x.isdigit() for x in args.StructureFiles]):
        settings.InputFiles = ['dsgdb9nsd_' + x.zfill(6) + '.xyz' for x in args.StructureFiles]
    elif '-' in args.StructureFiles[0]:
        [start, end] = args.StructureFiles[0].split('-')
        settings.InputFiles = ['dsgdb9nsd_' + str(x).zfill(6) + '.xyz' for x in range(int(start), int(end)+1)]
    else:
        settings.InputFiles = args.StructureFiles

    main(settings)
