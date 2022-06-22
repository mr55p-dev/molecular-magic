#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDB9 automation script

Created on 26/10/2020

@author: ke291

The main file, that should be called to start the frequency calcs in a GDB9/React8 folder
Interprets the arguments and takes care of the general workflow logic.
Will find all converged geometries in a folder and run all missing frequency calculations
on the chosen system.
"""
import importlib
import argparse
import glob
import os
import sys
import datetime
import Gaussian
import GDB9

DFTpackages = [['g', 'z', 'd'],
    ['Gaussian', 'GaussianZiggy', 'GaussianDarwinQ']]

settings = GDB9.Settings()
settings.Pattern = '*a.out'

SortIsomers = GDB9.SortIsomers

# Data struture keeping all of isomer data in one place.
class Isomer:
    def __init__(self, InputFile):

        self.Type = InputFile[-7]
        # A - base, B - cation, C - radical, D - anion
        # E - radical cation, F - carbene, G - protonated, H - radical anion
        if self.Type == 'A':    # A - base
            self.Charge = 0
            self.Multiplicity = 1

        if self.Type == 'B':    # B - cation
            self.Charge = 1
            self.Multiplicity = 1

        if self.Type == 'C':    # C - radical
            self.Charge = 0
            self.Multiplicity = 2

        if self.Type == 'D':    # D - anion
            self.Charge = -1
            self.Multiplicity = 1

        if self.Type == 'E':    # E - radical cation
            self.Charge = 1
            self.Multiplicity = 2

        if self.Type == 'F':    # F - carbene
            self.Charge = 0
            self.Multiplicity = 1

        if self.Type == 'G':    # G - protonated
            self.Charge = 1
            self.Multiplicity = 1

        if self.Type == 'H':    # H - radical anion
            self.Charge = -1
            self.Multiplicity = 2


        self.FreqInputFiles = []  # list of DFT input file names
        self.FreqOutputFiles = []  # list of DFT output file names


        self.Index = int(InputFile[-6])  # Which modification of this type for this base structure
        self.MolWeight = 0  # Useful heuristic for sorting structures by comp. cost

        charge, multiplicity, atoms, coords = self.ReadGausOutp(InputFile)

        if charge != self.Charge:
            print('Error: Type/Charge mismatch for ' + GOutpFile)
            print('Charge should be ' + str(self.Charge) + ', is ' + str(charge))
            quit()

        if multiplicity != self.Multiplicity:
            print('Error: Type/Multiplicity mismatch for ' + GOutpFile)
            print('Charge should be ' + str(self.Multiplicity) + ', is ' + str(multiplicity))
            quit()

        # in case atoms or coords are still empty, read the sdf file anyway
        if (atoms == []) or (coords == []):
            print('Error: Geometry data missing for ' + GOutpFile)
            quit()

        self.Atoms = atoms
        self.Coords = coords
        self.MolWeight = GDB9.MolWeight(atoms)

        self.BaseName = InputFile[:8] + 'f'  # Basename for calculation files


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
        charge = -1000
        multiplicity = -1000

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
            quit()

        if gindices == []:
            print('Error: No geometry found in file ' + GOutpFile)
            quit()

        # Read end geometry
        for line in GOutp[gindices[-1]:]:
            if '--------------' in line:
                break
            else:
                data = [_f for _f in line[:-1].split(' ') if _f]
                atoms.append(GDB9.GetAtomSymbol(int(data[1])))
                #coords.append([float(x) for x in data[3:]])
                coords.append([x for x in data[3:]])

        # Read charge, multiplicity
        # Read charge and multiplicity
        data = [_f for _f in GOutp[chindex][:-1].split(' ') if _f]
        charge = int(data[2])
        multiplicity = int(data[5])

        if converged == False:
            print('Error: Geometry unconverged in file ' + GOutpFile)
            quit()

        return charge, multiplicity, atoms, coords


def main(settings):

    print("==========================")
    print("React8 frequency calculation automation script")
    print("\nCopyright (c) 2020 Kristaps Ermanis")
    print("Distributed under MIT license")
    print("==========================\n\n")

    # glob all geometry optimisation files
    inputfiles = glob.glob(settings.Pattern)
    print(str(len(inputfiles)) + " potentional " + settings.Pattern + " geometry optimization files found.")

    # check for convergence
    inputfiles = [f for f in inputfiles if Gaussian.IsGausConverged(f)]
    print("Of those " + str(len(inputfiles)) + " have converged.")

    print('Sorting...')
    inputfiles.sort()
    settings.InputFiles = inputfiles

    #print("Initial input files: " + str(settings.InputFiles))

    print('Reading geometries...')
    # Create isomer data structures
    Isomers = []

    for f in settings.InputFiles:
        Isomers.append(Isomer(f))

    #sort by increasing complexity - base, cations, anions, radicals, then by molweight

    Isomers = SortIsomers(Isomers)

    DFT = ImportDFT(settings.DFT)

    # Run DFT optimizations
    now = datetime.datetime.now()
    settings.StartTime = now.strftime('%d%b%H%M')
    print('\nSetting up frequency calculations...')
    Isomers = DFT.SetupFreqCalcs(Isomers, settings)
    print('\nRunning frequency calculations...')
    Isomers = DFT.RunFreqCalcs(Isomers, settings)


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

    # Read config file and fill in settings in from that
    # These are then overridden by any explicit parameters given through the command line

    parser = argparse.ArgumentParser(description='React8 frequency calculation automation script')
    parser.add_argument('-w', '--workflow', default=settings.Workflow)

    parser.add_argument('-d', '--dft', help="Select DFT program, \
    g for Gaussian, z for Gaussian on ziggy, d for Gaussian on \
    Darwin, default is g", choices=DFTpackages[0], default='g')

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
    parser.add_argument("--pattern", help="Specify pattern for input file selection, default is " +
                                          settings.Pattern, default=settings.Pattern)

    parser.add_argument("--GroundOnly", help="No reactive variant generation, just ground state optimization",
                        action="store_true")
    parser.add_argument("--GOutOnly", help="No raw sdf file geometries, only read from Gaussian outputs",
                        action="store_true")

    parser.add_argument('-n', '--Charge', help="Specify\
    charge of the molecule. Do not use when input files have different charges")
    parser.add_argument('-B', '--oBasisSet', help="Selects the basis set for\
    DFT NMR calculations", default=settings.oBasisSet)
    parser.add_argument('-F', '--oFunctional', help="Selects the functional for\
    DFT NMR calculations", default=settings.oFunctional)

    args = parser.parse_args()

    settings.Title = os.getcwd().split('/')[-1] + 'freq' #Set calculation title to the name of the currect directory
    settings.Workflow = args.workflow

    settings.DFT = args.dft
    settings.queue = args.queue
    settings.ScriptDir = getScriptPath()
    settings.oBasisSet = args.oBasisSet
    settings.oFunctional = args.oFunctional
    settings.nProc = args.nProc
    settings.MaxConcurrentJobs = args.batch
    settings.project = args.project
    settings.Pattern = args.pattern
    if args.GroundOnly:
        settings.GroundOnly = True
    if args.GOutOnly:
        settings.GOutOnly = True

    if args.TimeLimit:
        settings.TimeLimit = args.TimeLimit

    if args.Charge is not None:
        settings.charge = int(args.Charge)

    now = datetime.datetime.now()
    settings.StartTime = now.strftime('%d%b%H%M')

    with open('cmd.log', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    main(settings)
