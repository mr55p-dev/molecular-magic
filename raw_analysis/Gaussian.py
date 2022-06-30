#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:56:54 2014
Rewritten during April 2019

@author: ke291

Contains all of the Gaussian specific code for input generation and calculation
execution. Called by PyDP4.py.
"""

import subprocess
import os
import time


def SetupOptCalcs(Isomers, settings):

    for iso in Isomers:
        filename = iso.BaseName

        if os.path.exists(filename + '.out') and IsGausCompleted(filename + '.out'):

            """
            if IsGausCompleted(filename + '.out'):
               
                if IsGausConverged(filename + '.out') or (settings.AssumeConverged == True):
                    iso.OptOutputFiles.append(filename + '.out')
                    continue
                else:
                    # If calculation completed, but didn't converge, reuse geometry and resubmit
                    atoms, coords = ReadGeometry(filename + '.out')
                    if coords != []:
                        print('Partially optimised structure found for ' + filename + ', reusing')
                        iso.Coords = coords
                    os.remove(filename + '.out')
               
                continue
            else:
                os.remove(filename + '.out')
            """
            continue
        else:
            if os.path.exists(filename + '.out'):
                os.remove(filename + '.out')

        WriteGausFile(filename, iso.Coords, iso.Atoms, iso.Charge, iso.Multiplicity, settings, 'opt')
        iso.OptInputFiles.append(filename + '.com')

    return Isomers


def SetupFreqCalcs(Isomers, settings):
    for iso in Isomers:
        filename = iso.BaseName

        if os.path.exists(filename + '.out') and IsGausCompleted(filename + '.out'):
            continue
        else:
            if os.path.exists(filename + '.out'):
                os.remove(filename + '.out')

        WriteGausFile(filename, iso.Coords, iso.Atoms, iso.Charge, iso.Multiplicity, settings, 'freq')
        iso.FreqInputFiles.append(filename + '.com')

    return Isomers


def Converged(Isomers):

    jobdir = os.getcwd()

    for iso in Isomers:
        filename = iso.BaseName

        if os.path.exists(filename + '.out'):
            if IsGausConverged(filename + '.out') == False:
                os.chdir(jobdir)
                return False
        else:
            os.chdir(jobdir)
            return False

    return True


def RunOptCalcs(Isomers, settings):

    print('\nRunning Gaussian DFT geometry optimizations locally...')

    GausJobs = []

    for iso in Isomers:
        GausJobs.extend([x for x in iso.OptInputFiles if (x[:-4] + '.out') not in iso.OptOutputFiles])

    Completed = RunCalcs(GausJobs)

    for iso in Isomers:
        iso.OptOutputFiles.extend([x[:-4] + '.out' for x in iso.OptInputFiles if (x[:-4] + '.out') in Completed])

    return Isomers


def RunFreqCalcs(Isomers, settings):

    print('\nRunning Gaussian frequency calculations locally...')

    GausJobs = []

    for iso in Isomers:
        GausJobs.extend([x for x in iso.FreqInputFiles if (x[:-4] + '.out') not in iso.FreqOutputFiles])

    Completed = RunCalcs(GausJobs)

    for iso in Isomers:
        iso.FreqOutputFiles.extend([x[:-4] + '.out' for x in iso.FreqInputFiles if (x[:-4] + '.out') in Completed])

    return Isomers


def RunCalcs(GausJobs):

    NCompleted = 0
    Completed = []
    gausdir = os.environ['GAUSS_EXEDIR']
    GausPrefix = gausdir + "/g09 < "

    for f in GausJobs:
        time.sleep(3)
        print(GausPrefix + f + ' > ' + f[:-3] + 'out')
        outp = subprocess.check_output(GausPrefix + f + ' > ' + f[:-3] + 'out', shell=True)

        NCompleted += 1
        if IsGausCompleted(f[:-4] + '.out'):
            Completed.append(f[:-4] + '.out')
            print("Gaussian job " + str(NCompleted) + " of " + str(len(GausJobs)) + \
                  " completed.")
        else:
            print("Gaussian job terminated with an error. Continuing.")

    if NCompleted > 0:
        print(str(NCompleted) + " Gaussian jobs completed successfully.")
    elif len(GausJobs) == 0:
        print("There were no jobs to run.")

    return Completed


def WriteGausFile(Gausinp, conformer, atoms, charge, multiplicity, settings, type):
    
    f = open(Gausinp + '.com', 'w')
    if(settings.nProc > 1):
        f.write('%nprocshared=' + str(settings.nProc) + '\n')
    if settings.DFT == 'z':
        f.write('%mem=500MB\n%chk='+Gausinp + '.chk\n')
    else:
        f.write('%mem=3400MB\n%chk='+Gausinp + '.chk\n')

    if type == 'nmr':
        f.write(NMRRoute(settings))
    elif type == 'e':
        f.write(ERoute(settings))
    elif type == 'opt':
        f.write(OptRoute(settings))
    elif type == 'freq':
        f.write(FreqRoute(settings))
    else:
        print("WriteGausFile: Unrecognized calculation type " + type)
        quit()

    f.write('\n'+Gausinp+'\n\n')
    f.write(str(charge) + ' ' + str(multiplicity) + '\n')

    natom = 0

    for atom in conformer:
        f.write(atoms[natom] + '  ' + atom[0] + '  ' + atom[1] + '  ' +
                atom[2] + '\n')
        natom = natom + 1
    f.write('\n')

    f.close()


def FreqRoute(settings):
    route = '# ' + settings.oFunctional + '/' + settings.oBasisSet
    if (settings.oFunctional).lower() == 'm062x':
        route += ' int=ultrafine'

    if settings.Solvent != '':
        route += ' scrf=(solvent=' + settings.Solvent + ')'

    route += ' NoSymm Freq\n'

    return route


def NMRRoute(settings):

    route = '# ' + settings.nFunctional + '/' + settings.nBasisSet
    if (settings.nFunctional).lower() == 'm062x':
        route += ' int=ultrafine'

    route += ' nmr=giao'

    if settings.Solvent != '':
        route += ' scrf=(solvent=' + settings.Solvent + ')'

    route += '\n'

    return route


def ERoute(settings):

    route = '# ' + settings.eFunctional + '/' + settings.eBasisSet
    if (settings.eFunctional).lower() == 'm062x':
        route += ' int=ultrafine'

    if settings.Solvent != '':
        route += ' scrf=(solvent=' + settings.Solvent + ')'

    route += '\n'

    return route


def OptRoute(settings):

    route = '# ' + settings.oFunctional + '/' + settings.oBasisSet

    if (settings.oFunctional).lower() == 'm062x':
        route += ' int=ultrafine'

    route += ' Opt=(maxcycles=' + str(settings.MaxDFTOptCycles)
    if settings.CalcFC == True:
        route += ',CalcFC'
    if (settings.OptStepSize != 30):
        route += ',MaxStep=' + str(settings.OptStepSize)
    route += ')'

    if settings.Solvent != '':
        route += ' scrf=(solvent=' + settings.Solvent + ')'

    route += '\n'

    return route


def IsGausCompleted(f):
    import os.path
    if not os.path.isfile(f):
        return False
    Gfile = open(f, 'r')
    outp = Gfile.readlines()
    Gfile.close()
    if len(outp) < 10:
        return False
    if ("Normal termination" in outp[-1]):
        return True
    if ('termination' in '\n'.join(outp[-3:])):
        return True
    """if (('termination' in '\n'.join(outp[-3:])) and ('l9999.exe' in '\n'.join(outp[-3:]))):
        return True
    if (('termination' in '\n'.join(outp[-3:])) and ('l103.exe' in '\n'.join(outp[-3:]))):
        return True
    """
    return False


def IsGausConverged(f):
    Gfile = open(f, 'r')
    outp = Gfile.readlines()
    Gfile.close()
    ginp = '\n'.join(outp)
    if ("Normal termination" not in ginp):
        return False
    if 'Stationary point found' in ginp:
        return True
    else:
        return False


#Read energy from e, if not present, then o, if not present, then nmr
def ReadEnergies(Isomers, settings):
    jobdir = os.getcwd()

    for i, iso in enumerate(Isomers):

        GOutpFiles = iso.OptOutputFiles

        DFTEnergies = []
        for GOutpFile in GOutpFiles:
            gausfile = open(GOutpFile, 'r')
            GOutp = gausfile.readlines()
            gausfile.close()

            for line in GOutp:
                if 'SCF Done:' in line:
                    start = line.index(') =')
                    end = line.index('A.U.')
                    energy = float(line[start + 4:end])

            #iso.DFTEnergies.append(energy)
            DFTEnergies.append(energy)

        Isomers[i].DFTEnergies = DFTEnergies

    os.chdir(jobdir)
    return Isomers


def ReadGeometry(GOutpFile):

    gausfile = open(GOutpFile, 'r')
    GOutp = gausfile.readlines()
    gausfile.close()

    atoms = []
    coords = []
    gindex = -1

    #Find the last geometry section
    for index in range(len(GOutp)):
        if ('Input orientation:' in GOutp[index]) or ("Standard orientation:" in GOutp[index]):
            gindex = index + 5

    if gindex < 0:
        print('Error: No geometry found in file ' + GOutpFile)
        quit()

    #Read geometry
    for line in GOutp[gindex:]:
        if '--------------' in line:
            break
        else:
            data = [_f for _f in line[:-1].split(' ') if _f]
            atoms.append(GetAtomSymbol(int(data[1])))
            coords.append(data[3:])

    #return atoms, coords, charge

    return atoms, coords


def ReadFrequencies(GOutpFile):
    gausfile = open(GOutpFile, 'r')
    GOutp = gausfile.readlines()
    gausfile.close()

    freqs = []
    for line in GOutp:
        if 'Frequencies --' in line:
            data = [_f for _f in line[17:-1].split(' ') if _f]
            print(data)
            freqs.extend([float(x) for x in data])

    print(freqs)

    return freqs


def ReadChargeMult(GOutpFile):

    gausfile = open(GOutpFile, 'r')
    GOutp = gausfile.readlines()
    gausfile.close()

    charge = 0
    multiplicity = 1
    chindex = -1

    #Find the last geometry section
    for index in range(len(GOutp)):
        if ('Multiplicity' in GOutp[index]):
            chindex = index

    if chindex < 0:
        print('Error: No charge and/or multiplicity found in file ' + GOutpFile)
        quit()

    #Read charge and multiplicity
    data = [_f for _f in GOutp[chindex][:-1].split(' ') if _f]
    charge = int(data[2])
    multiplicity = int(data[5])

    return charge, multiplicity


def ReadGeometries(Isomers):

    for iso in Isomers:

        iso.DFTConformers = [[] for x in iso.OptOutputFiles]

        for num, GOutpFile in enumerate(iso.OptOutputFiles):

            atoms, coords = ReadGeometry(GOutpFile)

            iso.DFTConformers[num] = coords

    #return atoms, coords, charge
    return Isomers


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
