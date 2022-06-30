#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for finding and analyzing reactions in React8 database

Created on 2/10/2019

@author: ke291

"""

import Database
import psutil
import numpy as np

MEMLIM = 10.0

def FindDeprotonations(pklfile):

    moldata = Database.LoadDBFromPickle(pklfile)

    print('Total molecules: ' + str(len(moldata)))

    process = psutil.Process()
    print(str(process.memory_info().rss/1000000000) + 'GB memory used ')

    molnames = np.array([x.geomfilename.split('/')[-1] for x in moldata])

    deprlogf = open('deprotlog', 'w')
    ndeprots = 0
    for i in range(len(moldata)):
        if ('A' in molnames[i]) and moldata[i].converged:
            basename = molnames[i].split('A')[0]
            basescfe = moldata[i].SCFenergy
            deprotmolids = [j for j, x in enumerate(molnames) if basename + 'D' in x]
            #print([molnames[x] for x in deprotmolids])

            for deprmol in deprotmolids:
                if moldata[deprmol].converged:
                    deprscfe = moldata[deprmol].SCFenergy
                    outp = ','.join([molnames[i], molnames[deprmol], str(basescfe), str(deprscfe)]) +\
                           ',' + '{:0.2f}'.format((deprscfe - basescfe)*627.509474) + '\n'
                    deprlogf.write(outp)
                    #print(outp)
                    ndeprots += 1

        if i % 1000 == 0:
            print(str(i+1) + ' molecules checked')
            print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')

        if process.memory_info().rss / 1000000000 > MEMLIM:
            print('Memory limit exceeded, quitting!')
            quit()

    deprlogf.close()

    print(str(ndeprots) + ' deprotonations written to the file')
    print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')


def FindProtonations(pklfile):

    moldata = Database.LoadDBFromPickle(pklfile)

    print('Total molecules: ' + str(len(moldata)))

    process = psutil.Process()
    print(str(process.memory_info().rss/1000000000) + 'GB memory used ')

    molnames = np.array([x.geomfilename.split('/')[-1] for x in moldata])

    protlogf = open('protlog', 'w')
    nprots = 0
    for i in range(len(moldata)):
        if ('A' in molnames[i]) and moldata[i].converged:
            basename = molnames[i].split('A')[0]
            basescfe = moldata[i].SCFenergy
            protmolids = [j for j, x in enumerate(molnames) if basename + 'G' in x]
            #print([molnames[x] for x in deprotmolids])

            for prmol in protmolids:
                if moldata[prmol].converged:
                    prscfe = moldata[prmol].SCFenergy
                    outp = ','.join([molnames[i], molnames[prmol], str(basescfe), str(prscfe)]) +\
                           ',' + '{:0.2f}'.format((prscfe - basescfe)*627.509474) + '\n'
                    protlogf.write(outp)
                    #print(outp)
                    nprots += 1

        if i % 1000 == 0:
            print(str(i+1) + ' molecules checked')
            print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')

        if process.memory_info().rss / 1000000000 > MEMLIM:
            print('Memory limit exceeded, quitting!')
            quit()

    protlogf.close()

    print(str(nprots) + ' protonations written to the file')
    print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')

    quit()


def FindHAbstractions(pklfile):

    moldata = Database.LoadDBFromPickle(pklfile)

    print('Total molecules: ' + str(len(moldata)))

    process = psutil.Process()
    print(str(process.memory_info().rss/1000000000) + 'GB memory used ')

    molnames = np.array([x.geomfilename.split('/')[-1] for x in moldata])

    habstlogf = open('habstlog', 'w')
    nhabsts = 0
    for i in range(len(moldata)):
        if ('A' in molnames[i]) and moldata[i].converged:
            basename = molnames[i].split('A')[0]
            basescfe = moldata[i].SCFenergy
            habstmolids = [j for j, x in enumerate(molnames) if basename + 'C' in x]
            #print([molnames[x] for x in deprotmolids])

            for radmol in habstmolids:
                if moldata[radmol].converged:
                    radscfe = moldata[radmol].SCFenergy
                    outp = ','.join([molnames[i], molnames[radmol], str(basescfe), str(radscfe)]) +\
                           ',' + '{:0.2f}'.format((radscfe - basescfe)*627.509474) + '\n'
                    habstlogf.write(outp)
                    #print(outp)
                    nhabsts += 1

        if i % 1000 == 0:
            print(str(i+1) + ' molecules checked')
            print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')

        if process.memory_info().rss / 1000000000 > MEMLIM:
            print('Memory limit exceeded, quitting!')
            quit()

    habstlogf.close()

    print(str(nhabsts) + ' H atom abstractions written to the file')
    print(str(process.memory_info().rss / 1000000000) + 'GB memory used ')
