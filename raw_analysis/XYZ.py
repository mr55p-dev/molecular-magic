#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:56:54 2019

@author: ke291

Code for reading Lilienfeld's xyz files
"""

import GDB9

GDB9DataRoot = '/scratch/ke291/GDB9/dsgdb9nsd.xyz/'

def ReadInputGeometries(Isomers):

    for i, iso in enumerate(Isomers):

        XYZfile = open(GDB9DataRoot + iso.InputFile, 'r')
        XYZdata = XYZfile.readlines()
        XYZfile.close()

        natoms = int(XYZdata[0])
        atoms = []
        coords = []

        #cycle through xyz geometry section, read geometry
        for j in range(2, 2+natoms):
            data = XYZdata[j][:-1].split('\t')
            atoms.append(data[0])
            c = [x.replace('*^', 'e') for x in data[1:4]]
            coords.append([x.strip() for x in c])

        #return atoms, coords, charge

        Isomers[i].Atoms = atoms
        Isomers[i].Coords = coords
        Isomers[i].MolWeight = GDB9.MolWeight(atoms)
        print('Len of XYZ.py returned Isomers: ' + str(len(Isomers)))
    return Isomers

