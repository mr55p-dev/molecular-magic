#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Kristaps Ermanis

Contains code for analyzing multimolecule conformational searches
"""

from openbabel.openbabel import OBMol, OBAtom, OBConversion, OBMolBondIter, OBMolAtomIter, OBAtomAtomIter
import numpy
from math import sqrt, pi, cos, sin, acos

def GetMolParts(obmol):
    # get molecular graph
    molgraph = GenMolGraph(obmol)
    atoms = [x[0] for x in molgraph]
    PartAtoms = []
    parts = []

    for a in atoms:
        if a not in PartAtoms:
            parts.append(MapPart(molgraph, a))
            PartAtoms.extend(parts[-1])

    return parts


def MapPart(molgraph, atom):

    explored = [atom]
    seen = molgraph[atom-1]

    for a in seen:
        if a not in explored:
            nbs = molgraph[a-1][1:]
            seen.extend([x for x in nbs if x not in explored])
            explored.append(a)

    return explored


def GenMolGraph(obmol):

    molgraph = []

    for atom in OBMolAtomIter(obmol):
        idx = atom.GetIdx()
        molgraph.append([])
        molgraph[idx-1].append(idx)

        for NbrAtom in OBAtomAtomIter(atom):
            molgraph[idx-1].append(NbrAtom.GetIdx())

    return molgraph


def GetAtomCoordsMasses(obmol):
    masses = []
    coords = []

    for atom in OBMolAtomIter(obmol):
        idx = atom.GetIdx()
        masses.append(atom.GetAtomicMass())
        coords.append([atom.GetX(), atom.GetY(), atom.GetZ()])

    return masses, coords


def PartCoMs(obmol, parts):

    masses, coords = GetAtomCoordsMasses(obmol)
    
    CMs = []
    for part in parts:
        partcoords = numpy.array([coords[x] for x in range(len(coords)) if x+1 in part])
        partmasses = numpy.array([masses[x] for x in range(len(masses)) if x+1 in part])

        print(partcoords)
        print(partmasses)
        CMs.append(numpy.average(partcoords, axis = 0, weights=partmasses))

    return CMs


def GetCoMDists(obmol, parts):

    CoMs = PartCoMs(obmol, parts)
    print(CoMs)

    import itertools
    CoMidxs = list(range(len(CoMs)))

    PartDistIdxs = [list(x) for x in itertools.combinations(CoMidxs, 2)]

    PartDists = []
    for pair in PartDistIdxs:
        dist = numpy.linalg.norm(CoMs[pair[0]]-CoMs[pair[1]])
        PartDists.append(dist)

    return CoMs, PartDists


def PrincipalAxes(obmol, parts):
    masses, coords = GetAtomCoordsMasses(obmol)

    PrincipalAxes = []
    AllEVals = []
    AllEVecs = []
    for part in parts:
        print("This part has " + str(len(part)) + " atoms")
        partcoords = numpy.array([coords[x] for x in range(len(coords)) if x + 1 in part])
        partmasses = numpy.array([masses[x] for x in range(len(masses)) if x + 1 in part])

        #get centre of mass
        CoM = numpy.average(partcoords, axis = 0, weights=partmasses)

        #centre the coordinates
        partcoords = partcoords - CoM

        # compute principal axis matrix
        inertia = numpy.dot(partcoords.transpose(), partcoords)

        e_values, e_vectors = numpy.linalg.eig(inertia)

        print("(Unordered) eigen values:")
        print(e_values)
        print("(Unordered) eigen vectors:")
        print(e_vectors)

        order = numpy.argsort(e_values)
        eval3, eval2, eval1 = e_values[order]
        AllEVals.append([eval1, eval2, eval3])
        axis3, axis2, axis1 = e_vectors[:, order].transpose()
        EVecs = [list(x/numpy.linalg.norm(x)) for x in [axis1, axis2, axis3]]
        AllEVecs.append(EVecs)

    #lEVecs = []
    #for Vecs in AllEVecs:
    #    lEVecs.append([list(x) for x in Vecs])

    angles = GetPartAngles(AllEVecs, parts)

    return AllEVals, AllEVecs, angles


def GetPartAngles(EVecs, parts):

    MaxPartID = parts.index(max(parts, key=len))

    RefVecs = EVecs[MaxPartID]

    angles = []

    for i in range(len(EVecs)):
        if i != MaxPartID:
            for rvec in RefVecs:
                angles.append(180*VecAngle(rvec, EVecs[i][0])/pi)

    return angles


def dotproduct(v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return sqrt(dotproduct(v, v))


def VecAngle(v1, v2):
    return acos(dotproduct(v1, v2) / (length(v1) * length(v2)))