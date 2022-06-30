#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for showing molecules with OpenGL

Created on 17/10/2019

@author: ke291

"""

from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import numpy as np
from math import sqrt, pi, acos
import math
from openbabel import *
import sys

cpk = ['#FFFFFF', '#D9FFFF', '#CC80FF',  '#C2FF00', '#FFB5B5', '#909090',\
       '#3050F8', '#FF0D0D', '#90E050', '#B3E3F5', '#AB5CF2', '#8AFF00',\
       '#BFA6A6', '#F0C8A0', '#FF8000', '#FFFF30', '#1FF01F', '#80D1E3',
       '#8F40D4', '#3DFF00', '#E6E6E6', '#BFC2C7', '#A6A6AB', '#8A99C7',
       '#9C7AC7', '#E06633', '#F090A0', '#50D050', '#C88033', '#7D80B0',
       '#C28F8F', '#668F8F', '#BD80E3', '#FFA100', '#A62929', '#5CB8D1',
       '#702EB0', '#00FF00', '#94FFFF', '#94E0E0', '#73C2C9', '#54B5B5',
       '#3B9E9E', '#248F8F', '#0A7D8C', '#006985', '#C0C0C0', '#FFD98F',
       '#A67573', '#668080', '#9E63B5', '#D47A00', '#940094', '#429EB0',
       '#57178F', '#00C900', '#70D4FF', '#FFFFC7', '#D9FFC7', '#C7FFC7',
       '#A3FFC7', '#8FFFC7', '#61FFC7', '#45FFC7', '#30FFC7', '#1FFFC7',
       '#00FF9C', '#00E675', '#00D452', '#00BF38', '#00AB24', '#4DC2FF',
       '#4DA6FF', '#2194D6', '#267DAB', '#266696', '#175487', '#D0D0E0',
       '#FFD123', '#B8B8D0', '#A6544D', '#575961', '#9E4FB5', '#AB5C00',
       '#754F45', '#428296']

AtomRadius = 0.3
BondRadius = 0.1

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


name = 'GLMol'
mousedown = False
mousewheel = 10
wheelchange = False
last_mx = 0
last_my = 0
cur_mx = 0
cur_my = 0
rotation_matrix = []

window_width = 800
window_height = 800

atoms = []
coords = []
colors = []
bonds = []


def ReadGausGeometry(GOutpFile, gindex = -1):

    gausfile = open(GOutpFile, 'r')
    GOutp = gausfile.readlines()
    gausfile.close()

    atoms = []
    coords = []

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
            coords.append([float(x) for x in data[3:]])

    #return atoms, coords, charge
    obmol = BuildOBMol(atoms, coords)

    bonds = []
    for bond in OBMolBondIter(obmol):
        satom = bond.GetBeginAtomIdx()
        eatom = bond.GetEndAtomIdx()
        bonds.append([satom - 1, eatom - 1])

    return atoms, coords, bonds


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

    return mol


def init(gausoutp):
    global atoms, coords, colors, bonds
    global window_width, window_height
    atoms, coords, bonds = ReadGausGeometry(gausoutp)
    colors = [cpk[GetAtomNum(a)-1] for a in atoms]
    colors = [Hex2RGBA(c) for c in colors]
    print(coords)
    glutInit(name)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(window_width, window_height)
    glutInitWindowPosition(350, 200)
    glutCreateWindow(name)
    glClearColor(0., 0., 0., 1.)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    lightZeroPosition = [10., 4., 10., 1.]
    lightZeroColor = [1.0, 1.0, 1.0, 1.0]
    glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor)
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
    glEnable(GL_LIGHT0)


def on_click(button, state, x, y):
    global mousedown, last_mx, last_my, cur_mx, cur_my, mousewheel, wheelchange
    if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
        mousedown = True
        cur_mx = x
        cur_my = y
        last_mx = x
        last_my = y
    elif button == GLUT_LEFT_BUTTON and state == GLUT_UP:
        mousedown = False
    elif button == 4:
        mousewheel += 3
        wheelchange = True
    elif button == 3:
        mousewheel -= 3
        wheelchange = True


def on_mouse_motion(x, y):
    global mousedown, cur_mx, cur_my
    if mousedown:
        cur_mx = x
        cur_my = y

def get_arcball_vector(x, y):
    global window_width, window_height
    P = [1.0*x/window_width*2 -1.0, 1.0*y/window_height*2 - 1.0, map_hemisphere(x, y)]
    P[1] = -P[1]
    OP_squared = P[0]*P[0] + P[1]*P[1]
    if (OP_squared <= 1.0):
        P[2] = sqrt(1.0 - OP_squared)
    else:
        normfactor = 1/sqrt(np.dot(P, P))
        P = [x/normfactor for x in P]

    return P

#Hemisphere mapping
def map_hemisphere(x,y):
    z = math.sqrt(abs(1-math.pow(x,2)-math.pow(y,2)))
    return z


def Hex2RGBA(hexcolor):
    hexcolor.lstrip('#')
    return [int(hexcolor.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)] + [1.0]


def draw_bond(c1, c2, color):
    v2r = np.array(c2) - np.array(c1)
    z = np.array([0.0, 0.0, 1.0])
    # the rotation axis is the cross product between Z and v2r
    ax = np.cross(z, v2r)
    l = sqrt(np.dot(v2r, v2r))
    # get the angle using a dot product
    angle = 180.0 / pi * acos(np.dot(z, v2r) / l)

    glPushMatrix()
    glMaterialfv(GL_FRONT, GL_DIFFUSE, color)
    glTranslatef(c1[0], c1[1], c1[2])
    glRotatef(angle, ax[0], ax[1], ax[2])
    glutSolidCylinder(BondRadius, l, 20, 1)
    glPopMatrix()


def magnitude(v):
    return math.sqrt(np.sum(v ** 2))


def normalize(v):
    m = magnitude(v)
    if m == 0:
        return v
    return v / m

def rotate(a, xyz):
    x, y, z = normalize(xyz)
    s = math.sin(a)
    c = math.cos(a)
    nc = 1 - c
    return np.matrix([[x*x*nc +   c, x*y*nc - z*s, x*z*nc + y*s, 0],
                      [y*x*nc + z*s, y*y*nc +   c, y*z*nc - x*s, 0],
                      [x*z*nc - y*s, y*z*nc + x*s, z*z*nc +   c, 0],
                      [           0,            0,            0, 1]])

def display_scene():
    global coords, colors, mousewheel
    global cur_mx, cur_my, last_mx, last_my
    global rotation_matrix

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0, 0, mousewheel,
              0, 0, 0,
              0, 1, 0)

    if (cur_mx != last_mx) or (cur_my != last_my):
        va = get_arcball_vector(last_mx, last_my)
        vb = get_arcball_vector(cur_mx, cur_my)
        angle = acos(min(1.0, np.dot(va, vb)))
        axis = np.cross(va, vb)
        rotation_matrix = np.matmul(rotation_matrix, rotate(angle, -axis))
        last_mx = cur_mx
        last_my = cur_my

    glMultMatrixf(rotation_matrix)

    for (x, y, z), color in zip(coords, colors):
        glPushMatrix()
        glMaterialfv(GL_FRONT, GL_DIFFUSE, color)
        glTranslatef(x, y, z)
        glutSolidSphere(AtomRadius, 50, 20)
        glPopMatrix()

    for a1, a2 in bonds:
        draw_bond(coords[a1], coords[a2], Hex2RGBA('#909090'))

    glFlush()
    glutSwapBuffers()
    glutPostRedisplay()


def main(gausoutp):
    global rotation_matrix
    init(gausoutp)
    glutDisplayFunc(display_scene)
    glutMouseFunc(on_click)
    glutMotionFunc(on_mouse_motion)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(54., window_width/window_height, 1., 40.)
    glMatrixMode(GL_MODELVIEW)
    gluLookAt(0, 0, 10,
              0, 0, 0,
              0, 1, 0)
    glPushMatrix()
    rotation_matrix = np.identity(4)
    glutMainLoop()


if __name__ == '__main__':
    main(sys.argv[1])