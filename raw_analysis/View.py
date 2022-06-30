#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Kristaps Ermanis

Main window class for conex GUI
"""

import os
import sys
import glob
import Analyse

from PyQt5 import QtGui, QtWidgets, QtCore
from PIL import Image
from PIL.ImageQt import ImageQt

DIRPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))

GDataRoot = "/scratch/ke291/GDB9React"

class Window(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        self.dpi = dpi
        dirlist = sorted(os.listdir(GDataRoot), key=lambda x: int(x.split('GDB')[-1]))
        self.GDBdirs = [os.path.join(GDataRoot, o) for o in dirlist
         if os.path.isdir(os.path.join(GDataRoot, o))]
        self.scalefactor = dpi/96.0

        self.vbox = QtWidgets.QVBoxLayout()
        self.centralw = QtWidgets.QWidget()
        self.centralw.setLayout(self.vbox)
        self.setCentralWidget(self.centralw)

        self.hbox = QtWidgets.QHBoxLayout()

        self.foldcombo = QtWidgets.QComboBox()
        for dir in self.GDBdirs:
            self.foldcombo.addItem(dir.split('/')[-1], dir)

        self.outfilelist = glob.glob(os.path.join(self.foldcombo.currentData()) + '/*A*a.out')
        self.outfilelist = [x.split('A')[0].split('/')[-1] for x in self.outfilelist]
        self.outfilelist = sorted(self.outfilelist, key=int)
        self.minnum = min(self.outfilelist, key=int)
        self.maxnum = max(self.outfilelist, key=int)

        self.startlbl = QtWidgets.QLabel('Start structure (' + self.minnum + ' - ' + self.maxnum + ')')
        self.startinp = QtWidgets.QLineEdit(self.minnum)
        self.loadbutton = QtWidgets.QPushButton('Load')
        self.backbutton = QtWidgets.QPushButton('<--')
        self.forwardbutton = QtWidgets.QPushButton('-->')

        self.hbox.addWidget(self.foldcombo)
        self.hbox.addWidget(self.startlbl)
        self.hbox.addWidget(self.startinp)
        self.hbox.addWidget(self.loadbutton)
        self.hbox.addWidget(self.backbutton)
        self.hbox.addWidget(self.forwardbutton)

        self.vbox.addLayout(self.hbox)
        self.piclbl = QtWidgets.QLabel('blabla')
        self.setWindowTitle('GDB9React Explorer')
        self.vbox.addWidget(self.piclbl)
        #self.table = Table(self.SelectConfs)

        self.foldcombo.currentTextChanged.connect(self.ChDir)
        self.backbutton.clicked.connect(self.Back)
        self.forwardbutton.clicked.connect(self.Forward)
        self.loadbutton.clicked.connect(self.LoadData)

        self.resize(1300, 600)

        self.DataLoaded = False
        self.piclbl.show()  # show label with qim image

        print(dpi)

    def PILimageToQImage(self, pilimage):
        """converts a PIL image to QImage"""
        imageq = ImageQt(pilimage)  # convert PIL image to a PIL.ImageQt object
        qimage = QtGui.QImage(imageq)  # cast PIL.ImageQt object to QImage object -that´s the trick!!!
        return qimage

    def pil2pixmap(self, im):
        if im.mode == "RGB":
            r, g, b = im.split()
            im = Image.merge("RGB", (b, g, r))
        elif  im.mode == "RGBA":
            r, g, b, a = im.split()
            im = Image.merge("RGBA", (b, g, r, a))
        elif im.mode == "L":
            im = im.convert("RGBA")
        # Bild in RGBA konvertieren, falls nicht bereits passiert
        im2 = im.convert("RGBA")
        data = im2.tobytes("raw", "RGBA")
        qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_ARGB32)
        pixmap = QtGui.QPixmap.fromImage(qim)

        return pixmap

    def ChDir(self):
        self.outfilelist = glob.glob(os.path.join(self.foldcombo.currentData()) + '/*A*a.out')
        self.outfilelist = [x.split('A')[0].split('/')[-1] for x in self.outfilelist]
        self.outfilelist = sorted(self.outfilelist, key=int)
        self.minnum = min(self.outfilelist, key=int)
        self.maxnum = max(self.outfilelist, key=int)

        self.startlbl.setText('Start structure (' + self.minnum + ' - ' + self.maxnum + ')')
        self.startinp.setText(self.minnum)

    def Forward(self):
        if self.startinp.text() != self.outfilelist[-1]:
            index = self.outfilelist.index(self.startinp.text()) + 1
            self.startinp.setText(self.outfilelist[index])
            self.LoadData()

    def Back(self):
        if self.startinp.text() != self.outfilelist[0]:
            index = self.outfilelist.index(self.startinp.text()) - 1
            self.startinp.setText(self.outfilelist[index])
            self.LoadData()

    def LoadData(self):
        pilimg = Analyse.GetMoleculeImage(self.foldcombo.currentData() + '/' + self.startinp.text())

        self.DataLoaded = True
        #pilimg = Image.open('001118.png')
        pixImg = self.pil2pixmap(pilimg)
        self.piclbl.setPixmap(pixImg)


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    screen = app.screens()[0]
    dpi = float(screen.logicalDotsPerInch())

    window = Window()

    window.show()

    sys.exit(app.exec_())