#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDB9 monitoring script

Created on 12/07/2019

@author: ke291

"""

import curses
import time
import subprocess
import os
import glob

user = 'ke291'

def main(stdscr):
    # Clear screen
    stdscr.clear()

    curses.use_default_colors()
    for i in range(0, curses.COLORS):
        curses.init_pair(i, i, -1);

    #Color pair 1 - red on black, 2 - green on black, 3 - yellow on black
    stdscr.bkgd(' ', curses.color_pair(0))
    stdscr.border()
    stdscr.nodelay(1)
    #stdscr.addstr('Screen size: {} x {}\n'.format(curses.LINES, curses.COLS))
    stdscr.addstr(0,1,' MONITOR ', curses.A_BOLD | curses.color_pair(0))

    stdscr.addstr(3, 3, 'DARWIN', curses.A_BOLD | curses.color_pair(0))
    stdscr.addstr(4, 4, 'Jobs:', curses.color_pair(0))

    stdscr.addstr(5, 5, 'Pending:{:3d}      Running:{:3d}'.format(0, 0), curses.color_pair(0))

    stdscr.addstr(6, 4, 'Quota:', curses.color_pair(0))

    stdscr.addstr(7, 5, '/home/:{:3d}GB    /scratch/:{:3d}GB'.format(0, 0), curses.color_pair(0))


    stdscr.addstr(10, 3, 'ZIGGY', curses.A_BOLD | curses.color_pair(0))
    stdscr.addstr(11, 4, 'Jobs:', curses.color_pair(0))

    stdscr.addstr(12, 5, 'Pending:{:3d}      Running:{:3d}'.format(0, 0), curses.color_pair(0))
    stdscr.addstr(13, 4, 'Quota:', curses.color_pair(0))
    stdscr.addstr(14, 5, '/home/:{:3d}GB    /scratch/:{:3d}GB'.format(0, 0), curses.color_pair(0))

    stdscr.addstr(17, 3, 'GDB9React', curses.A_BOLD | curses.color_pair(0))
    stdscr.addstr(18, 5, 'Inputs:{:3d}'.format(0), curses.color_pair(0))
    stdscr.addstr(19, 5, 'Outputs:{:3d}'.format(0), curses.color_pair(0))
    stdscr.addstr(20, 5, 'Completed:{:3d}'.format(0), curses.color_pair(0))
    stdscr.addstr(21, 5, 'Converged:{:3d}'.format(0), curses.color_pair(0))

    datex = curses.COLS - len(' ' + time.asctime(time.localtime(time.time())) + ' ') - 1

    countdown = 0
    checkinterval = 900
    while True:
        if stdscr.getch() == ord('q'):
            break
        stdscr.addstr(0,datex, ' ' + time.asctime(time.localtime(time.time())) + ' ')
        nupdate = 'Next update in {:2d}m {:2d}s'.format(int(countdown/60), countdown % 60)
        stdscr.addstr(1, curses.COLS - len(nupdate) - 2, nupdate)
        stdscr.refresh()

        countdown -= 1

        if countdown < 0:
            Pending, Running, home, scratch = CheckDarwin()
            stdscr.addstr(5, 5, 'Pending:{:3d}      Running:{:3d}'.format(Pending, Running), curses.color_pair(0))
            stdscr.addstr(7, 5, '/home/:{:3.1f}GB    /scratch/:{:3.1f}GB'.format(home[0], scratch[0]), curses.color_pair(0))
            Pending, Running, home, scratch = CheckZiggy()
            stdscr.addstr(12, 5, 'Pending:{:3d}      Running:{:3d}'.format(Pending, Running), curses.color_pair(0))
            stdscr.addstr(14, 5, '/home/:{:3.1f}GB    /scratch/:{:3.1f}GB'.format(
                float(home[0])/1000000,float(scratch[0])/1000000), curses.color_pair(0))
            stdscr.refresh()

            Inputs, Outputs, Completed, Converged = CheckGDB9()
            stdscr.addstr(18, 5, 'Inputs:{:3d}'.format(Inputs), curses.color_pair(0))
            stdscr.addstr(19, 5, 'Outputs:{:3d}'.format(Outputs), curses.color_pair(0))
            stdscr.addstr(20, 5, 'Completed:{:3d}'.format(Completed), curses.color_pair(0))
            stdscr.addstr(21, 5, 'Converged:{:3d}'.format(Converged), curses.color_pair(0))

            countdown = checkinterval

        stdscr.refresh()
        time.sleep(1)


def CheckDarwin():

    outp = subprocess.Popen(['ssh darwin squeue -u ' + user + ' | grep -c " R "'], shell=True, stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE).communicate()[0]
    Running = int(outp.decode().split('\n')[-2])

    outp = subprocess.Popen(['ssh darwin squeue -u ' + user + ' | grep -c " PD "'], shell=True, stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE).communicate()[0]
    Pending = int(outp.decode().split('\n')[-2])

    home = [0, 0]
    scratch = [0, 0]
    outp = subprocess.Popen(['ssh darwin quota'], shell=True, stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE).communicate()[0]
    outp = outp.decode().split('\n')
    for line in outp:
        if 'home' in line:
            data = list(filter(None, line.split(' ')))
            home = [float(data[1]), float(data[2])]
        if ('rds-d3' in line) and ('ke291' in line):
            data = list(filter(None, line.split(' ')))
            scratch = [float(data[1]), float(data[2])]

    return Pending, Running, home, scratch


def CheckZiggy():

    outp = subprocess.Popen(['ssh ziggy qstat -u ' + user + ' | grep -c " R "'], shell=True, stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE).communicate()[0]
    Running = int(outp.decode())

    outp = subprocess.Popen(['ssh ziggy qstat -u ' + user + ' | grep -c " Q "'], shell=True, stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE).communicate()[0]
    Pending = int(outp.decode())

    home = [0, 0]
    scratch = [0, 0]
    outp = subprocess.Popen(['ssh ziggy quota'], shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    outp = outp.decode().split('\n')
    for i, line in enumerate(outp):
        if 'HOME' in line:
            data = list(filter(None, outp[i+1].split(' ')))
            home = [float(data[0]), float(data[1])]
        if ('SHAREDSCRATCH' in line):
            data = list(filter(None, outp[i + 1].split(' ')))
            scratch = [float(data[0]), float(data[1])]

    return Pending, Running, home, scratch


def CheckGDB9():

    #Get GDB folders
    GDBroot = '/scratch/ke291/GDB9React'
    dirlist = os.listdir(GDBroot)
    GDBfolders = []
    for name in dirlist:
        if os.path.isdir(os.path.join(GDBroot, name)):
            GDBfolders.append(os.path.join(GDBroot, name))

    Inputs = []
    Outputs = []
    Completed = []
    Converged = []

    for folder in GDBfolders:
        outfilelist = glob.glob(folder + '/*.out')
        inp, outp, compl, conv = 0, 0, 0, 0
        if (not os.path.exists(os.path.join(folder, 'fstats'))) or (len(outfilelist) == 0):
            inp, outp, compl, conv = AnalyzeGDB9Folder(folder)
            f = open(os.path.join(folder, 'fstats'), 'w')
            f.write(','.join([str(inp), str(outp), str(compl), str(conv)]))
            f.close()
        elif os.path.getmtime(os.path.join(folder, 'fstats')) < max([os.path.getmtime(x) for x in outfilelist]):
            os.remove(os.path.join(folder, 'fstats'))
            inp, outp, compl, conv = AnalyzeGDB9Folder(folder)
            f = open(os.path.join(folder, 'fstats'), 'w')
            f.write(','.join([str(inp), str(outp), str(compl), str(conv)]))
            f.close()
        else:
            f = open(os.path.join(folder, 'fstats'), 'r')
            data = f.readlines()
            for line in data:
                if ',' in line:
                    [inp, outp, compl, conv] = [int(x) for x in line.split(',')]
            f.close()

        Inputs.append(inp)
        Outputs.append(outp)
        Completed.append(compl)
        Converged.append(conv)

    return sum(Inputs), sum(Outputs), sum(Completed), sum(Converged)


def AnalyzeGDB9Folder(folder, pattern='/*a'):

    outp = subprocess.Popen(['ls -l ' + folder + pattern + '.com | wc -l'], shell=True, stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE).communicate()[0]
    if outp.decode()[:-1].isnumeric():
        Inputs = int(outp.decode())
    else:
        Inputs = 0

    outp = subprocess.Popen(['ls -l ' + folder + pattern + '.out | wc -l'], shell=True, stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE).communicate()[0]
    if outp.decode()[:-1].isnumeric():
        Outputs = int(outp.decode())
    else:
        Outputs = 0

    outp = subprocess.Popen(['grep cpu ' + folder + pattern + '.out | wc -l'], shell=True, stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE).communicate()[0]
    if outp.decode()[:-1].isnumeric():
        Completed = int(outp.decode())
    else:
        Completed = 0

    outp = subprocess.Popen(['grep Normal ' + folder + pattern + '.out | wc -l'], shell=True, stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE).communicate()[0]
    if outp.decode()[:-1].isnumeric():
        Converged = int(outp.decode())
    else:
        Converged = 0

    return Inputs, Outputs, Completed, Converged


if __name__ == '__main__':
    curses.wrapper(main)
