#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:56:54 2019

@author: ke291

Contains all of the specific code for running Gaussian jobs on Darwin.
A lot of code is reused from Gaussian.py. Called by GDB9.py.
"""

import subprocess
import os
import time
import shutil
import math

import Gaussian

MaxConcurrentJobs = 40       # these are CSD3 jobs (whole 32-node jobs), not individual Gaussian jobs

SetupOptCalcs = Gaussian.SetupOptCalcs

SetupFreqCalcs = Gaussian.SetupFreqCalcs

ReadEnergies = Gaussian.ReadEnergies

ReadGeometries = Gaussian.ReadGeometries

IsGausCompleted = Gaussian.IsGausCompleted

Converged = Gaussian.Converged


def RunOptCalcs(Isomers, settings):
    print('\nRunning Gaussian DFT geometry optimizations on Darwin...')

    GausJobs = []

    for iso in Isomers:
        GausJobs.extend([x for x in iso.OptInputFiles if (x[:-4] + '.out') not in iso.OptOutputFiles])

    Completed = RunCalcs(GausJobs, settings)

    for iso in Isomers:
        iso.OptOutputFiles.extend([x[:-4] + '.out' for x in iso.OptInputFiles if (x[:-4] + '.out') in Completed])

    return Isomers


def RunFreqCalcs(Isomers, settings):

    print('\nRunning Gaussian frequency calculations on Darwin...')

    GausJobs = []

    for iso in Isomers:
        GausJobs.extend([x for x in iso.FreqInputFiles if (x[:-4] + '.out') not in iso.FreqOutputFiles])

    Completed = RunCalcs(GausJobs, settings)

    for iso in Isomers:
        iso.FreqOutputFiles.extend([x[:-4] + '.out' for x in iso.FreqInputFiles if (x[:-4] + '.out') in Completed])

    return Isomers


def RunCalcs(GausJobs, settings):

    MaxCon = MaxConcurrentJobs

    Jobs2Run = list(GausJobs)   # full list of all jobs that still need to be run, list of com files
    JobsSubmitted = []           # jobs that have been submitted to the cluster and need to be cleaned up afterwards
    GausSets = []                # a list of lists of com files corresponding to job IDs in JobsSubmitted
    nJobsRunning = 0            # number of jobs that are in queue waiting to be run or running
    NCompleted = 0              # number of jobs completed
    Completed = []          # jobs that are completed, list of com files

    OldQRes = [[], [], []]  # result from queue checking, list of 3 lists - pending, running and absent jobs

    folder = settings.StartTime + settings.Title
    scrfolder = settings.StartTime + settings.Title + 's'

    # Check that results folder does not exist, create job folder on darwin
    outp = subprocess.Popen(['ssh', 'darwin', 'ls ' + settings.DarwinScrDir],
                            stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    print("Results folder: " + folder)

    if folder in outp.decode():
        print("Results folder exists on Darwin, choose another folder name.")
        quit()

    outp = subprocess.Popen(['ssh', 'darwin', 'mkdir', settings.DarwinScrDir + folder],
                            stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]

    # Check that scratch directory does not exist, create scratch folder on darwin
    outp = subprocess.Popen(['ssh', 'darwin', 'ls ' + settings.DarwinScrDir],
                            stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    print("Scratch directory: " + settings.DarwinScrDir + scrfolder)

    if scrfolder in outp.decode():
        print("Scratch folder exists on Darwin, choose another folder name.")
        quit()

    outp = subprocess.Popen(['ssh', 'darwin', 'mkdir', settings.DarwinScrDir + scrfolder],
                            stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]

    #Main loop - periodically check queue, add more jobs if there is room
    #continue until all jobs have been added to the queue and run
    while (len(Jobs2Run) > 0) or (nJobsRunning > 0):

        if (nJobsRunning < MaxCon * 0.9) and (len(Jobs2Run) > 0):
            #Submits jobs, removes them from Jobs2Run, adds them to JobsSubmitted
            nDarwinJobs = MaxCon - nJobsRunning
            Jobs2Run, JobsSubmitted, GausSets = SubmitJobs(Jobs2Run, JobsSubmitted, GausSets, nDarwinJobs, settings)

        time.sleep(180)

        QRes = CheckDarwinQueue(JobsSubmitted, settings)

        if OldQRes != QRes:
            print('Pending: ' + str(len(QRes[0])) + ', Running: ' + str(len(QRes[1])) + ', Not in queue: ' + str(
                len(QRes[2])))
            OldQRes = QRes

        nJobsRunning = len(QRes[0]) + len(QRes[1])

        #Checks every job in JobsSubmitted - if in NotInQueue, downloads it. Returns JobsSubmitted without finished jobs
        JobsSubmitted, GausSets, NewCompleted = FinishJobs(JobsSubmitted, GausSets, QRes[2], settings)

        Completed.extend(NewCompleted)
        NCompleted += len(NewCompleted)
        if len(NewCompleted) > 0:
            print('{:d}/{:d} ({:.1f}%) jobs done'.format(NCompleted, len(GausJobs), NCompleted*100/len(GausJobs)))

        time.sleep(180)

    if NCompleted > 0:
        fullfolder = settings.DarwinScrDir + folder
        print("\nDeleting checkpoint files...")
        print('ssh darwin rm ' + fullfolder + '/*.chk')
        outp = subprocess.Popen(['ssh', 'darwin', 'rm', fullfolder + '/*.chk'],
                                stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]

        fullscrfolder = settings.DarwinScrDir + scrfolder
        print("\nDeleting scratch folder...")
        print('ssh darwin rm -r ' + fullscrfolder)

        outp = subprocess.Popen(['ssh', 'darwin', 'rm -r', fullscrfolder],
                                stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]

        print(str(NCompleted) + " Gaussian jobs of " + str(len(GausJobs)) + \
            " completed successfully.")
    elif len(GausJobs) == 0:
        print("There were no jobs to run.")

    return Completed

#Submits jobs, removes them from Jobs2Run, adds them to JobsSubmitted
#nJobs - how many new Darwin jobs to submit
#The number of Gaussian jobs submitted will be nJobs*DarwinNodesize/nProcs
# or all of Jobs2Run, whichever is smaller
# returns remaining (unsubmitted) Jobs2Run, and slurm job ids for the submitted jobs
def SubmitJobs(Jobs2Run, PrevJobsSubmitted, PrevGausSets, nJobs, settings):

    nGausJobs = min(int(nJobs*settings.DarwinNodeSize/settings.nProc), len(Jobs2Run))
    GausFiles = Jobs2Run[:nGausJobs]
    folder = settings.StartTime + settings.Title

    #Write the slurm scripts
    SubFiles, GausSets = WriteDarwinScripts(GausFiles, settings)

    print(str(len(SubFiles)) + ' slurm scripts generated')

    #Upload .com files and slurm files to directory
    print("Uploading files to darwin...")

    filelist = open('filelist', 'w')
    filelist.write('\n'.join(GausFiles))
    filelist.write('\n')

    filelist.write('\n'.join(SubFiles))
    filelist.write('\n')
    filelist.close()

    outp = subprocess.Popen(['rsync', '-a', '--files-from=filelist', '.', 'darwin:' + settings.DarwinScrDir + folder + '/'], stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE).communicate()[0]
    print(' '.join(['rsync', '-a', '--files-from=filelist', '.', 'darwin:' + settings.DarwinScrDir + folder + '/']))
    print(str(len(GausFiles)) + ' .com and ' + str(len(SubFiles)) +\
        ' slurm files uploaded to darwin')

    print(str(len(GausFiles)) + ' .com and slurm files uploaded to CSD3')

    JobsSubmitted = []
    #Launch the calculations
    fullfolder = settings.DarwinScrDir + folder

    # Launch the calculations
    for f in SubFiles:
        outp = subprocess.Popen(['ssh', 'darwin', 'cd ' + fullfolder + ';sbatch', f], \
                                stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
        status = outp.decode().split('\n')[-2]
        print(status)
        JobsSubmitted.append(status.split('job ')[1])

    print(str(len(SubFiles)) + ' jobs added to the queue on CSD3 ' + \
          'containing ' + str(len(GausFiles)) + ' Gaussian jobs,\n in addition to the ' +
          str(len(PrevJobsSubmitted)) + ' existing jobs in queue')

    return Jobs2Run[nGausJobs:], JobsSubmitted + PrevJobsSubmitted, GausSets + PrevGausSets

# Make a list of files to download, rsync them back from CSD3
def FinishJobs(JobsSubmitted, GausSets, NotInQueue, settings):
    folder = settings.StartTime + settings.Title
    RemainingJobs = []
    RemainingSets = []
    Completed = []

    OutFiles = []

    for i, jobid in enumerate(JobsSubmitted):

        if jobid in NotInQueue:
            # If job done, add outputfile to the download list
            OutFiles.extend([x[:-4] + '.out' for x in GausSets[i]])
        else:
            RemainingJobs.append(jobid)
            RemainingSets.append(GausSets[i])

    if len(NotInQueue) > 0:
        # Download .out files
        filelist = open('downlist', 'w')
        filelist.write('\n'.join(OutFiles))
        filelist.write('\n')
        filelist.close()

        outp = subprocess.Popen(
            ['rsync', '-a', '--files-from=downlist', 'darwin:' + settings.DarwinScrDir + folder + '/','.'],
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE).communicate()[0]
        print(' '.join(['rsync', '-a', '--files-from=downlist', 'darwin:' + settings.DarwinScrDir + folder + '/', '.']))

        for f in OutFiles:
            if IsGausCompleted(f[:-4] + '.out'):
                Completed.append(f[:-4] + '.out')

        print(str(len(OutFiles)) + " .out files downloaded, " + str(len(Completed)) + " successfully completed")

    return RemainingJobs, RemainingSets, Completed


def RunBatchOnDarwin(findex, GausJobs, settings):

    if findex == 0:
        folder = settings.StartTime + settings.Title
    else:
        folder = settings.StartTime + findex + settings.Title

    scrfolder = settings.StartTime + settings.Title + 's'

    #Check that results folder does not exist, create job folder on darwin
    outp = subprocess.Popen(['ssh', 'darwin', 'ls ' + settings.DarwinScrDir], \
      stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    print("Results folder: " + folder)
    
    if folder in outp.decode():
        print("Results folder exists on Darwin, choose another folder name.")
        quit()

    outp = subprocess.Popen(['ssh', 'darwin', 'mkdir', settings.DarwinScrDir + folder], \
      stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]

    # Check that scratch directory does not exist, create scratch folder on darwin
    outp = subprocess.Popen(['ssh', 'darwin', 'ls ' + settings.DarwinScrDir], \
                            stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    print("Scratch directory: " + settings.DarwinScrDir + scrfolder)

    if scrfolder in outp.decode():
        print("Scratch folder exists on Darwin, choose another folder name.")
        quit()

    outp = subprocess.Popen(['ssh', 'darwin', 'mkdir', settings.DarwinScrDir + scrfolder], \
                            stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]

    #Write the slurm scripts
    SubFiles = WriteDarwinScripts(GausJobs, settings, scrfolder)
        
    print(str(len(SubFiles)) + ' slurm scripts generated')

    #Upload .com files and slurm files to directory
    print("Uploading files to darwin...")
    filelists = []
    for i in range(math.ceil(len(GausJobs)/32)):
        filelists.append(' '.join(GausJobs[i*32:(i+1)*32]))

    filelist = open('filelist', 'w')
    filelist.write('\n'.join(GausJobs))
    filelist.write('\n')

    filelist.write('\n'.join(SubFiles))
    filelist.write('\n')
    filelist.close()

    outp = subprocess.Popen(['rsync', '-a', '--files-from=filelist', '.', 'darwin:' + settings.DarwinScrDir + folder + '/'], stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE).communicate()[0]
    print(' '.join(['rsync', '-a', '--files-from=filelist', '.', 'darwin:' + settings.DarwinScrDir + folder + '/']))
    print(str(len(GausJobs)) + ' .com and ' + str(len(SubFiles)) +\
        ' slurm files uploaded to darwin')

    fullfolder = settings.DarwinScrDir + folder
    JobIDs = []

    #Launch the calculations
    for f in SubFiles:
        outp = subprocess.Popen(['ssh', 'darwin', 'cd ' + fullfolder + ';sbatch', f], \
            stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
        status = outp.decode().split('\n')[-2]
        print(status)
        JobIDs.append(status.split('job ')[1])

    print(str(len(SubFiles)) + ' jobs submitted to the queue on darwin ' + \
        'containing ' + str(len(GausJobs)) + ' Gaussian jobs')

    time.sleep(60)

    OldQRes = CheckDarwinQueue(JobIDs, settings)

    while OldQRes[0] < 0:
        OldQRes = CheckDarwinQueue(JobIDs, settings)
        time.sleep(60)

    print('Pending: ' + str(OldQRes[0]) + ', Running: ' + str(OldQRes[1]) + ', Not in queue: ' + str(OldQRes[2]))

    TotalJobs = len(list(GausJobs))
    n2complete = TotalJobs
    
    #Check and report on the progress of calculations
    while n2complete > 0:
        JobsFinished = IsDarwinGComplete(folder, settings)
        
        JobsRemaining = TotalJobs - JobsFinished
        if n2complete != JobsRemaining:
            n2complete = JobsRemaining
            print(str(n2complete) + " Gaussian jobs remaining.")

        QRes = CheckDarwinQueue(JobIDs, settings)
        if QRes != OldQRes:
            if QRes[0] < 0:
                QRes = OldQRes
            else:
                OldQRes = QRes
                print('Darwin queue:')
                print('Pending: ' + str(OldQRes[0]) + ', Running: ' + str(OldQRes[1]) + ', Not in queue: ' + str(OldQRes[2]))


        if (QRes[2] == len(JobIDs)) and (QRes[0] >= 0):
            #check each gaussian file to ascertain the status of individual gaus jobs
            print('No jobs left in Darwin queue')
            break

        time.sleep(300)

    #When done, copy the results back
    print("\nCopying the output files back to localhost...")
    print('scp darwin:' + fullfolder + '/*.out ' + os.getcwd() + '/')

    outp = subprocess.Popen(['scp', 'darwin:' + fullfolder + '/*.out',
            os.getcwd() + '/'], \
            stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]

    print("\nDeleting checkpoint files...")
    print('ssh darwin rm ' + fullfolder + '/*.chk')
    outp = subprocess.Popen(['ssh', 'darwin', 'rm', fullfolder + '/*.chk'], \
                            stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]

    fullscrfolder = settings.DarwinScrDir + scrfolder
    print("\nDeleting scratch folder...")
    print('ssh darwin rm -r ' + fullscrfolder)

    outp = subprocess.Popen(['ssh', 'darwin', 'rm -r', fullscrfolder], \
            stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    

def WriteDarwinScripts(GausJobs, settings):

    SubFiles = []
    GausSets = []   # list of file lists storing which gaussian inputs are in which job
    AdjNodeSize = int(math.floor(settings.DarwinNodeSize/settings.nProc))
    scrfolder = settings.StartTime + settings.Title + 's'

    # If the jobs exactly fill the node, just write the submission script
    if len(GausJobs) == AdjNodeSize:
        SubFiles.append(WriteSlurm(GausJobs, settings, scrfolder))
        GausSets.append(GausJobs)

    # If the jobs do not fill the node, increase the processor count to fill it
    elif len(GausJobs) < AdjNodeSize:
        NewNProc = int(math.floor(settings.DarwinNodeSize / len(GausJobs)))
        SubFiles.append(WriteSlurm(GausJobs, settings, scrfolder, nProc=NewNProc))
        print("Jobs don't fill the Darwin node, nproc increased to " + str(NewNProc))
        for j, GausJob in enumerate(GausJobs):
            line = '%nprocshared=' + str(NewNProc) + '\n'
            ReplaceLine(GausJob, 0, line)
        GausSets.append(GausJobs)

    # If the jobs more than fill the node, submit as several jobs
    else:
        print("The Gaussian calculations will be submitted as " +\
                    str(math.ceil(len(GausJobs)/AdjNodeSize)) + \
                    " jobs")
        i = 0
        while (i+1)*AdjNodeSize < len(GausJobs):
            PartGausJobs = list(GausJobs[(i*AdjNodeSize):((i+1)*AdjNodeSize)])
            print("Writing script nr " + str(i+1))
            SubFiles.append(WriteSlurm(PartGausJobs, settings, scrfolder, str(i+1)))
            GausSets.append(PartGausJobs)
            
            i += 1
        
        PartGausJobs = list(GausJobs[(i*AdjNodeSize):])

        # if the last few jobs do not fill the node, increase the processor count to fill it
        if len(PartGausJobs) < AdjNodeSize:
            NewNProc = int(math.floor(settings.DarwinNodeSize / len(PartGausJobs)))
            print("Jobs don't fill the last Darwin node, nproc increased to " + str(NewNProc))
            print("Writing script nr " + str(i + 1))
            SubFiles.append(WriteSlurm(PartGausJobs, settings, scrfolder, str(i+1), nProc=NewNProc))
            GausSets.append(PartGausJobs)
            for j, GausJob in enumerate(PartGausJobs):
                line = '%nprocshared=' + str(NewNProc) + '\n'
                ReplaceLine(GausJob, 0, line)
        else:
            print("Writing script nr " + str(i + 1))
            SubFiles.append(WriteSlurm(PartGausJobs, settings, scrfolder, str(i+1)))
            GausSets.append(PartGausJobs)

    return SubFiles, GausSets


def WriteSlurm(GausJobs, settings, scrfolder, index='', nProc = -1):

    if nProc == -1:
        nProc = settings.nProc

    cwd = os.getcwd()
    #filename = settings.Title + 'slurm' + index
    filename = GausJobs[0][:-4] + 'slurm'
    scrsubfolder = settings.DarwinScrDir + scrfolder + '/' + GausJobs[0][:-4]
    
    shutil.copyfile(settings.ScriptDir + '/Defaultslurm',
                    cwd + '/' + filename)
    slurmf = open(filename, 'r+')
    slurm = slurmf.readlines()
    slurm[12] = '#SBATCH -J ' + settings.Title + '\n'
    slurm[14] = '#SBATCH -A ' + settings.project + '\n'
    slurm[19] = '#SBATCH --ntasks=' + str(len(GausJobs)*nProc) + '\n'
    slurm[21] = '#SBATCH --time=' + format(settings.TimeLimit,"02") +\
        ':00:00\n'
    slurm[59] = 'mkdir ' + scrsubfolder + '\n'
    slurm[61] = 'export GAUSS_SCRDIR=' + scrsubfolder + '\n'
    
    for f in GausJobs:
        slurm.append('srun --exclusive -n1 -c' + str(nProc) + ' $application < ' + f[:-3] + \
            'com > ' + f[:-3] + 'out 2> ' + f[:-3] + '.err &\n')

    slurm.append('wait\n')

    slurmf.truncate(0)
    slurmf.seek(0)
    slurmf.writelines(slurm)
    
    return filename


def IsDarwinGComplete(folder, settings):

    path = settings.DarwinScrDir + folder + '/'
    results = {}
    outp = subprocess.Popen('ssh darwin grep Normal ' + path + '*out | wc -l', shell=True,
                            stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    JobsComplete = int(outp.decode().split('\n')[-2])

    return JobsComplete


def CheckDarwinQueue(JobIDs, settings):

    outp = subprocess.Popen(['ssh', 'darwin', 'squeue', '-u ' + settings.user], \
                            stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    outp = outp.decode().split('\n')
    QStart = -1000
    for i, line in enumerate(outp):
        if 'JOBID' in line:
            QStart = i+1
            break

    if QStart < 0:
        return -100, -100, -100

    QueueReport = outp[QStart:-1]
    JobStats = []

    PendingJobs = []
    RunningJobs = []
    NotInQueueJobs = []

    for job in JobIDs:
        status = ''
        for i, line in enumerate(QueueReport):
            if job in line:
                status = list(filter(None, line.split(' ')))[4]
        JobStats.append(status)
        if status == 'PD':
            PendingJobs.append(job)
        if status =='R':
            RunningJobs.append(job)
        if status == '':
            NotInQueueJobs.append(job)

    return PendingJobs, RunningJobs, NotInQueueJobs


def ReplaceLine(File, LineN, Line):
    gausf = open(File, 'r+')
    gauslines = gausf.readlines()
    gauslines[LineN] = Line
    gausf.truncate(0)
    gausf.seek(0)
    gausf.writelines(gauslines)
    gausf.close()
