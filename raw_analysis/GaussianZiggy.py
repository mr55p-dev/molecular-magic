#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:56:54 2019

@author: ke291

Contains all of the specific code for running Gaussian jobs on Ziggy.
A lot of code is reused from Gaussian.py. Called by PyDP4.py.
"""

import subprocess
import socket
import os
import time
import math

import Gaussian

MaxConcurrentJobsZiggy = 300

SetupOptCalcs = Gaussian.SetupOptCalcs

SetupFreqCalcs = Gaussian.SetupFreqCalcs

ReadEnergies = Gaussian.ReadEnergies

ReadGeometries = Gaussian.ReadGeometries

IsGausCompleted = Gaussian.IsGausCompleted

Converged = Gaussian.Converged


def RunOptCalcs(Isomers, settings):
    print('\nRunning Gaussian DFT geometry optimizations on Ziggy...')

    GausJobs = []

    for iso in Isomers:
        GausJobs.extend([x for x in iso.OptInputFiles if (x[:-4] + '.out') not in iso.OptOutputFiles])

    Completed = RunCalcs(GausJobs, settings)

    for iso in Isomers:
        iso.OptOutputFiles.extend([x[:-4] + '.out' for x in iso.OptInputFiles if (x[:-4] + '.out') in Completed])

    return Isomers


def RunFreqCalcs(Isomers, settings):

    print('\nRunning Gaussian frequency calculations on Ziggy...')

    GausJobs = []

    for iso in Isomers:
        GausJobs.extend([x for x in iso.FreqInputFiles if (x[:-4] + '.out') not in iso.FreqOutputFiles])

    Completed = RunCalcs(GausJobs, settings)

    for iso in Isomers:
        iso.FreqOutputFiles.extend([x[:-4] + '.out' for x in iso.FreqInputFiles if (x[:-4] + '.out') in Completed])

    return Isomers


def RunCalcs(GausJobs, settings):

    MaxCon = MaxConcurrentJobsZiggy
    Jobs2Run = list(GausJobs)   # full list of all jobs that still need to be run, list of com files
    JobsSubmitted = []           # jobs that have been submitted to the cluster and need to be cleaned up afterwards
                                 # a list of com file and slurm job ID pairs
    nJobsRunning = 0            # number of jobs that are in queue waiting to be run or running
    NCompleted = 0              # number of jobs completed
    Completed = []          # jobs that are completed, list of com files

    OldQRes = [[],[],[]]           # result from queue checking, list of 3 lists - pending, running and absent jobs

    folder = settings.StartTime + settings.Title

    print("Running " + str(len(GausJobs)) + " jobs on ziggy...\n")

    #Check that folder does not exist, create job folder on ziggy
    outp = subprocess.check_output('ssh ziggy ls', shell=True)
    if folder in outp.decode():
        print("Folder exists on ziggy, choose another folder name.")
        quit()

    outp = subprocess.check_output('ssh ziggy mkdir ' + folder, shell=True)

    print("Results folder: " + folder)

    #Main loop - periodically check queue, add more jobs if there is room
    #continue until all jobs have been added to the queue and run
    while (len(Jobs2Run) > 0) or (nJobsRunning > 0):

        if (nJobsRunning < MaxCon * 0.9) and (len(Jobs2Run) > 0):
            #Submits jobs, removes them from Jobs2Run, adds them to JobsSubmitted
            Jobs2Run, JobsSubmitted = SubmitJobs(Jobs2Run, JobsSubmitted, MaxCon - nJobsRunning, settings)

        time.sleep(180)
        #Gets a queue status for all jobs in JobsSubmitted
        QRes = CheckZiggyQueue(JobsSubmitted, settings.user)
        if OldQRes != QRes:
            print('Pending: ' + str(len(QRes[0])) + ', Running: ' + str(len(QRes[1])) + ', Not in queue: ' + str(len(QRes[2])))
            OldQRes = QRes
            
        nJobsRunning = len(QRes[0]) + len(QRes[1])

        #Checks every job in JobsSubmitted - if in NotInQueue, downloads it. Returns JobsSubmitted without finished jobs
        JobsSubmitted, NewCompleted = FinishJobs(JobsSubmitted, QRes[2], settings)
        Completed.extend(NewCompleted)
        NCompleted += len(NewCompleted)
        if len(NewCompleted) > 0:
            print('{:d}/{:d} ({:.1f}%) jobs done'.format(NCompleted, len(GausJobs), NCompleted*100/len(GausJobs)))

    if NCompleted > 0:
        print(str(NCompleted) + " Gaussian jobs of " + str(len(GausJobs)) + \
            " completed successfully.")
    elif len(GausJobs) == 0:
        print("There were no jobs to run.")

    return Completed


def SubmitJobs(Jobs2Run, PrevJobsSubmitted, nJobs, settings):

    GausFiles = Jobs2Run[:nJobs]
    folder = settings.StartTime + settings.Title

    #Write the qsub scripts
    for f in GausFiles:
        WriteSubScript(f, settings.queue, folder, settings)

    #Upload .com files and .qsub files to directory
    for f in GausFiles:
        outp = subprocess.check_output('scp ' + f +' ziggy:~/' + folder,
                                           shell=True)
        outp = subprocess.check_output('scp ' + f[:-4] +'slurm ziggy:~/' +
                                       folder, shell=True)

    print(str(len(GausFiles)) + ' .com and slurm files uploaded to ziggy')

    JobsSubmitted = []
    #Launch the calculations
    for f in GausFiles:
        job = '~/' + folder + '/' + f[:-4]
        outp = subprocess.check_output('ssh ziggy sbatch ' + job + 'slurm', shell=True)
        status = outp.decode()[:-1]
        JobsSubmitted.append([f, status.split('job ')[1]])

    print(str(len(GausFiles)) + ' jobs added to the ' + str(len(PrevJobsSubmitted)) + ' existing jobs in queue')

    return Jobs2Run[nJobs:], JobsSubmitted + PrevJobsSubmitted


def FinishJobs(JobsSubmitted, NotInQueue, settings):

    folder = settings.StartTime + settings.Title
    RemainingJobs = []
    Completed = []
        
    for (gausfile, jobid) in JobsSubmitted:
        
        if [gausfile, jobid] in NotInQueue:
            #When done, copy the results back
            try:
                outp = subprocess.check_output('scp ziggy:/home/' + settings.user +
                                           '/' + folder + '/' + gausfile[:-4] + '.out ' + socket.getfqdn()
                                           + ':' + os.getcwd(), shell=True)
            except subprocess.CalledProcessError as e:
                print('File ' + gausfile[:-4] + '.out not found in the job folder.')
        else:
            RemainingJobs.append([gausfile, jobid])

    if len(NotInQueue) > 0:
        for (f, jobid) in NotInQueue:
            if IsGausCompleted(f[:-4] + '.out'):
                Completed.append(f[:-4] + '.out')

        print(str(len(NotInQueue)) + " .out files downloaded, " + str(len(Completed)) + " successfully completed")

    return RemainingJobs, Completed


# JobIDs - list of gaus input filename and ziggy job number pairs
def CheckZiggyQueue(JobIDs, user):

    outp = subprocess.Popen(['ssh', 'ziggy', 'qstat', '-u ' + user], \
                            stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    outp = outp.decode().split('\n')

    QStart = 0
    for i, line in enumerate(outp):
        if '------' in line:
            QStart = i+1
            break
    QueueReport = outp[QStart:-1]

    JobStats = []

    PendingJobs = []
    RunningJobs = []
    NotInQueueJobs = []

    for job in JobIDs:
        status = ''
        for i, line in enumerate(QueueReport):
            if job[1] in line:
                status = list(filter(None, line.split(' ')))[9]
        JobStats.append(status)
        if status == 'Q':
            PendingJobs.append(job)
        if status =='R':
            RunningJobs.append(job)
        if status == '':
            NotInQueueJobs.append(job)

    return PendingJobs, RunningJobs, NotInQueueJobs


def WriteSubScript(GausJob, queue, ZiggyJobFolder, settings):

    if not (os.path.exists(GausJob)):
        print("The input file " + GausJob + " does not exist. Exiting...")
        return

    #Create the submission script
    QSub = open(GausJob[:-4] + "slurm", 'w')

    #Choose the queue
    QSub.write('#!/bin/bash\n\n')
    QSub.write('#SBATCH -p ' + settings.queue + '\n')
    if settings.nProc >1:
        QSub.write('#SBATCH --nodes=1\n#SBATCH --cpus-per-task=' + str(settings.nProc) + '\n')
    else:
        QSub.write('#SBATCH --nodes=1\n#SBATCH --cpus-per-task=1\n')
    QSub.write('#SBATCH --time=' + format(settings.TimeLimit,"02") +':00:00\n\n')

    #define input files and output files
    QSub.write('file=' + GausJob[:-4] + '\n\n')
    QSub.write('inpfile=${file}.com\noutfile=${file}.out\n')

    #define cwd and scratch folder and ask the machine
    #to make it before running the job
    QSub.write('HERE=/home/' + settings.user +'/' + ZiggyJobFolder + '\n')
    QSub.write('SCRATCH=/scratch/' + settings.user + '/' +
               GausJob[:-4] + '\n')
    QSub.write('mkdir ${SCRATCH}\n')

    #Setup GAUSSIAN environment variables
    QSub.write('set OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n')
    
    QSub.write('export GAUSS_EXEDIR=/usr/local/shared/gaussian/em64t/09-D01/g09\n')
    QSub.write('export g09root=/usr/local/shared/gaussian/em64t/09-D01\n')
    QSub.write('export PATH=/usr/local/shared/gaussian/em64t/09-D01/g09:$PATH\n')
    QSub.write('export GAUSS_SCRDIR=$SCRATCH\n')
    QSub.write('exe=$GAUSS_EXEDIR/g09\n')
    #copy the input file to scratch
    QSub.write('cp ${HERE}/${inpfile}  $SCRATCH\ncd $SCRATCH\n')

    #write useful info to the job output file (not the gaussian)
    QSub.write('echo "Starting job $SLURM_JOBID"\necho\n')
    QSub.write('echo "SLURM assigned me this node:"\nsrun hostname\necho\n')

    QSub.write('ln -s $HERE/$outfile $SCRATCH/$outfile\n')
    QSub.write('srun $exe > $outfile < $inpfile\n')

    #Cleanup
    QSub.write('rm -rf ${SCRATCH}/\n')
    
    QSub.close()

