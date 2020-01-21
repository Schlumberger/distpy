# (C) 2020, Schlumberger. Refer to LICENSE

########################################
# USER CONFIGURABLE VALUES:
# A system configuration to optimize performance
jsonSystemConfig = "{PATH}\\config\\systemConfig.json"
# A signal processing workflow configuration
jsonConfig = "{PATH}\\config\\strainrate2summary.json"
########################################
import argparse

import os
import numpy
from distpy.workers.strainrate2summary import strainrate2summary
import multiprocessing
import distpy.io_help.io_helpers as io_helpers



def main(configOuterFile):
    basedir,dirout,jsonConfig,PARALLEL,NCPU,BOX_SIZE, xaxisfile,taxisfile,prf = io_helpers.systemConfig(configOuterFile)

    xaxis = None
    try:
        xaxis = numpy.load(xaxisfile)
    except FileNotFoundError:
        print('No x-axis specified')

    configFile = io_helpers.json_io(jsonConfig,1)
    configData = configFile.read()
    # Note this copy - we don't (from a user point-of-view)
    # want to be doing system configuration in the strainrate2summary.json,
    # but we do from an internal point-of-view
    configData['BOX_SIZE']=BOX_SIZE
    print(configData)

    #scan for directories
    datafiles=[]
    for root, dirs, files in os.walk(basedir):
        for datafile in files:
            if not datafile=='measured_depth.npy':
                if not datafile=='time_axis.npy':
                    datafiles.append(os.path.join(root,datafile))
        # break here because we don't want subdirectories (SLB case)
        break

    if not PARALLEL:
        # parallel does not work in Techlog...
        for datafile in datafiles:
            print(datafile)
            strainrate2summary(datafile, xaxis, prf, dirout, configData)
    else:
        manager = multiprocessing.Manager()
        q = manager.Queue()
        print(multiprocessing.cpu_count())
        
        pool = multiprocessing.Pool(processes=NCPU, maxtasksperchild=1)


        jobs = []
        for datafile in datafiles:
            job = pool.apply_async(strainrate2summary, [datafile, xaxis, prf,dirout, configData])
            print(job)
            jobs.append(job)

        for job in jobs:
            job.get()
            print(job)

        q.put('kill')
        pool.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--filename", help='A configuration file')
    args = parser.parse_args()
    main(args.filename)
