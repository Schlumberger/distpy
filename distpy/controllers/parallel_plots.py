# (C) 2020, Schlumberger. Refer to LICENSE

########################################
# USER CONFIGURABLE VALUES:
# A system configuration to optimize performance
jsonSystemConfig = "{PATH}\\config\\systemConfig.json"
# A plot specification to draw the required figures
jsonConfig = "{PATH}\\config\\plots.json"
########################################
import argparse

import os
import numpy
from distpy.workers.plotgenerator import plotgenerator
import multiprocessing
import distpy.io_help.io_helpers as io_helpers


def main(configOuterFile):
    basedir,dirout,jsonConfig,PARALLEL,NCPU,BOX_SIZE, xaxisfile,taxisfile,prf = io_helpers.systemConfig(configOuterFile)
    
    xaxis = numpy.load(xaxisfile)

    configFile = io_helpers.json_io(jsonConfig,1)
    configData = configFile.read()

    #scan for directories
    datafiles=[]
    for root, dirs, files in os.walk(basedir):
        for datafile in files:
            if not datafile=='measured_depth.npy':
                datafiles.append(os.path.join(root,datafile))
        # break here because we don't want subdirectories (SLB case)
        break

    if not PARALLEL:
        # parallel does not work in Techlog...
        plotgenerator(basedir, dirout, configData)
    else:
        manager = multiprocessing.Manager()
        q = manager.Queue()
        print(multiprocessing.cpu_count())
        
        pool = multiprocessing.Pool(processes=NCPU, maxtasksperchild=1)

        jobs = []
        for datafile in datafiles:
            job = pool.apply_async(plotgenerator, [datafile, dirout, configData])
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
