# (C) 2020, Schlumberger. Refer to LICENSE

########################################
# USER CONFIGURABLE VALUES:
basedir = "{PATH}\\witsml"
dirout = "{PATH}\\results"
PARALLEL=True
NCPU=1
repackage_size=10
########################################
import argparse

import os
import distpy.io_help.witsmlfbe as witsmlfbe
import distpy.io_help.io_helpers as io_helpers
import distpy.io_help.directory_services as ds
import multiprocessing

SUFFIX = ['fbe','dts','xml']

'''
 readFBEController : dependencies are captured here to keep the main() as
                     generic as possible. In the future refactoring might
                     result in one parallel controller.
'''
def readFBEController(dirin,dirout, repackage_size):
    datafiles=[]
    prefixes = []
    for root, dirs, files in os.walk(dirin):
        for datafile in files:
            if datafile[-3:] in SUFFIX:
                prefixes.append(datafile[:-4])
                datafiles.append(os.path.join(root,datafile))
    # alphanumeric sorting is assumed to work...
    # a more detail-oriented approach would extract all the unixtime stamps in key-value pairs and sort on those.
    datafiles.sort()
    nt=len(datafiles)
    if (repackage_size<1):
        repackage_size=nt
    # repackage size is now only zero if there are no fbe files found..
    if (repackage_size>1):
        for a in range(0,nt,repackage_size):
            endIdx = a+repackage_size
            if endIdx>=nt:
                endIdx=nt-1
            datafileset = datafiles[a:endIdx]
            witsmlfbe.readFBE(datafileset,dirout,prefixes[a])
        
def main(configOuterFile):
    # standard configuration information
    basedir,dirout,configFile,PARALLEL,NCPU,repackage_size, xaxisfile,taxisfile,prf = io_helpers.systemConfig(configOuterFile)

    #scan for directories
    witsml_dirs=[]
    results_dirs=[]
    for root, dirs, files in os.walk(basedir):
        for resultdir in dirs:
            print(resultdir)
            # NOTE we handle the 1 exception directly.
            # in the future something more general may be needed
            if not resultdir=='png':
                witsml_dirs.append(os.path.join(root,resultdir))
                onedirout = os.path.join(dirout,resultdir)
                results_dirs.append(onedirout)
                ds.makedir(onedirout)

    # packaging the WITSML into chunks - in case there were days and days of DAS recording so that
    # the data are still too big to handle...
    if not PARALLEL:
        # parallel does not work in Techlog...
        for (witsmldir,resultdir) in zip(witsml_dirs,results_dirs):
            print(resultdir)            
            readFBEController(witsmldir,resultdir,repackage_size)
    else:
        manager = multiprocessing.Manager()
        q = manager.Queue()
        print(multiprocessing.cpu_count())
        
        pool = multiprocessing.Pool(processes=NCPU, maxtasksperchild=1)
        
        jobs = []
        for (witsmldir,resultdir) in zip(witsml_dirs,results_dirs):
            job = pool.apply_async(readFBEController, [witsmldir,resultdir,repackage_size])
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

