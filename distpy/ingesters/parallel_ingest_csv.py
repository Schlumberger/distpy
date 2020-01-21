# (C) 2020, Schlumberger. Refer to LICENSE

########################################
# USER CONFIGURABLE VALUES:
basedir = "{PATH}\\RawData"
dirout = "{PATH}\\backscatter"
PARALLEL=True
NCPU=1
########################################
import argparse

import os
import numpy
import distpy.io_help.io_helpers as io_helpers
import distpy.io_help.directory_services as ds
import multiprocessing

'''
 ingets1csv : dependencies are captured in this function rather than main()
              This keeps the main() as generic as possible so that a future
              refactoring might be able to standardize the parallel run controller.
'''
def ingest1csv(fname, dirout, jsonArgs):
    data, timestamp, jsonArgs = io_helpers.ingest_csv(fname,jsonArgs)
    fname=str(int(timestamp[0]))+'.npy'
    numpy.save(os.path.join(dirout,fname),data)
    numpy.save(os.path.join(dirout,'measured_depth.npy'),jsonArgs['xaxis'])
    numpy.save(os.path.join(dirout,'time_axis.npy'),jsonArgs['taxis'])
    


def main(configOuterFile):
    # standard configuration information
    basedir,dirout,jsonConfig,PARALLEL,NCPU,BOX_SIZE, xaxisfile,taxisfile,prf = io_helpers.systemConfig(configOuterFile)
    ds.makedir(dirout)

    configData = {}
    if jsonConfig is not None:
        configFile = io_helpers.json_io(jsonConfig,1)
        configData = configFile.read()
    #scan for directories
    csv_files=[]
    for root, dirs, files in os.walk(basedir):
        for datafile in files:
            csv_files.append(os.path.join(root,datafile))

    if not PARALLEL:
        # parallel does not work in Techlog...
        for fname in csv_files:
            print(fname)
            ingest1csv(fname,dirout, configData)
    else:
        manager = multiprocessing.Manager()
        q = manager.Queue()
        print(multiprocessing.cpu_count())

        pool = multiprocessing.Pool(processes=NCPU, maxtasksperchild=1)

        jobs = []
        for fname in csv_files:
            job = pool.apply_async(ingest1csv,[fname,dirout, configData])
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
