# (C) 2020, Schlumberger. Refer to LICENSE

########################################
# USER CONFIGURABLE VALUES:
jsonSystemConfig = "{PATH}\\systemh5Config.json"
########################################
import argparse

import os
import multiprocessing
import distpy.io_help.io_helpers as io_helpers
import distpy.io_help.h5_helpers as h5_helpers
import distpy.io_help.directory_services as ds

def main(configOuterFile,ingester=h5_helpers):
    # standard configuration information
    basedir,dirout,configFile,PARALLEL,NCPU,BOX_SIZE, xaxisfile,taxisfile,prf = io_helpers.systemConfig(configOuterFile)
    ds.makedir(dirout)

    print(basedir)
    print(dirout)
    #scan for directories
    h5_files=[]
    for root, dirs, files in os.walk(basedir):
        for datafile in files:
            h5_files.append(os.path.join(root,datafile))

    if not PARALLEL:
        for fname in h5_files:
            print(fname)
            ingester.ingest(fname, dirout)
    else:
        manager = multiprocessing.Manager()
        q = manager.Queue()
        print(multiprocessing.cpu_count())
        
        pool = multiprocessing.Pool(processes=NCPU, maxtasksperchild=1)

        jobs = []
        for fname in h5_files:
            job = pool.apply_async(ingester.ingest, [fname,dirout])
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
