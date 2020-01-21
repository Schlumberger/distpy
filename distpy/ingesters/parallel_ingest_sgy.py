# (C) 2020, Schlumberger. Refer to LICENSE

########################################
# USER CONFIGURABLE VALUES:
jsonSystemConfig = "{PATH}\\systemSGYConfig.json"
########################################
import argparse

import os
import distpy.io_help.sgy as sgy
import multiprocessing
import distpy.io_help.io_helpers as io_helpers
import distpy.io_help.directory_services as ds



def main(configOuterFile):
    # standard configuration information
    basedir,dirout,configFile,PARALLEL,NCPU,BOX_SIZE, xaxisfile,taxisfile,prf = io_helpers.systemConfig(configOuterFile)
    ds.makedir(dirout)

    print(basedir)
    print(dirout)
    #scan for directories
    sgy_files=[]
    for root, dirs, files in os.walk(basedir):
        for datafile in files:
            sgy_files.append(os.path.join(root,datafile))

    if not PARALLEL:
        # parallel does not work in Techlog...
        for fname in sgy_files:
            print(fname)
            sgy.SEGYingest(fname, dirout)
    else:
        manager = multiprocessing.Manager()
        q = manager.Queue()
        print(multiprocessing.cpu_count())
        
        pool = multiprocessing.Pool(processes=NCPU, maxtasksperchild=1)

        jobs = []
        for fname in sgy_files:
            job = pool.apply_async(sgy.SEGYingest, [fname,dirout])
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
