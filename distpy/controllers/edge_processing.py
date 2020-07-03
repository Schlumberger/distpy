# (C) 2020, Schlumberger. Refer to LICENSE

########################################
# USER CONFIGURABLE VALUES:
jsonSystemConfig = "{PATH}\\systemh5Config.json"
########################################
import argparse

import numpy

import os
import multiprocessing
import distpy.io_help.io_helpers as io_helpers
import distpy.io_help.h5_helpers as h5_helpers
import distpy.io_help.directory_services as ds
import distpy.controllers.parallel_strainrate_processing

def process_one_file(dataPack,ingest,configOuterFile,extended_list):
    distpy.controllers.parallel_strainrate_processing.main(configOuterFile, extended_list=extended_list,dataPack=dataPack)
    

def main(h5_files,configOuterFile, extended_list=[], ingester=h5_helpers, nthreads=1):

    if nthreads<2:
        for fname in h5_files:
            print(fname)
            dataPack = ingester.ingest(fname,"none",inMem=True)
            nx = dataPack['data'].shape[1]
            nt = dataPack['data'].shape[2]
            for a in range(dataPack['data'].shape[0]):
                # take one slice of data
                localDataPack = { "data" : numpy.reshape(dataPack['data'][a,:,:],(1,nx,nt)), "xaxis" : dataPack['xaxis'], "unixtime" : dataPack['unixtime']+a } 
                process_one_file(localDataPack,ingester.ingest,configOuterFile,extended_list)
    else:
        manager = multiprocessing.Manager()
        q = manager.Queue()
        print(multiprocessing.cpu_count())
        
        pool = multiprocessing.Pool(processes=nthreads, maxtasksperchild=1)

        jobs = []
        for fname in h5_files:
            print(fname)
            dataPack = ingester.ingest(fname,"none",inMem=True)
            nx = dataPack['data'].shape[1]
            nt = dataPack['data'].shape[2]
            for a in range(dataPack['data'].shape[0]):
                # take one slice of data
                localDataPack = { "data" : numpy.reshape(dataPack['data'][a,:,:],(1,nx,nt)), "xaxis" : dataPack['xaxis'], "unixtime" : dataPack['unixtime']+a } 
                process_one_file(localDataPack,ingester.ingest,configOuterFile,extended_list)
                job = pool.apply_async(process_one_file, [localDataPack,ingester.ingest,configOuterFile,extended_list])
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
