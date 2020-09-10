# (C) 2020, Schlumberger. Refer to LICENSE

########################################
# USER CONFIGURABLE VALUES:
# A system configuration to optimize performance
jsonSystemConfig = "{PATH}\\config\\systemConfig.json"
# A signal processing workflow configuration
jsonConfig = "{PATH}\\config\\strainrate2summary.json"
########################################
import argparse

import copy
import os
import numpy
from distpy.workers.strainrate2summary import strainrate2summary
import multiprocessing
import distpy.io_help.io_helpers as io_helpers



def main(configOuterFile, extended_list=[], dataPack = {}):
    basedir,dirout,jsonConfig,PARALLEL,NCPU,BOX_SIZE, xaxisfile,taxisfile,prf = io_helpers.systemConfig(configOuterFile)

    # in-memory option
    data = dataPack.get('data',None)
    

    xaxis = None
    taxis = None
    if data is None:
        try:
            xaxis = numpy.load(xaxisfile)
        except FileNotFoundError:
            print('No x-axis specified')
        try:
            taxis = numpy.load(taxisfile)
        except FileNotFoundError:
            taxis = None
    else:
        # in-memory option
        xaxis = dataPack['xaxis']
        unixtime = dataPack['unixtime']
        prf = data.shape[2]
    

    configFile = io_helpers.json_io(jsonConfig,1)
    configData = configFile.read()
    # Note this copy - we don't (from a user point-of-view)
    # want to be doing system configuration in the strainrate2summary.json,
    # but we do from an internal point-of-view
    configData['BOX_SIZE']=BOX_SIZE
    configData['taxis'] = taxis
    verbose = configData.get('verbose',0)
    if verbose==1:
        print(configData)

    
    #scan for directories
    datafiles=[]
    if data is None:
        for root, dirs, files in os.walk(basedir):
            for datafile in files:
                if not datafile=='measured_depth.npy':
                    if not datafile=='time.npy':
                        datafiles.append(os.path.join(root,datafile))
            # break here because we don't want subdirectories (SLB case)
            break
    else:
        for a in range(data.shape[0]):
            # virtual filename
            datafiles.append(os.path.join(basedir,str(unixtime+a)+'.npy'))

    if not PARALLEL:
        # parallel does not work in Techlog...
        ii=0
        for datafile in datafiles:
            print(datafile)
            if data is None:
                strainrate2summary(datafile, xaxis, prf, dirout, configData, copy.deepcopy(extended_list),None)
            else:
                strainrate2summary(datafile, xaxis, prf, dirout, configData, copy.deepcopy(extended_list),numpy.squeeze(data[ii,:,:]))
            ii+=1
    else:
        manager = multiprocessing.Manager()
        q = manager.Queue()
        print(multiprocessing.cpu_count())
        
        pool = multiprocessing.Pool(processes=NCPU, maxtasksperchild=1)


        jobs = []
        ii=0
        for datafile in datafiles:
            if data is None:
                job = pool.apply_async(strainrate2summary, [datafile, xaxis, prf,dirout, configData,copy.deepcopy(extended_list),None])
            else:
                job = pool.apply_async(strainrate2summary, [datafile, xaxis, prf,dirout, configData,copy.deepcopy(extended_list),numpy.squeeze(data[ii,:,:])])
            ii+=1
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
