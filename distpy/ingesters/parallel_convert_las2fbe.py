# (C) 2020, Schlumberger. Refer to LICENSE

########################################
# USER CONFIGURABLE VALUES:
inpath = "{PATH}\\las2"
outpath = "{PATH}\\witsml"
PARALLEL=True
NCPU=1
########################################
import argparse

import os
import distpy.io_help.witsmlfbe as witsmlfbe
import multiprocessing

'''
convert : separates the call to the conversion from the main() function
          This allows the main function itself to not depend on distpy.io_help
          the dependency is fielded here
'''
def convert(infile,outfile):
    root=witsmlfbe.las2fbe(infile)
    witsmlfbe.writeFBE(outpath,outfile,root)


####################################################################################################
# Executable code...

def main(configOuterFile):
    # standard configuration information
    inpath,outpath,configFile,PARALLEL,NCPU,BOX_SIZE, xaxisfile,taxisfile,prf = io_helpers.systemConfig(configOuterFile)

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    #################################
    # Process all the files with the
    # .las extension
    #################################
    inlist = []
    outlist = []
    for filename in os.listdir(inpath):
        if os.path.isfile(os.path.join(inpath,filename)):
            if filename[-4:]=='.las':
                inlist.append(os.path.join(inpath,filename))
                outlist.append(filename)

    if not PARALLEL:
        # parallel does not work in Techlog...
        for (infile,outfile) in zip(inlist,outlist):
            convert(infile,outfile)
    else:
        manager = multiprocessing.Manager()
        q = manager.Queue()
        print(multiprocessing.cpu_count())
        
        pool = multiprocessing.Pool(processes=NCPU, maxtasksperchild=1)
        
        jobs = []
        for (infile,outfile) in zip(inlist,outlist):
            job = pool.apply_async(convert, [infile,outfile])
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
