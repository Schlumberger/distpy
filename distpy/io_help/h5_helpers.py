# (C) 2020, Schlumberger. Refer to LICENSE


########################################
# USER CONFIGURABLE VALUES:
# h5 formats - note that PRODML h5 is not universally adopted
#
# Note that these functions are based on reading an HDF5 file, and not defined from any standard
# It is fairly generic and can be adapted to the different HDF5 formats of other
# vendors.

filenameH5 = "{PATH}\\h5\\das_data.h5"
basepath = "{PATH}\\data"
WRITE_DATA = False
########################################

import h5py
import os
import numpy
import distpy.io_help.io_helpers
import distpy.io_help.directory_services as ds


'''
 list_keys : recursively list the keys from an h5 file
'''
def list_keys(root, basepath):
    for value in root.values():
        try:
            print(value.name,'   ',value.shape)
        except:
            print(value.name)
            list_keys(value, basepath)
        finally:
            pass

'''
 read_h5 : recursively try to dump matrices and arrays as numpy *.npy files - blindly...
'''
def read_h5(root, basepath):
    for value in root.values():
        try:
            # failure is based on value.shape throwing an exception in the print statement
            # ...so do not remove this print statement
            print(value.name,'   ',value.shape)
            numpy.save(os.path.join(basepath,value.name),value)
        except:
            try:
                dirname = basepath+value.name
                print(value.name)
                ds.makedir(dirname)
                ingest_h5(value, basepath)
            except:
                print('FAILED at ',value.name)
                pass
            finally:
                pass
        finally:
            pass

'''
 type0_h5  : a first example of hdf5 import.
             Reads an h5 file. First version supports files that write 1-second
             chunks in a 'data','depth,'time' format.
'''
def type0_h5(readObj, basepath, filename, inMem=False):
    # filename contains datestamp...
    # 161215_053839.h5
    # 1234567890123456
    endFname = filename[-16:]
    day = int(endFname[:2])
    month = int(endFname[2:4])
    year = int(endFname[4:6])
    hour = int(endFname[7:9])
    minute = int(endFname[9:11])
    second = int(endFname[11:13])
    timeObj = dtime(2000+year,month,day,hour,minute,second)
    unixtime = int(timeObj.timestamp())
    print(str(unixtime))

    xaxis = readObj['depth'][()]
    if inMem==False:
        numpy.save(os.path.join(basepath,'measured_depth.npy'),xaxis)
    total_traces = readObj['data'].shape[0]
    chunksize = readObj['data'].chunks[0]

    prf=chunksize
    nt=total_traces
    nx = xaxis.shape[0]

    data = numpy.zeros((1,1,1),dtype=numpy.double)
    if inMem==True:
        data = numpy.zeros((int(nt/prf),nx,prf),dtype=numpy.double)

 
    increment=0
    for a in range(0,total_traces,chunksize):
        if inMem==False:
            fname = str(int(unixtime+increment))+'.npy'
            fullPath = os.path.join(basepath,fname)
            if not os.path.exists(fullPath):
                numpy.save(fullPath,readObj['data'][a:a+chunksize,:].transpose())
            else:
                print(fullPath,' exists, assuming restart run and skipping.')
        else:
            data[increment,:,:]=fullPath,readObj['data'][a:a+chunksize,:].transpose()
        increment+=1
    return { "data" : data, "xaxis" : xaxis, "unixtime" : unixtime }

'''
 type1_h5  : A common HDF5 format used in DAS
     depth = '/Acquisition/FacilityCalibration[0]/Calibration[0]/LocusDepthPoint' in readObj
     time  = '/Acquisition/Raw[0]/RawDataTime' in readObj
     data  = '/Acquisition/Raw[0]/RawData' in readObj

     This example is given in https://github.com/Schlumberger/distpy/wiki/Dev-Tutorial-:-Extending-distpy-with-new-ingesters
'''
def type1_h5(readObj, basepath, inMem=False):
    depths = readObj['/Acquisition/FacilityCalibration[0]/Calibration[0]/LocusDepthPoint']
    times  = readObj['/Acquisition/Raw[0]/RawDataTime']
    
    # The data can be large - so don't read it all into memory at once.
    data_name = '/Acquisition/Raw[0]/RawData'

    time_vals = numpy.zeros((times.shape[0]))
    for a in range(times.shape[0]):
        time_vals[a] = times[a]*1e-6
    prf = int(numpy.round(1.0/numpy.mean(time_vals[1:]-time_vals[:-1])))


    xaxis = numpy.zeros((depths.shape[0]))
    for a in range(depths.shape[0]):
        tuple_val = depths[a]
        xaxis[a] = tuple_val[1]
    nx = depths.shape[0]

    if inMem==False:
        numpy.save(os.path.join(basepath,'measured_depth.npy'),depth_vals)

    unixtime = int(time_vals[0])
    print(str(unixtime))

    # How many 1 second files to write from this HDF5
    nsecs = int(times.shape[0]/prf)

    data = numpy.zeros((1,1,1),dtype=numpy.double)
    if inMem==True:
        data = numpy.zeros((nsecs,nx,prf),dtype=numpy.double)    
    ii=0
    for a in range(nsecs):
        if inMem==False:
            fname = str(int(unixtime+a))+'.npy'
            fullPath = os.path.join(basepath,fname)
            if not os.path.exists(fullPath):
                numpy.save(fullPath,readObj[data_name][ii:ii+prf,:].transpose())
            else:
                print(fullPath,' exists, assuming restart run and skipping.')
        else:
            data[a,:,:] = readObj[data_name][ii:ii+prf,:].transpose()
        ii+=prf
    return { "data" : data, "xaxis" : xaxis, "unixtime" : unixtime }

'''
 ingest_h5 : select the reader that recognizes the file type, and read the file
'''
def ingest(filename,basepath, inMem=False):
    readObj = h5py.File(filename,'r')
    retObj={}
    
    # TYPE 0:
    depth = 'depth' in readObj
    #time  = '/Acquisition/Raw[0]/RawDataTime' in readObj
    data  = 'data' in readObj
    if depth and data:
        retObj = type0_h5(readObj,basepath,filename, inMem)
    
    # TYPE 1:
    depth = '/Acquisition/FacilityCalibration[0]/Calibration[0]/LocusDepthPoint' in readObj
    time  = '/Acquisition/Raw[0]/RawDataTime' in readObj
    data  = '/Acquisition/Raw[0]/RawData' in readObj
    if depth and time and data:
        retObj = type1_h5(readObj,basepath, inMem)
    readObj.close()
    return retObj


'''
 main : two modes, WRITE_DATA==True - recursively try to dump data
                   WRITE_DATA==False - recursively list the key-value contents of the file

        In general we use WRITE_DATA==False to determine format information, and then construct
        a custom reader. In some cases WRITE_DATA==True is sufficiently close to what we need that
        the custom reader just performs directory renaming. However, note that in blob storage the
        directory rename is a deep copy and usually very slow - so on cloud custom h5 readers that
        set up the output name at the start are really important from a cost perspective.
'''
def main():
    filename = filenameH5
    readObj = h5py.File(filename,'r')
    if WRITE_DATA==False:
        list_keys(readObj,basepath)
    else:
        read_h5(readObj, basepath)
    readObj.close()


if __name__ == "__main__":
    main()


        
