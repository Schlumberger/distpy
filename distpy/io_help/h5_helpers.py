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
 ingest_h5 : recursively try to dump matrices and arrays as numpy *.npy files
'''
def ingest_h5(root, basepath):
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
        ingest_h5(readObj, basepath)
    readObj.close()


if __name__ == "__main__":
    main()


        
