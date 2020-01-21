# (c) 2020 Schlumberger. Refer to LICENSE

# directory services - in the short term this is just a wrapper on
#   some standard os functions.
#   It supports a storage_type identifier, as some cloud storage types
#   do not support python's os (POSIX) naming. Some examples for Google
#   Cloud Buckets using blobs rather than *.npy files are coded for reference.
#
import os
import numpy
import string
import csv

BACKSCATTER_BUCKET = "google-storage-cloud-address"

'''
 path_join : a wrapper for os.path.join()
             The storage_type default uses os.path.join
'''
def path_join(dirname1, dirname2, storage_type=1):
    if storage_type==1:
        return os.path.join(dirname1,dirname2)
    else:
        return os.path.join(dirname1,dirname2)
'''
 path_join : a wrapper for os.path.join()
             The storage_type default uses os.path.join
'''
def exists(pathname):
    if not os.path.exists(pathname):
        return False
    return True

'''
 makedir : a wrapper for os.path.makedirs generating the path if it does not
           exist.

           Current question: For Google Blob storage the naming convention can
           look likea a directory\file so does this just pass?
'''
def makedir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

'''
 load : wrappers numpy.load() for filestores, otherwise interprets a byte-based
        blob from a google cloud bucket.
'''
def load(filename, storage_type=1):
    if storage_type == 1:
        return numpy.load(filename)
    else:
        #rawObj = blob_server(filename,PROVISIONED)
        bucket = client.get_bucket(BACKSCATTER_BUCKET)
        blob = bucket.get_blob(filename)
        byteval = blob.download_as_string()
        time_ref = numpy.frombuffer(byteval, dtype=numpy.double)
        return time_ref

# read zones - used to read cluster and perforation locations
# from csv files. Notethis is NOT used to read DTS data from CSV format files!
def csv_reader(filename, storage_type=1):
    lines = []
    reader = None
    if storage_type==1:
        with open(filename) as csvFile:
            reader = csv.reader(csvFile, delimiter=' ')
            for row in reader:
                lines.append(row)
    else:
        bucket = client.get_bucket(BACKSCATTER_BUCKET)
        blob = bucket.get_blob(filename)
        reader = csv.reader(io.StringIO((blob.download_as_string()).decode("utf-8")), delimiter=' ')
        for row in reader:
            lines.append(row)
    return lines



