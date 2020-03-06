# (c) Schlumberger 2020. See LICENSE
"""
distpy-docker : A general docker container for building orchestrated
                 execution chains with Kubernetes

                 1. Copy this example and the Dockerfile to a clean
                 location

                 2. Create a Docker container using
                   docker build --no-cache -t distpy-python-app .

                 3. Test your container using
                   docker run distpy-python-app

                   The output should look like (additional missing-file messages will appear on non-gpu systems):
                   Using TensorFlow backend.
                   usage: distpy-docker.py [-h] [-f FILE] [-c JSON] [-m MODULE]

                   optional arguments:
                     -h, --help            show this help message and exit
                     -f FILE, --file FILE  write report to FILE
                     -c JSON, --config JSON
                                           read config from JSON
                     -m MODULE, --module MODULE
                                           select a processing module

                4. Advanced test, ingest a SEGY file
                docker run -v C:\\NotBackedUp:/scratch distpy-python-app -f /scratch/myproject/sgy/test.sgy -c /scratch/myproject/config/docker_sgyConfig.json

                5. Kubernetes - see
                https://docs.docker.com/docker-for-windows/kubernetes/
"""
from argparse import ArgumentParser
import os
import numpy
import distpy.io_help.sgy as sgy
import distpy.io_help.io_helpers as io_helpers
import distpy.io_help.h5_helpers as h5_helpers
from distpy.workers.strainrate2summary import strainrate2summary
from distpy.workers.plotgenerator import plotgenerator
import distpy.io_help.directory_services as ds
import distpy


def distpy_run(fname,configOuterFile,keyword):
    basedir,dirout,jsonConfig,PARALLEL,NCPU,BOX_SIZE, xaxisfile,taxisfile,prf = io_helpers.systemConfig(configOuterFile)
    ds.makedir(dirout)
    extended_list=[]
    xaxis = None
    try:
        xaxis = numpy.load(xaxisfile)
    except FileNotFoundError:
        print('No x-axis specified')

    configData={}
    configData['BOX_SIZE']=500

    if jsonConfig is not None:
        configFile = io_helpers.json_io(jsonConfig,1)
        configData = configFile.read()
        # Note this copy - we don't (from a user point-of-view)
        # want to be doing system configuration in the strainrate2summary.json,
        # but we do from an internal point-of-view
        configData['BOX_SIZE']=BOX_SIZE

    # Currently support ingestion of SEGY & HDF5 plus processing from strainrate2summary
    if keyword=='strainrate2summary':
        strainrate2summary(fname, xaxis, prf, dirout, configData, copy.deepcopy(extended_list))
    if keyword=='plotgenerator':
        plotgenerator(basedir, dirout, configData)
    if keyword=='segy_ingest':
        sgy.SEGYingest(fname, dirout)
    if keyword=='ingest_h5':
        h5_helpers.ingest_h5(fname, dirout)




if __name__ == "__main__":
    # based on https://stackoverflow.com/questions/1009860/how-to-read-process-command-line-arguments
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename",
                        help="write report to FILE", metavar="FILE")
    parser.add_argument("-c", "--config", dest="config",
                        help="read config from JSON", metavar="JSON")
    parser.add_argument("-m", "--module", dest="module",
                        help="select a processing module", metavar="MODULE")

    args = parser.parse_args()
    print(distpy.__version__)
    print(os.listdir('/scratch'))

    distpy_run(vars(args)['filename'],
               vars(args)['config'],
               vars(args)['module'])
