#...ingeseters
import distpy.io_help.sgy as sgy
import distpy.io_help.h5_helpers as h5

#..........process each data chunk and store results as WITSML
import distpy.controllers.edge_processing

import os

### Windows example
ARCHIVE_LOCATION = "D:/Archive"
PROJECT_DRIVE = "C:/NotBackedUp"
### Linux example
ARCHIVE_LOCATION = "/archive/projects/"
PROJECT_DRIVE = "/scratch/username/"
### Azure example
ARCHIVE_LOCATION = "/dbfs/mnt/segy/"
PROJECT_DRIVE = "/dbfs/tmp/segy/"


PROJECT = "2020prj0001"
CONFIG = "config"
BASE = os.path.join(ARCHIVE_LOCATION,PROJECT,CONFIG)

'''
 CASE01.py - in-memory usage example.
  Rather than ingesting as one step and processing as
  a separate step, this script will loop over the archive and create
  in-memory data chunks. This approach is more suited to edge-based real-time
  (due to lower file I/O) on systems where the memory is accessible across
  the CPUs.

  For cloud-based systems refer to CASE00.py
'''
def main():
    # In this configuration we specify the total number of processes
    # and set the PARALLEL configuration to 0
    ncpu = 32
    configStrainrate2Summary = os.path.join(BASE,"strainrate2fbeConfig.json")
    basedir = os.path.join(ARCHIVE_DRIVE,PROJECT,'sgy')

    #STEP 1: ingest SEG-Y
    ingester = sgy
    #ingester = h5

    print(basedir)
    # scan the directory and read all the filenames
    h5_files=[]
    for root, dirs, files in os.walk(basedir):
        for datafile in files:
            h5_files.append(os.path.join(root,datafile))

    extended_command_sets = []
    #--- add any additional command sets here
    #extended_command_sets.append('distpy_mine.calc.my_command_set')
    distpy.controllers.edge_processing.main(h5_files,configStrainrate2Summary, extended_list=extended_command_sets,ingester=ingester,nthreads=ncpu)

if __name__ == '__main__':
    main()
