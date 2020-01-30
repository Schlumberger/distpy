#..........ingest SEG-Y from an archived data store
import distpy.ingesters.parallel_ingest_sgy
#..........process each data chunk and store results as WITSML
import distpy.controllers.parallel_strainrate_processing
#..........ingest WITSML from an archived data store
import distpy.ingesters.parallel_ingest_witsml
#..........summarize results in plot views
import distpy.controllers.parallel_plots
import os

### Windows example
ARCHIVE_DRIVE = "D:\\Archive"
PROJECT_DRIVE = "C:\\NotBackedUp"
### Linux example
ARCHIVE_DRIVE = "/archive/projects/"
PROJECT_DRIVE = "/scratch/username/"
### Azure example
ARCHIVE_DRIVE = "/dbfs/mnt/segy/"
PROJECT_DRIVE = "/dbfs/tmp/segy/"


PROJECT = "2020prj0001"
CONFIG = "config"
BASE = os.path.join(ARCHIVE_LOCATION,PROJECT,CONFIG)

def main():
    configSGYIngest = os.path.join(BASE,"sgyConfig.json")
    configStrainrate2Summary = os.path.join(BASE,"strainrate2fbeConfig.json")
    configWITSMLIngest = os.path.join(BASE,"ingestWITSMLConfig.json")
    configPlots = os.path.join(BASE,"plotsConfig.json")
    

    #STEP 1: ingest SEG-Y
    distpy.ingesters.parallel_ingest_sgy.main(configSGYIngest)
    

    #STEP 2: process strain-rate to summary
    extended_command_sets = []
    #--- add any additional command sets here
    #extended_command_sets.append('distpy_mine.calc.my_command_set')
    distpy.controllers.parallel_strainrate_processing.main(configStrainrate2Summary, extended_list=extended_command_sets)

    #STEP 3: ingest WITSML FBE
    distpy.ingesters.parallel_ingest_witsml.main(configWITSMLIngest)

    #STEP 4: Generate default plots - available from version 1.1.0
    distpy.controllers.parallel_plots.main(configPlots)

if __name__ == '__main__':
    main()
