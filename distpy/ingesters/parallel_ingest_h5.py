# (C) 2020, Schlumberger. Refer to LICENSE

########################################
# USER CONFIGURABLE VALUES:
jsonSystemConfig = "{PATH}\\systemh5Config.json"
########################################
import argparse

import os
import distpy.io_help.h5_helpers as h5_helpers
import distpy.ingesters.parallel_ingest as parallel_ingest

def main(configOuterFile):
    # This module is superceded by the more generic ingester
    print("The use of parallel_ingest_h5 is now deprecated.")
    print("instead use parallel_ingest with ingester=distpy.io_help.h5_helpers")
    parallel_ingest(configOuterFile,ingester=h5_helpers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--filename", help='A configuration file')
    args = parser.parse_args()
    main(args.filename)
