# (C) 2020, Schlumberger. Refer to LICENSE

########################################
# USER CONFIGURABLE VALUES:
jsonSystemConfig = "{PATH}\\systemSGYConfig.json"
########################################
import argparse

import os
import distpy.io_help.sgy as sgy


def main(configOuterFile):
    # This module is superceded by the more generic ingester
    print("The use of parallel_ingest_sgy is now deprecated.")
    print("instead use parallel_ingest with ingester=distpy.io_help.sgy")
    parallel_ingest(configOuterFile,ingester=sgy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--filename", help='A configuration file')
    args = parser.parse_args()
    main(args.filename)
