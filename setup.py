import os
from setuptools import setup, find_packages
from distpy import __version__, __license__
'''
 distpy : module for supporting signal processing workflows for DAS/DTS
          across multiple platforms.

VERSIONING
===========
Semantic versioning follows the pattern of ‹major›.‹minor›.‹patch›.

Rules:
Major : only for backwards incompatible changes
Minor : if not Major and not Patch
Patch : if and only if there is a fix that does not change functionality

The following story shows how the version name might evolve:
0.0.0 - first version
...Someone finds a bug, it gets fixed: 0.0.1
...We change the name of the controllers so that the old names won't work anymore 1.0.1
...We add some features to the command set so that you have more processing options 1.1.1


BUILD
=====
python -m setup sdist

INSTALL
=======
python -m setup install
      packages=['calc', 'controllers', 'ingesters', 'io_help','workers'],
      package_dir={                                                                                                                                                                                                
              'calc'       : 'distpy\calc',                                                                                                                                                                                       
              'controllers': 'distpy\controllers',                                                                                                                                                                              
              'ingesters'  : 'distpy\ingesters',                                                                                                                                                                            
              'io_help'    : 'distpy\io_help',
              'workers'    : 'distpy\workers'},             
      package_dir={'': 'distpy',
                   'distpy.io_help' : 'distpy\io_help'},
      packages=find_namespace_packages(where='distpy'),
          package_dir = {
              'distpy' : 'distpy',
              'distpy.calc' : 'distpy\calc',
              'dispty.controllers': 'distpy\controllers',
              'distpy.ingesters'  : 'distpy\ingesters',
              'distpy.io_help'    : 'distpy\io_help',
              'distpy.workers'    : 'distpy\workers'},

'''
import builtins

# Get a list of all files in the JS directory to include in our module
builtins.__DISTPY_SETUP__ = True
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths
py_files = package_files('distpy')

with open("README.md", "r") as fh:
    long_description = fh.read()

def setup_package():
    setup(name='distpy',
          version=__version__,
          description='distpy : Processing for distributed fibre-optic sensor data',
          long_description=long_description,
          long_description_content_type="text/markdown",
          author="Michael J. Williams",
          author_email="miw@slb.com",
          packages = find_packages(),
          package_data={"distpy":py_files},
          license=__license__,
          keywords = ["DAS", "DTS", "hDVS", "DVS"],
          url="https://github.com/Schlumberger/distpy",
          classifiers=[
              'Development Status :: 2 - Pre-Alpha',
              'Environment :: Other Environment',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: MIT License',
              'Natural Language :: English',
              'Programming Language :: Python',
              'Programming Language :: Python :: 3.7',
              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Physics',
              'Topic :: Software Development :: Libraries',
              'Topic :: Utilities'
          ],
          install_requires=[
              'numpy',
              'scipy',
              'numba',
              'matplotlib',
              'h5py',
              'pandas',
              'keras'
              ],
          zip_safe = False
           )

if __name__ == '__main__':
    setup_package()
    del builtins.__DISTPY_SETUP__
