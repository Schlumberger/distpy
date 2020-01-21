# (c) 2020, Schlumberger. Refer to LICENSE

import os
import numpy

dirintest = "{PATH}\\results\\vlf"

'''
  dirswap : a utility for converting stage 1 (strainrate-to-summary)
            results (1 file per second) to
            inputs (all seconds in 1 file) for stage 2 (summary-to-plots)
            
  Swaps the structure so that the numpy files are named by their directory
  and so that the filenames become the directory names.


 e.g. the structure
 1535463913\band_0-nyquist.npy
 1535463913\band_01_0-1.npy
 1535463913\measured_depth.npy
 1535463913\time.npy
 1535463923\band_0-nyquist.npy
 1535463923\band_01_0-1.npy
 1535463923\measured_depth.npy
 1535463923\time.npy
 becomes...
 band_0-nyquist\1535463913.npy
 band_0-nyquist\1535463923.npy
 band_01_0-1\1535463913.npy
 band_01_0-1\1535463923.npy
 measured_depth\1535463913.npy
 measured_depth\1535463923.npy
 time\1535463913.npy
 time\1535463923.npy

 This then means that you can operate on the *.npy files, for example
 for generating plots or analyzing perceptual hashes.
'''
def dirswap(dirin):
    MD_NAME = 'measured_depth.npy'
    dirnames = [d for d in os.listdir(dirin) if os.path.isdir(os.path.join(dirin,d))]
    testdir = os.path.join(dirin,dirnames[0])
    fnames = [f for f in os.listdir(testdir) if os.path.isfile(os.path.join(testdir,f))]
    # make directories for the results
    xaxis=None
    for filename in fnames:
        if filename[-3:]=='npy':
            if filename==MD_NAME:
                xaxis=numpy.load(os.path.join(testdir,filename))
            dirname = filename[:-4]
            #fulldir = os.path.join(dirin,dirname)
            #if not os.path.exists(fulldir):
            #    os.makedirs(fulldir)
    for dirname in dirnames:
        testdir = os.path.join(dirin,dirname)
        fnames = [f for f in os.listdir(testdir) if os.path.isfile(os.path.join(testdir,f))]
        for filename in fnames:
            if filename[-3:]=='npy':
                dirout = os.path.join(dirin,filename[:-4])
                data = numpy.load(os.path.join(testdir,filename))
                #numpy.save(os.path.join(dirout,dirname+'.npy'), data)
                numpy.save(os.path.join(dirout+'.npy'), data)                
                #if not os.path.exists(os.path.join(dirout,MD_NAME)):
                #    numpy.save(os.path.join(dirout,MD_NAME),xaxis)
                # remove the old file...
                print('removing ',os.path.join(testdir,filename))
                os.remove(os.path.join(testdir,filename))

# testing
if __name__ == "__main__":
    dirswap(dirintest)
