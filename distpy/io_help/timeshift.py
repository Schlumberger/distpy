# (c) 2020, Schlumberger. Refer to LICENSE

import os
import numpy

dirintest = "{PATH}\\results\\vlf"

'''
  timeshift : A utility for adding or subtracting a number of hours on
              unixtimestamp filenames.
              
'''
def timeshift(dirin, hours=0.0):
    MD_NAME = 'measured_depth.npy'
    TIME_NAME = 'time.npy'
    seconds = int(hours*3600)
    if seconds==0:
        return
    
    dirnames = [d for d in os.listdir(dirin) if os.path.isdir(os.path.join(dirin,d))]
    testdir = os.path.join(dirin,dirnames[0])
    fnames = [f for f in os.listdir(testdir) if os.path.isfile(os.path.join(testdir,f))]
    fnames.sort()
    if seconds>0:
        fnames.reverse()
    # make directories for the results
    xaxis=None
    for filename in fnames:
        if filename[-3:]=='npy':
            if filename==MD_NAME:
                xaxis=numpy.load(os.path.join(testdir,filename))
            dirname = filename[:-4]
            unixtime = int(dirname)+seconds
            
            fulldirin = os.path.join(dirin,filename)
            fulldirout = os.path.join(dirin,str(unixtime)+'.npy')
            os.rename(fulldirin,fulldirout)

# testing
if __name__ == "__main__":
    timeshift(dirintest)
