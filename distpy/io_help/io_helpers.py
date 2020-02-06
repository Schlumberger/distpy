# (C) 2020, Schlumberger. Refer to LICENSE

import json
import numpy
import h5py

#...pandas has good support for CSV files
import pandas
from datetime import datetime as dtime
try:
    from . import wistmlfbe
except:
    import distpy.io_help.witsmlfbe as witsmlfbe
import os
import matplotlib.pyplot as plt
import matplotlib.colors

'''
 Automatic documentation support in latex and dot (graphviz)
 
'''
def substitute(lines, character, substitution):
    linesOut=[]
    for line in lines:
        linesOut.append(line.replace(character,substitution))
    return linesOut

def latexTop(lines):
    lines.append('\\documentclass{article}')
    # to allow us to incorporate figures...
    lines.append('\\usepackage{graphicx}')
    # to allow us to pretty-print JSON
    lines.append('\\usepackage{listings}')
    lines.append('\\usepackage{color}')
    lines.append('\\definecolor{dkgreen}{rgb}{0,0.6,0}')
    lines.append('\\definecolor{gray}{rgb}{0.5,0.5,0.5}')
    lines.append('\\definecolor{mauve}{rgb}{0.58,0,0.82}')

    lines.append('\\lstset{frame=tb,')
    lines.append('  language=Java,')
    lines.append('  aboveskip=3mm,')
    lines.append('  belowskip=3mm,')
    lines.append('  showstringspaces=false,')
    lines.append('  columns=flexible,')
    lines.append('  basicstyle={\\small\\ttfamily},')
    lines.append('  numbers=none,')
    lines.append('  numberstyle=\\tiny\\color{gray},')
    lines.append('  keywordstyle=\\color{blue},')
    lines.append('  commentstyle=\\color{mauve},')
    lines.append('  stringstyle=\\color{dkgreen},')
    lines.append('  breaklines=true,')
    lines.append('  breakatwhitespace=true,')
    lines.append('  tabsize=3')
    lines.append('}')
    lines.append('\\begin{document}')
    return lines
    
    
def latexTail(lines):
    lines.append('\\end{document}')
    return lines

def latexPng(lines):
    lines.append('\\subsection{Directed graph}')
    lines.append('\\begin{figure}')
    lines.append('\\includegraphics[width=\linewidth]{config.png}')
    lines.append('\\caption{The directed graph of the commands}')
    lines.append('\\label{fig:graph1}')
    lines.append('\\end{figure}')
    return lines


def latexJson(jsonIn,lines):
    lines.append('\\subsection{JSON  example}')
    lines.append('\\begin{lstlisting}')
    try:
        lines.append(json.dumps(jsonIn, indent=4, sort_keys=False))
    except TypeError:
        lines.append('Code not shown')
    lines.append('\\end{lstlisting}')
    return lines


def dot_graph(jsonArgs):
    lines = []
    fromList=[]
    fromList.append('load_data_0')
    lines.append('digraph G {')
    for command in jsonArgs:
        indx=command['uid']
        fromList.append(command['name']+'_'+str(indx))
        # from in_uid
        lines.append(fromList[command['in_uid']]+' -> '+fromList[indx] )
        prevlist = command.get('gather_uids',[-1])
        for prev in prevlist:
            if prev>=0:
                lines.append(fromList[prev]+' -> '+fromList[indx])
    lines.append('}')
    return lines
    
#
# io_helpers.py
#     write2witsml   : output results to the real-time WITSML 1.3.1 format
#     thumbnail_plot : outputa 2D array (e.g. 1 second from SEGY) to a thumbnail image
#     json_io        : handle reading and writing of JSON to different storage targets
#     command2md     : markdown representation of the signal processing workflow
def command2latex(command_list,linesOut,section='Command Set',topline='Commands used'):
    lines=[]
    lines.append('\\section{'+section+'}')
    lines.append(topline)
    for command in command_list:
        localDoc = command.docs()            
        lines.append('\\subsection{'+command.get_name()+'}')
        if 'one_liner' in localDoc:
            lines.append(localDoc['one_liner'])
        else:
            lines.append('Undocumented feature')
        if 'args' in localDoc:
            lines.append('\\begin{description}')
            for arg in localDoc['args']:
                lines.append('\\item['+arg+'] '+localDoc['args'][arg]['description'])
                if 'default' in localDoc['args'][arg]:
                    lines.append('\\\\default : '+str(localDoc['args'][arg]['default']))
            lines.append('\\end{description}')
                 
    #doc.generate_pdf(fileout,clean_tex=False, compiler='C:\\Users\\miw\\AppData\\Local\\Programs\\MiKTeX 2.9\\miktex\\bin\\x64\\pdflatex.exe')
    lines = substitute(lines,'_','\\textunderscore ')
    for line in lines:
        linesOut.append(line)
    return linesOut


    
def command2md(command_list):
    lines=[]
    lines.append("---------Workflow documentation---------")
    lines.append("# Command Set")
    for command in command_list:
        localDoc = command.docs()            
        lines.append('## '+command.get_name())
        if 'one_liner' in localDoc:
            lines.append(localDoc['one_liner'])
        if 'args' in localDoc:
            for arg in localDoc['args']:
                lines.append('### '+arg)
                lines.append(localDoc['args'][arg]['description'])
                if 'default' in localDoc['args'][arg]:
                    lines.append('default :'+str(localDoc['args'][arg]['default']))
    lines.append("---------Workflow documentation---------")
    return lines

'''
 write2witsml : Export the results of stage 1 (strainrate-to-summary) as
                WITSML FBE format. This is compatible with Techlog.
'''
# Custom writer for FBE...
def write2witsml(dirout,fileout,datestring,xaxis, band00, data, low_freq, high_freq, prf, data_style='UNKNOWN', label_list=[]):
    curves = witsmlfbe.generic_curves(low_freq,high_freq,prf)
    if (len(label_list)>0):
        icurve=0
        for label in label_list:
            curves[icurve]['mnemonic']=label
            curves[icurve]['description']=label
            icurve+=1

    dataOutsize = len(low_freq);
    useB0 = True
    if band00 is None:
        useB0 = False
    if useB0:
        dataOutsize=dataOutsize+1
    dataOut = numpy.zeros((data.shape[0],dataOutsize),dtype=numpy.double)
    if useB0:
        dataOut[:,0]=numpy.squeeze(band00)
        if (len(data.shape))<2:
            dataOut[:,1] = data
        else:
            dataOut[:,1:]=data
    else:
        dataOut = data
        curves = curves[1:]
    root = witsmlfbe.witsml_fbe(datestring,xaxis, curves, dataOut, data_style=data_style)
    # The datestring has characters that we do not want in a filename...
    # so we remo
    witsmlfbe.writeFBE(dirout,fileout,root)

'''
 Write a thumbnail plot using matplotlib
'''
def thumbnail_plot(stem_name, fname, data, xscale=[-1,-1],tscale=[-1,-1],plt_format='png', std_val_mult=1.0):
    plt.figure()
    nx=data.shape[0]
    nt=data.shape[1]
    if xscale[0]<0:
        xscale=[0,nx]
    if tscale[0]<0:
        tscale=[0,nt]
    # double ended fibre...special case
    if xscale[0]==xscale[1]:
        xscale=[0,nx]
        
    cscale='magma'
    #despiked = despike(data,3)
    meanval = numpy.mean(data)
    stdval = numpy.std(data)
    stdval = stdval*std_val_mult
    vminval = meanval - stdval
    vmaxval = meanval + stdval
    print(vminval,vmaxval)
    plt.imshow(data, cmap=cscale, aspect='auto', vmin=vminval, vmax=vmaxval, extent=[tscale[0],tscale[1],xscale[1],xscale[0]])
    plt.title(fname, fontsize=10)
    plt.xlabel('time (s)')
    plt.ylabel('measured depth (m)')
    fnameout = os.path.join(stem_name,fname+'.' + plt_format)
    print(fnameout)
    plt.savefig(os.path.join(stem_name,fname+'.'+plt_format),format=plt_format)
    plt.close()


'''
 write_plot : under development, this can replace plt.savefig() for compatibility with google cloud blob storage.
'''
def write_plot(stem_name, fname, plt):
    '''
    # For GOOGLE CLOUD FUNCTIONS USE...
    fig_to_upload = plt.gcf()
    # Save figure image to a bytes buffer
    buf = io.BytesIO()
    fig_to_upload.savefig(buf, format='png')
    buf.seek(0)
    # init GCS client and upload buffer contents
    blob = bucket.blob(OUT_PLOT_BLOB)  # This defines the path where the file will be stored in the bucket
    your_file_contents = blob.upload_from_string(buf.read(), content_type='image/png')
    '''
    # for everyone else...    
    fnameout = os.path.join(stem_name,fname+'.png')
    print(fnameout)
    plt.savefig(os.path.join(stem_name,fname+'.png'),format='png')
    plt.close()



'''
 json_io : all configuration files are JSON format. This class is already comfortable with both
           POSIX and blob-storage.
'''
# json_io allows the code to use local storage in provisioned mode...            
# NOTE: uses 'if' rather than Facade - currently there
#       are only minor differences between the codes
#       Later refactoring opportunity...
#       a facade would remove the 'if' options
class json_io(object):
    def __init__(self,filename, storage_flag):
        self.m_filename = filename
        self.m_provisioned = storage_flag
        #bucket = client.get_bucket(BACKSCATTER_BUCKET)
        self.m_json_blob = None
        self.m_json_file = None
        if self.m_provisioned==1:
            pass
        else:
            #self.m_json_blob = bucket.blob(filename)
            pass
    def write(self,string_file):
        if self.m_provisioned==1:
            with open(self.m_filename, 'w') as json_file:
                json_file.write(string_file)
        else:
            self.m_json_blob.upload_from_string(string_file)
    def read(self):
        if self.m_provisioned==1:
            with open(self.m_filename) as json_file:
                return json.load(json_file)
        else:
            configjson = self.m_json_blob.download_as_string()
            return json.loads(configjson.decode("utf-8"))

'''
 ingest_h5 : a first example of hdf5 import.
             Reads an h5 file. First version supports files that write 1-second
             chunks in a 'data','depth,'time' format.
'''
def ingest_h5(filename, basepath):
    # filename contains datestamp...
    # 161215_053839.h5
    # 1234567890123456
    endFname = filename[-16:]
    day = int(endFname[:2])
    month = int(endFname[2:4])
    year = int(endFname[4:6])
    hour = int(endFname[7:9])
    minute = int(endFname[9:11])
    second = int(endFname[11:13])
    timeObj = dtime(2000+year,month,day,hour,minute,second)
    unixtime = timeObj.timestamp()
    print(str(unixtime))
    readObj = h5py.File(filename,'r')
    numpy.save(os.path.join(basepath,'measured_depth.npy'),readObj['depth'][()])
    total_traces = readObj['data'].shape[0]
    chunksize = readObj['data'].chunks[0]
    increment=0
    for a in range(0,total_traces,chunksize):
        fname = str(int(unixtime+increment))+'.npy'
        fullPath = os.path.join(basepath,fname)
        if not os.path.exists(fullPath):
            numpy.save(fullPath,readObj['data'][a:a+chunksize,:].transpose())
        else:
            print(fullPath,' exists, assuming restart run and skipping.')
        increment+=1
    readObj.close()

'''
 ingest_csv :  Reads a chunk from a CSV file - note that we have DTS data which
               is often double-ended
'''
def ingest_csv(filename, jsonArgs):
    TIME_FORMAT='%Y/%m/%d%H:%M:%S%p'

    # xdir is 1 for increasing and -1 for decreasing measured-depth
    time_start = jsonArgs.get('tmin',None)
    time_end = jsonArgs.get('tmax',None)
    
    # currently assume the CSV format corresponds to a typical DTS dataset
    # This means that the column headings are dates...
    topline = pandas.read_csv(filename,nrows=1)
    nt = topline.shape[1]-1
    print('nt=',nt)
    time_scale = numpy.zeros((nt),dtype=numpy.double)
    icol1 = -1
    icol2 = -1
    t1=-1
    t2=-1
    if time_start is not None:
        t1 = dtime.strptime(time_start.replace(" ",""),TIME_FORMAT).timestamp()
    if time_start is not None:
        t2 = dtime.strptime(time_end.replace(" ",""),TIME_FORMAT).timestamp()
    for a in range(nt):
        if a==0:
            print(topline.keys()[a+1])
        if a==nt-1:
            print(topline.keys()[a+1])
        # special case - don't know if this is standard date format for DTS
        timestring=topline.keys()[a+1]
        dt = dtime.strptime(timestring.replace(" ",""),TIME_FORMAT)    
        time_scale[a] = dt.timestamp()
        if a==0:
            print(time_scale[a])
        if a==nt-1:
            print(time_scale[a])
    print(t1,'-',time_scale[0],'=',t1-time_scale[0])
    print(time_scale[-1],'-',t2,'=',time_scale[-1]-t2)
    if time_start is None:
        icol1=0
    else:                
        icol1 = numpy.argmin(numpy.abs(time_scale-t1))
        if icol1>nt-2:
            print('Time out of range')
            icol1=nt-2
    if time_end is None:
        icol2 = nt
    else:
        icol2 = numpy.argmin(numpy.abs(time_scale-t2))
        if icol2<=icol1:
            print('Single time value')
            icol2=icol1+1
    time_scale=time_scale[icol1:icol2]
    print(t1,t2,icol1,icol2)
    print(filename)
    print(topline.shape)
    topline=pandas.read_csv(filename,skiprows=10,usecols=[0])
    print('one column ',topline.shape)
    measured_depth=topline.astype(numpy.double)
    topline=pandas.read_csv(filename,skiprows=10,usecols=range(icol1+1,icol2+1))
    print('data ',topline.shape)
    data = topline.astype(numpy.double)
    
    jsonArgs['xaxis']=measured_depth
    jsonArgs['taxis']=time_scale
    return data, time_scale, jsonArgs

'''
 systemConfig : reads and interprets the JSON file for system configuration
'''
def systemConfig(configOuterFile):
    # an outer configuration file...
    configOuterJson = json_io(configOuterFile,1)
    csData = configOuterJson.read()
    inpath = csData.get('in_directory', None)
    outpath =csData.get('out_directory', None)
    #...better option
    indrive = csData.get('in_drive',None)
    if not (indrive is None):
        inpath = os.path.join(csData['in_drive'],csData['project'],csData['data'])
        outpath  = os.path.join(csData['out_drive'],csData['project'],csData['results'])
    xaxisfile = os.path.join(inpath,csData.get('measured_depth','none.npy'))
    taxisfile = os.path.join(inpath,csData.get('time_scale','none.npy'))
    prf = csData.get('prf',10000)

    BOX_SIZE = csData.get('BOX_SIZE',400)
    
    configFile = csData.get('config', None)
    parallel_flag = csData.get('parallel_flag',0)
    PARALLEL = False
    NCPU=1
    if parallel_flag>0:
        PARALLEL = True
        NCPU = csData.get('ncpu',1)
    return inpath, outpath, configFile, PARALLEL, NCPU, BOX_SIZE, xaxisfile, taxisfile, prf
