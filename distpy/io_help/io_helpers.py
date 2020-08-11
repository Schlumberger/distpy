# (C) 2020, Schlumberger. Refer to LICENSE

import json
import numpy
import h5py
import copy
import re

#...pandas has good support for CSV files
import pandas
from datetime import datetime as dtime
try:
    from . import wistmlfbe
except:
    import distpy.io_help.witsmlfbe as witsmlfbe
import os
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

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


def dot_graph(jsonArgs, command_list=None):
    lines = []
    fromList=[]
    nodeList=[]
    lines.append('digraph G {')
    if command_list is None:
        nodeList.append('load_data_0')
        # Basic graph
        for command in jsonArgs:
            comment = ''
            if 'comment' in command.keys():
                comment = command['comment']
            else:
                comment = command['name']
            formatting = command.get('formatting','')
            indx=command['uid']
            nodeList.append(command['name']+'_'+str(indx))
            lines.append(nodeList[indx] + '[label=\"'+comment+'\",'+formatting+'];')
            # from in_uid
            lines.append(nodeList[command['in_uid']]+' -> '+nodeList[indx] )
            prevlist = command.get('gather_uids',[-1])
            for prev in prevlist:
                if prev>=0:
                    lines.append(nodeList[prev]+' -> '+nodeList[indx])
    else:
        # We will pre-pend the list, so make sure we are working on a copy
        jsonArgs = copy.deepcopy(jsonArgs)
        jsonArgs.insert(0,{ "name" : "load_data", "uid" : 0, "in_uid" : -1 })
        # Advanced graphs
        for command, concreteCommand in zip(jsonArgs,command_list):
            indx=command['uid']
            localDoc = concreteCommand.docs()
            description = 'Undocumented feature'
            if 'one_liner' in localDoc:
                description = (localDoc['one_liner'])
            color = 'color=red,style=filled,fontcolor=white'
            if concreteCommand.isGPU()==True:
                color = 'color=green,style=filled,fontcolor=black'
            #label = '[label=\"'+command['name']+ ' '+description+'\",'+color+']'
            label = '[label=\"'+command['name']+ '\",'+color+']'
            nodeList.append(command['name']+'_'+str(indx))
            lines.append(nodeList[indx] + ' ' + label + ';')
            # from in_uid
            if command['in_uid']>=0:
                lines.append(nodeList[command['in_uid']]+' -> '+nodeList[indx] )
            prevlist = command.get('gather_uids',[-1])
            for prev in prevlist:
                if prev>=0:
                    lines.append(nodeList[prev]+' -> '+nodeList[indx])
    lines.append('}')
    return lines
    
#
# io_helpers.py
#     write2witsml   : output results to the real-time WITSML 1.3.1 format
#     thumbnail_plot : outputa 2D array (e.g. 1 second from SEGY) to a thumbnail image
#     json_io        : handle reading and writing of JSON to different storage targets
#     command2md     : markdown representation of a single command
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
        if command.isGPU()==True:
            lines.append('This command can be used with GPU.')
        else:
            lines.append('This command cannot be used with GPU.')
                 
    #doc.generate_pdf(fileout,clean_tex=False, compiler='C:\\Users\\miw\\AppData\\Local\\Programs\\MiKTeX 2.9\\miktex\\bin\\x64\\pdflatex.exe')
    lines = substitute(lines,'_','\\textunderscore ')
    for line in lines:
        linesOut.append(line)
    return linesOut


    
def command2md(command, lines):
    localDoc = command.docs()            
    lines.append('## '+command.get_name())
    if 'one_liner' in localDoc:
        lines.append(localDoc['one_liner'])
    if 'args' in localDoc:
        for arg in localDoc['args']:
            lines.append('`'+arg+'` : '+localDoc['args'][arg]['description'])
            if 'default' in localDoc['args'][arg]:
                lines.append('default : '+str(localDoc['args'][arg]['default']))
    if command.isGPU()==True:
        lines.append('This command can be used with GPU.')
    else:
        lines.append('This command cannot be used with GPU.')
    return lines

'''
 write2witsml : Export the results of stage 1 (strainrate-to-summary) as
                WITSML FBE format. This is compatible with Techlog.
'''
# Custom writer for FBE...
def write2witsml(dirout,fileout,datestring,xaxis, band00, data, low_freq, high_freq, prf, data_style='UNKNOWN', label_list=[], unit_list=[]):
    # handle the case of a transposed trace....
    if (data.shape[0]==1):
        data = numpy.transpose(data)
        band00 = numpy.transpose(band00)
        # simple axis replaces the actual axis - deferred better axis handling to a future version
        xaxis = numpy.linspace(0, data.shape[0], data.shape[0],endpoint=False,dtype=numpy.double)
    curves = witsmlfbe.generic_curves(low_freq,high_freq,prf)
    if (len(label_list)>0):
        icurve=0
        for label in label_list:
            curves[icurve]['mnemonic']=label
            curves[icurve]['description']=label
            icurve+=1
    if (len(unit_list)>0):
        icurve=0
        for unit in unit_list:
            curves[icurve]['unit']=unit
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
 csv2fbe - handles a basic CSV input, converting to WITSML FBE

 The CSV must be interpretable by pandas into a dataframe
'''
def csv2fbe(filein, stem, configJson):
    # Sometimes data comes in the DTS-style (depth as rows, time as columns), so we catch that and pass it on
    time_row = configJson.get('time_row',-1)
    if time_row>-1:
        return csv2dts(filein, stem, configJson)
    
    # output directory name
    head_tail = os.path.split(filein)
    # Assumption: original suffix is .csv or other 3 letter suffix.
    stage = (head_tail[1])[:-4]
    dirout = os.path.join(stem,stage)
    if not os.path.exists(dirout):
        os.makedirs(dirout)
    print('writing to '+dirout)

    # recover local variables from the config
    TIME_FORMAT = configJson.get('time_format','%Y-%m-%d %H:%M:%S')
    csv_labels = configJson.get('csv_labels',[])
    curves = configJson.get('curves',[])
    if not (len(csv_labels)==len(curves)):
        print('Different number of csv_labels to curves in the output file, please check the config.')
        return
    if (len(csv_labels)==0) or (len(curves)==0):
        print('Either no csv_labels or curves found, please check the config.')
        return
    depth_unit = configJson.get('depth_unit','m')
    data_style = configJson.get('data_style','UNKNOWN')

    # Read the CSV and sort by time-stamps
    df = pandas.read_csv(filein)
    uniquevals = numpy.unique(df[configJson['time_label']].values)
    
    
    for id in uniquevals:
        # extract all the data at the timestamp and sort by depth ready for output
        newdf = df[df[configJson['time_label']] == id]
        newdf = newdf.sort_values(by=[configJson['depth_label']])
        
        xaxis = numpy.array(newdf[configJson['depth_label']].tolist())
        
        timeval_curr = dtime.strptime(id,TIME_FORMAT)
        fileout = str(int(timeval_curr.timestamp()))
        datestring = dtime.fromtimestamp(int(fileout)).strftime("%Y-%m-%dT%H:%M:%S+00:00")

        # Convert the data to numpy for writing
        data = numpy.zeros((xaxis.shape[0],len(csv_labels)))
        for a in range(len(csv_labels)):
            data[:,a]=numpy.array(newdf[csv_labels[a]].tolist())
            
        root = witsmlfbe.witsml_fbe(datestring,xaxis, curves, data, data_style=data_style, depth_unit=depth_unit)
        # The datestring has characters that we do not want in a filename...
        # so we remo
        witsmlfbe.writeFBE(dirout,fileout,root)
'''
  remove_trailing_number() : Issue if a supplied CSV has duplicate timestamps in the columns
                            the pandas.read_csv() does not allow duplicated columns and so introduces appending .1, .2 etc.
                            to the column names.
                            However, datatime.strptime() demands a fixed format - which these columns no longer have...
                            so this function parses a list of strings and removes a trailing .* string, if one exists
'''
def remove_trailing_number(list_of_strings):
    date_re = re.compile(r'[.].*$')
    newlist = []
    for one_string in list_of_strings:
        endpart = date_re.search(one_string)
        if endpart is None:
            newlist.append(one_string)
        else:
            newlist.append(one_string[:endpart.span()[0]])
    return newlist

'''
 csv2dts - handles gridded DTS data with rows of depth and
           columns that are timesteamps

 The CSV must be interpretable by pandas into a dataframe
'''
def csv2dts(filein, stem, configJson):
    # output directory name
    output_format = configJson.get('output_format','witsml')
    head_tail = os.path.split(filein)
    # Assumption: original suffix is .csv or other 3 letter suffix.
    stage = (head_tail[1])[:-4]
    dirout = os.path.join(stem,stage)
    if not os.path.exists(dirout):
        os.makedirs(dirout)
    print('writing to '+dirout)

    # recover local variables from the config
    TIME_FORMAT = configJson.get('time_format','%Y-%m-%d %H:%M:%S')
    # time_row must exist or we wouldn't assume this data format
    time_row = configJson['time_row']
    first_data = configJson.get('first_data',1)
    
    
    curves = configJson.get('curves',[ { "mnemonic" : "T", "unit" : "F", "description" : "DTS temperature in Fahrenheit" } ])
    depth_unit = configJson.get('depth_unit','m')
    data_style = configJson.get('data_style','UNKNOWN')

    skiprows = []
    for a in range(first_data):
        if not a==time_row:
            skiprows.append(a)

    # Read the CSV
    df = pandas.read_csv(filein,header=time_row,skiprows=skiprows)

    if output_format == 'witsml':
        firstTime=True
        for column in df:
            if firstTime == True:
                xaxis = numpy.array(df[column].tolist())
            else:
                # Fix possible problem caused by pandas reader's "mangle" system
                column = str.strip(column)
                date_re = re.compile(r'[.].*$')
                endpart = date_re.search(column)
                if endpart is not None:
                    column = column[:endpart.span()[0]]
                timeval = dtime.strptime(column,TIME_FORMAT)
                fileout = str(int(timeval.timestamp()))
                datestring = dtime.fromtimestamp(int(fileout)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
                data = numpy.array(df[column].tolist())
                root = witsmlfbe.witsml_fbe(datestring,xaxis, curves, data, data_style=data_style, depth_unit=depth_unit)
                # The datestring has characters that we do not want in a filename...
                # so we remo
                witsmlfbe.writeFBE(dirout,fileout,root)
            firstTime=False
    else:
        xaxis = numpy.squeeze(numpy.array(df.values[:,0]))
        # PROBLEM - duplicate times are appended by .1, .2, .3 etc.
        cols = list(map(str.strip, list(df.columns.values)))
        # regex to find a trailing .* in the string...
        cols = remove_trailing_number(cols)
        taxis_dates = numpy.array(list(map(dtime.strptime, cols[1:], itertools.repeat(TIME_FORMAT,len(cols[1:])))))
        taxis = numpy.array([ts.timestamp() for ts in taxis_dates])
        data = df.values[:,1:]
        xaxisfilename = os.path.join(dirout,'measured_depth.npy')
        numpy.save(xaxisfilename, xaxis)
        taxisfilename = os.path.join(dirout,'time.npy')
        numpy.save(taxisfilename, taxis)
        datafilename = os.path.join(dirout,str(int(taxis[0]))+'.npy')
        numpy.save(datafilename, data)

'''
 Write a thumbnail plot using matplotlib
'''
def thumbnail_plot(stem_name, fname, data, xscale=[-1,-1],tscale=[-1,-1],plt_format='png',
                   std_val_mult=1.0, data2=None, jsonArgs={}):
    cmap=jsonArgs.get('cmap','viridis')
    axes=jsonArgs.get('axes',['depth (m)','time (s)'])
    xaxis=jsonArgs.get('xaxis',None)
    taxis=jsonArgs.get('taxis',None)
    invert_depth=jsonArgs.get('invert_depth',0)
    figure_size = jsonArgs.get('figure_size',(8,6))
    dots_per_inch = jsonArgs.get('dpi',300)
    plot_style = jsonArgs.get('plot_style','imshow')
    NLEVELS=30
    title_string = dtime.fromtimestamp(int(fname)).strftime('%Y-%m-%d %H:%M:%S')
    fig, ax = plt.subplots(figsize=figure_size, dpi=dots_per_inch)
    print('xscale',xscale)
    nx=data.shape[0]
    if xscale[0]==xscale[-1]:
        xscale=[0,nx]
    if data.ndim<2:
        plt.plot(xaxis,data)
        ax.set_xlim(xscale[0],xscale[-1])
        plt.title(title_string + ' (' + fname + ')', fontsize=10)
        return

    nt=data.shape[1]
    if tscale[0]==tscale[-1]:
        tscale=[0,nt]
        
    cscale=cmap
    #despiked = despike(data,3)
    meanval = numpy.mean(data)
    stdval = numpy.std(data)
    stdval = stdval*std_val_mult
    vminval = meanval - stdval
    vmaxval = meanval + stdval
    print(vminval,vmaxval)
    t_lims = tscale
    if tscale[0]>1e+7:
        # https://stackoverflow.com/questions/23139595/dates-in-the-xaxis-for-a-matplotlib-plot-with-imshow
        t_lims = list(map(dtime.fromtimestamp, [int(tscale[0]), int(tscale[1])]))
        t_lims = mdates.date2num(t_lims)

    taxis2= list(map(dtime.fromtimestamp, taxis.tolist()))
    taxis2= mdates.date2num(taxis2)

    if plot_style=='scatter':
        if data2 is None:
            plot_style='imshow'
        else:
            data_2 = data2[0]
            s=None
            c=None
            if len(data2)>1:
                c=data2[1].flatten()
            if len(data2)>2:
                s=data2[2].flatten()
            img = plt.scatter(data.flatten(),data_2.flatten(),c=c, s=s, cmap=cscale)
    if plot_style=='imshow':
        img = plt.imshow(data, cmap=cscale, aspect='auto', vmin=vminval, vmax=vmaxval, extent=[t_lims[0],t_lims[1],xscale[1],xscale[0]])
        cbar = fig.colorbar(img)
    if plot_style=='contourf':
        # create mesh to plot
        xv, yv = numpy.meshgrid(taxis2[:], xaxis[:])
        img = plt.contourf(xv,yv,data, NLEVELS,cmap=cscale)  # colormap,
        cbar = fig.colorbar(img)
    if tscale[0]>1e+7:
        # we have dates... so use dates
        # https://stackoverflow.com/questions/23139595/dates-in-the-xaxis-for-a-matplotlib-plot-with-imshow
        ax.set_xlim(t_lims[0],t_lims[1])
        ax.xaxis_date()
        date_format = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
        ax.xaxis.set_major_formatter(date_format)
        # set the dates to diagonal so they fit better
        fig.autofmt_xdate()
    plt.title(title_string + ' (' + fname + ')', fontsize=10)
    plt.xlabel(axes[1])
    plt.ylabel(axes[0])
    if invert_depth==1:
        plt.gca().invert_yaxis() # invert depth axis
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    plt.tight_layout()

    
    fnameout = os.path.join(stem_name,fname+'.' + plt_format)
    print(fnameout)
    plt.savefig(os.path.join(stem_name,fname+'.'+plt_format),format=plt_format)
    inline_plots = jsonArgs.get('inline_plots',0)
    if inline_plots==1:
        plt.show()
    else:
        plt.close()


'''
 numpy_out - write the current data block to file.
'''
def numpy_out(stem_name,fname, data):
    numpy.save(os.path.join(stem_name,fname+'.npy'),data)
    
'''
 concatente_commands - take a command set and concatenate another command set.
                       No efficiency gains here...
'''
def concatenate_commands(orig_commands, new_commands):
    id_offset = orig_commands[-1]['uid']
    for command in new_commands:
        command['uid'] = command['uid']+id_offset
        command['in_uid'] = command['in_uid']+id_offset
        gather_uids = command.get('gather_uids',[])
        gather_uids_new = []
        for uid in gather_uids:
            gather_uids_new.append(uid+id_offset)
        if len(gather_uids_new)>0:
            command['gather_uids']=gather_uids_new
        orig_command.append(command)
    return orig_commands


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



'''
 template : a generic approach to plain-text templating.

            The template dictionary contains two terms, the filename - which is the
            path to the template file, and the variables - a list of key-value pairs,
            everywhere the string in the key occurs it is substituted by the string
            in the value.

            For example:
                 "variables": { "var1" : "one", "var2" : "two2 }
                 applied to the string
                 "This string has a var2 in it"
                 would give
                 "This string has a two in it"
'''
def template(templateJSON={}):
    if not templateJSON:
        templateJSON = { "filename" : "../config_examples/strainrate2noiselog.json", "variables" : {}}
    lines=[]
    with open(template['filename'],'r') as f:
        lines = f.readlines()
    if 'variables' in template:
        for line in lines:
            [line.replace(key,val) for key,val in template['variables'].items() if key in line]
    return lines
