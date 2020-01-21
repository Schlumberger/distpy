# (C) 2020, Schlumberger. Refer to LICENSE
'''
 sgy : A module for reading SEG-Y files in 1-second chunks.
       Source document:
       https://seg.org/Portals/0/SEG/News%20and%20Resources/Technical%20Standards/seg_y_rev2_0-mar2017.pdf

       The headers are recorded into JSON format, which makes this reader slow. It is not intended to be high performance
       or to replace other useful SEGY readers such as obspy and segyio. The intention is to provide a useful tool for
       researchers to rapidly prototype, so complete data ingestion with header preservation was the priority

       IBM Float (data_type 1) is supported as we have seen vendors using it in DAS segy files

       However the following types are not current supported
       4  : 4-byte fixed-point with gain (obsolete)
       7  : 3-byte twos complement integer
       15 : 3-byte unsigned integer

       The ascii header is assumed to be cp500 format. If the ascii header looks like garbage try utf-8
       
       An attempt is made to correct the most common errors found in current DAS files that are in the wild.
       In particular some vendors writers fail to write the number of samples in a trace in the correct place
       when that number is larger than the SEG-Y rev.1 maximum. A fix is made to put match the number of samples
       with the correct header location for SEG-Y rev.2, which can handle very long traces.
       The aim is that, in the future, a SEGY writer could be constructed based on the JSON headers and npy files, and
       that recovered file would then be a well-formed SEG-Y rev.2
'''
import codecs
import os
import struct
import json
# This is the only numba dependency - we use it to vectorize the ibm2ieee
# https://github.com/numba/numba/blob/master/LICENSE

import numpy
import datetime
from numba import vectorize, float32, uint64, float64
import distpy.io_help.directory_services as directory_services

# If the SEGY is using the old-school IBM floats...
# this snippet is from https://stackoverflow.com/questions/7125890/python-unpack-ibm-32-bit-float-point
@vectorize([float64(uint64)])
def ibm2ieee(ibm):
    """
    Converts an IBM floating point number into IEEE format.
    :param: ibm - 32 bit unsigned integer: unpack('>L', f.read(4))
    """
    if ibm == 0:
        return 0.0
    sign = ibm >> 31 & 0x01
    sign = numpy.double(sign)
    exponent = ibm >> 24 & 0x7f
    exponent = numpy.double(exponent)-64.0;

    mantissa = (ibm & 0x00ffffff)
    mantissa = numpy.double(mantissa)
    mantissa = mantissa / numpy.double(numpy.power(2,24))

    return (1 - 2 * sign) * mantissa * numpy.power(16, exponent)

'''
 file_header : Read the main header.
'''
def file_header(fullHeader):
    # Useful constants and format codes
    formats = {}
    formats['ENDIAN'] = 'big'
    formats['DOUBLE'] = '>d'
    formats['UNSIGNED_LONG_LONG'] = '>Q'
    formats['UNSIGNED_CHAR'] = 'B'
    formats['UNSIGNED_SHORT'] = '>H'
    formats['UNSIGNED_INT'] = '>I'
    formats['INT'] = '>i'
    formats['SHORT'] = '>h'
    formats['EBCDIC'] = 'cp500'


    # Conversion to JSON - use python dictionaries
    header = {}
    # REV 2 - the endian test integer...check this first.
    header['endian_test_constant'] = struct.unpack(formats['UNSIGNED_INT'],fullHeader[3297:3301])[0]
    
    # Update the format codes now we konw the format
    if (header['endian_test_constant']==67305985):
        formats['ENDIAN']='little'
        formats['DOUBLE'] = '<d'
        formats['UNSIGNED_LONG_LONG'] = '<Q'
        formats['UNSIGNED_CHAR'] = 'B'
        formats['UNSIGNED_SHORT'] = '<H'
        formats['UNSIGNED_INT'] = '<I'
        formats['INT'] = '<i'
        formats['SHORT'] = '<h'

    #SEGY 3200 byte header
    #40 lines, 80 bytes each
    istep = 80
    iend = istep
    istart = 0
    asciiHeader = ""
    for item in range(40):
        byte80Part=fullHeader[istart:iend]
        istart+=istep
        iend+=istep
        asciiHeader += byte80Part.decode(formats['EBCDIC'])
        asciiHeader += "\n"
    header['ascii_header']=asciiHeader
    #print(asciiHeader)
    # https://seg.org/Portals/0/SEG/News%20and%20Resources/Technical%20Standards/seg_y_rev2_0-mar2017.pdf
    # table 2
    # Local variables for convenience
    ENDIAN = formats['ENDIAN']
    DOUBLE = formats['DOUBLE']
    UNSIGNED_LONG_LONG = formats['UNSIGNED_LONG_LONG']
    UNSIGNED_CHAR = formats['UNSIGNED_CHAR']
    UNSIGNED_SHORT = formats['UNSIGNED_SHORT']
    UNSIGNED_INT = formats['UNSIGNED_INT']
    INT = formats['INT']
    SHORT = formats['SHORT']   
    header['job_id'] = struct.unpack(INT,fullHeader[3200:3204])[0]
    header['line_number'] = struct.unpack(INT,fullHeader[3204:3208])[0]
    header['reel_number'] = struct.unpack(INT,fullHeader[3208:3212])[0]
    header['number_of_data_traces_per_ensemble'] = struct.unpack(SHORT,fullHeader[3212:3214])[0]
    header['number_of_auxiliary_traces_per_ensemble'] = struct.unpack(UNSIGNED_SHORT,fullHeader[3214:3216])[0]
    header['sample_interval'] = struct.unpack(UNSIGNED_SHORT,fullHeader[3216:3218])[0]
    header['sample_interval_of_original_field_recording'] = struct.unpack(UNSIGNED_SHORT,fullHeader[3218:3220])[0]
    header['number_of_samples_per_data_trace'] = struct.unpack(UNSIGNED_SHORT,fullHeader[3220:3222])[0]
    header['number_of_samples_per_data_trace_for_original_field_recording'] = struct.unpack(UNSIGNED_SHORT,fullHeader[3222:3224])[0]
    header['data_sample_format'] = struct.unpack(SHORT,fullHeader[3224:3226])[0]
    header['ensemble_fold'] = struct.unpack(SHORT,fullHeader[3226:3228])[0]
    header['trace_sorting_code'] = struct.unpack(SHORT,fullHeader[3228:3230])[0]
    header['vertical_sum_code'] = struct.unpack(SHORT,fullHeader[3230:3232])[0]
    header['sweep_frequency_at_start'] = struct.unpack(SHORT,fullHeader[3232:3234])[0]
    header['sweep_frequency_at_end'] = struct.unpack(SHORT,fullHeader[3234:3236])[0]
    header['sweep_length'] = struct.unpack(SHORT,fullHeader[3236:3238])[0]
    header['sweep_type_code']= struct.unpack(SHORT,fullHeader[3238:3240])[0]
    header['trace_number_of_sweep_channel'] = struct.unpack(SHORT,fullHeader[3240:3242])[0]
    header['sweep_trace_taper_length_at_start'] = struct.unpack(SHORT,fullHeader[3242:3244])[0]
    header['sweep_trace_taper_length_at_end'] = struct.unpack(SHORT,fullHeader[3244:3246])[0]
    header['taper_type'] = struct.unpack(SHORT,fullHeader[3246:3248])[0]
    header['correlated_data_traces'] = struct.unpack(SHORT,fullHeader[3248:3250])[0]
    header['binary_gain_recovered'] = struct.unpack(SHORT,fullHeader[3250:3252])[0]
    header['amplitude_recovery_method'] = struct.unpack(SHORT,fullHeader[3252:3254])[0]
    header['measurement_system'] = struct.unpack(SHORT,fullHeader[3254:3256])[0]
    header['impulse_signal_polarity'] = struct.unpack(SHORT,fullHeader[3256:3258])[0]
    header['vibratory_polarity_code'] = struct.unpack(SHORT,fullHeader[3258:3260])[0]
    header['extended_traces_per_ensemble'] = struct.unpack(INT,fullHeader[3260:3264])[0]
    header['extended_auxiliary_traces_per_ensemble'] = struct.unpack(INT,fullHeader[3264:3268])[0]
    header['extended_samples_per_trace'] = struct.unpack(INT,fullHeader[3268:3272])[0]
    header['extended_sample_interval'] = struct.unpack(DOUBLE,fullHeader[3272:3280])[0]
    header['extended_sample_interval_of_original'] = struct.unpack(DOUBLE,fullHeader[3280:3288])[0]
    header['extended_samples_per_trace_in_original'] = struct.unpack(INT,fullHeader[3288:3292])[0]
    header['extended_ensemble_fold'] = struct.unpack(INT,fullHeader[3292:3296])[0]
    header['endian_test_constant'] = struct.unpack(INT,fullHeader[3296:3300])[0]
    # 3301-3500 - rev 2
    header['segy_major_rev'] = struct.unpack(UNSIGNED_CHAR,fullHeader[3500:3501])[0]
    header['segy_minor_rev'] = struct.unpack(UNSIGNED_CHAR,fullHeader[3501:3502])[0]
    header['fixed_length_trace_flag'] = struct.unpack(SHORT,fullHeader[3502:3504])[0]
    header['number_of_extended_textual_headers'] = struct.unpack(SHORT,fullHeader[3504:3506])[0]
    header['max_additional_240byte_headers'] = struct.unpack(INT,fullHeader[3506:3510])[0]
    header['time_basis_code'] = struct.unpack(UNSIGNED_SHORT,fullHeader[3510:3512])[0]
    header['number_of_traces_in_file'] = struct.unpack(UNSIGNED_LONG_LONG,fullHeader[3512:3520])[0]
    header['byte_offset_of_first_trace'] = struct.unpack(UNSIGNED_LONG_LONG,fullHeader[3520:3528])[0]
    header['number_of_trailer_stanzas'] = struct.unpack(INT,fullHeader[3528:3532])[0]
    #3533-3600 unassigned
    
    # Correct a problem from older DAS file sthat failed to write the SEGY header correctly
    if (header['number_of_samples_per_data_trace']==55537)&(header['max_additional_240byte_headers']>55537):
        print('FOUND BAD SEGY...trying to fix')
        header['extended_samples_per_trace'] = header['max_additional_240byte_headers']
        header['max_additional_240byte_headers'] = 0

    if header['extended_samples_per_trace']==0:
        header['extended_samples_per_trace']=header['number_of_samples_per_data_trace']
    # QC for the header - using json pretty printing
    #print (json.dumps(header, sort_keys=True, indent=4, separators=(',', ': ')))

    # Next there are the Extended Textual File Headers
    etfh={}
    ETFH_LENGTH=3200
    for a in range(header['number_of_extended_textual_headers']):
        etfh[a]=rawObj.read(ETFH_LENGTH)
    #print (json.dumps(etfh, sort_keys=True, indent=4, separators=(',', ': ')))

    # Another common error in DAS SEGY files...
    if header['number_of_traces_in_file']==0:
        if header['ensemble_fold']==0:
            # Special case - stacked VSP...
            if header['number_of_data_traces_per_ensemble']==0:
                print("The ensemble_fold was not specified")
            else:
                header['number_of_traces_in_file']=header['number_of_data_traces_per_ensemble']
        else:
          header['number_of_traces_in_file']=header['ensemble_fold']
        
    print(header)
    return header, formats

'''
 trace_header : read the header from a given trace into a dictionary, for later writing to JSON
'''
def trace_header(stdHeaderBytes, formats):
    ENDIAN = formats['ENDIAN']
    DOUBLE = formats['DOUBLE']
    UNSIGNED_LONG_LONG = formats['UNSIGNED_LONG_LONG']
    UNSIGNED_CHAR = formats['UNSIGNED_CHAR']
    UNSIGNED_SHORT = formats['UNSIGNED_SHORT']
    UNSIGNED_INT = formats['UNSIGNED_INT']
    INT = formats['INT']
    SHORT = formats['SHORT']            
    stdHeader = {}
    stdHeader['trace_sequence_number_within_line']=struct.unpack(INT,stdHeaderBytes[0:4])[0]
    stdHeader['trace_sequence_number_within_file']=struct.unpack(INT,stdHeaderBytes[4:8])[0]
    stdHeader['original_field_record_number']=struct.unpack(INT,stdHeaderBytes[8:12])[0]
    stdHeader['trace_number_in_original_field_record']=struct.unpack(INT,stdHeaderBytes[12:16])[0]
    stdHeader['energy_source_point_number']=struct.unpack(INT,stdHeaderBytes[16:20])[0]
    stdHeader['ensemble_number']=struct.unpack(INT,stdHeaderBytes[20:24])[0]
    stdHeader['trace_number_in_ensemble']=struct.unpack(INT,stdHeaderBytes[24:28])[0]
    stdHeader['trace_identification_code']=struct.unpack(SHORT,stdHeaderBytes[28:30])[0]
    stdHeader['number_of_vertically_stacked_traces']=struct.unpack(SHORT,stdHeaderBytes[30:32])[0]
    stdHeader['number_of_horizontally_stacked_traces']=struct.unpack(SHORT,stdHeaderBytes[32:34])[0]
    stdHeader['data_use']=struct.unpack(SHORT,stdHeaderBytes[34:36])[0]
    stdHeader['distance_center_source_to_center_of_receiver_group']=struct.unpack(INT,stdHeaderBytes[36:40])[0]
    stdHeader['elevation_of_receiver_group']=struct.unpack(INT,stdHeaderBytes[40:44])[0]
    stdHeader['surface_elevation_at_source']=struct.unpack(INT,stdHeaderBytes[44:48])[0]
    stdHeader['source_depth_below_surface']=struct.unpack(INT,stdHeaderBytes[48:52])[0]
    stdHeader['seismic_datum_elevation_at_group']=struct.unpack(INT,stdHeaderBytes[52:56])[0]
    stdHeader['seismic_datum_elevation_at_source']=struct.unpack(INT,stdHeaderBytes[56:60])[0]
    stdHeader['water_column_height_at_source']=struct.unpack(INT,stdHeaderBytes[60:64])[0]
    stdHeader['water_column_height_at_group']=struct.unpack(INT,stdHeaderBytes[64:68])[0]
    stdHeader['scalar_applied_to_elevations_and_depths']=struct.unpack(SHORT,stdHeaderBytes[68:70])[0]
    stdHeader['scalar_applied_to_coordinates']=struct.unpack(SHORT,stdHeaderBytes[70:72])[0]
    stdHeader['source_x']=struct.unpack(INT,stdHeaderBytes[72:76])[0]
    stdHeader['source_y']=struct.unpack(INT,stdHeaderBytes[76:80])[0]
    stdHeader['receiver_x']=struct.unpack(INT,stdHeaderBytes[80:84])[0]
    stdHeader['receiver_y']=struct.unpack(INT,stdHeaderBytes[84:88])[0]
    stdHeader['coordinate_units']=struct.unpack(SHORT,stdHeaderBytes[88:90])[0]
    stdHeader['weathering_velocity']=struct.unpack(SHORT,stdHeaderBytes[90:92])[0]
    stdHeader['subweathering_velocity']=struct.unpack(SHORT,stdHeaderBytes[92:94])[0]
    stdHeader['uphole_time_at_source']=struct.unpack(SHORT,stdHeaderBytes[94:96])[0]
    stdHeader['uphole_time_at_group']=struct.unpack(SHORT,stdHeaderBytes[96:98])[0]
    stdHeader['source_static_correction']=struct.unpack(SHORT,stdHeaderBytes[98:100])[0]
    stdHeader['group_static_correction']=struct.unpack(SHORT,stdHeaderBytes[100:102])[0]
    stdHeader['total_static_applied']=struct.unpack(SHORT,stdHeaderBytes[102:104])[0]
    stdHeader['lag_time_A']=struct.unpack(SHORT,stdHeaderBytes[104:106])[0]
    stdHeader['lag_time_B']=struct.unpack(SHORT,stdHeaderBytes[106:108])[0]
    stdHeader['delay_recording_time']=struct.unpack(SHORT,stdHeaderBytes[108:110])[0]
    stdHeader['mute_time_start']=struct.unpack(SHORT,stdHeaderBytes[110:112])[0]
    stdHeader['mute_time_end']=struct.unpack(SHORT,stdHeaderBytes[112:114])[0]
    stdHeader['number_of_samples']=struct.unpack(SHORT,stdHeaderBytes[114:116])[0]
    stdHeader['sample_interval']=struct.unpack(SHORT,stdHeaderBytes[116:118])[0]
    stdHeader['gain_type']=struct.unpack(SHORT,stdHeaderBytes[118:120])[0]
    stdHeader['instrument_gain']=struct.unpack(SHORT,stdHeaderBytes[120:122])[0]
    stdHeader['instrument_initial_gain']=struct.unpack(SHORT,stdHeaderBytes[122:124])[0]
    stdHeader['correlated']=struct.unpack(SHORT,stdHeaderBytes[124:126])[0]
    stdHeader['sweep_frequency_at_start']=struct.unpack(SHORT,stdHeaderBytes[126:128])[0]
    stdHeader['sweep_frequency_at_end']=struct.unpack(SHORT,stdHeaderBytes[128:130])[0]
    stdHeader['sweep_length']=struct.unpack(SHORT,stdHeaderBytes[130:132])[0]
    stdHeader['sweep_type']=struct.unpack(SHORT,stdHeaderBytes[132:134])[0]
    stdHeader['sweep_taper_at_start']=struct.unpack(SHORT,stdHeaderBytes[134:136])[0]
    stdHeader['sweep_taper_at_end']=struct.unpack(SHORT,stdHeaderBytes[136:138])[0]
    stdHeader['sweep_taper_type']=struct.unpack(SHORT,stdHeaderBytes[138:140])[0]
    stdHeader['alias_filter_frequency']=struct.unpack(SHORT,stdHeaderBytes[140:142])[0]
    stdHeader['alias_filter_slope']=struct.unpack(SHORT,stdHeaderBytes[142:144])[0]
    stdHeader['notch_filter_frequency']=struct.unpack(SHORT,stdHeaderBytes[144:146])[0]
    stdHeader['notch_filter_slope']=struct.unpack(SHORT,stdHeaderBytes[146:148])[0]
    stdHeader['low_cut_frequency']=struct.unpack(SHORT,stdHeaderBytes[148:150])[0]
    stdHeader['high_cut_frequency']=struct.unpack(SHORT,stdHeaderBytes[150:152])[0]
    stdHeader['low_cut_slope']=struct.unpack(SHORT,stdHeaderBytes[152:154])[0]
    stdHeader['high_cut_slope']=struct.unpack(SHORT,stdHeaderBytes[154:156])[0]
    stdHeader['year']=struct.unpack(SHORT,stdHeaderBytes[156:158])[0]
    stdHeader['day']=struct.unpack(SHORT,stdHeaderBytes[158:160])[0]
    stdHeader['hour']=struct.unpack(SHORT,stdHeaderBytes[160:162])[0]
    stdHeader['minute']=struct.unpack(SHORT,stdHeaderBytes[162:164])[0]
    stdHeader['second']=struct.unpack(SHORT,stdHeaderBytes[164:166])[0]
    stdHeader['time_basis_code']=struct.unpack(SHORT,stdHeaderBytes[166:168])[0]
    stdHeader['trace_weighting_factor']=struct.unpack(SHORT,stdHeaderBytes[168:170])[0]
    stdHeader['roll_switch_position_one']=struct.unpack(SHORT,stdHeaderBytes[170:172])[0]
    stdHeader['first_trace_in_original']=struct.unpack(SHORT,stdHeaderBytes[172:174])[0]
    stdHeader['last_trace_in_original']=struct.unpack(SHORT,stdHeaderBytes[174:176])[0]
    stdHeader['gap_size']=struct.unpack(SHORT,stdHeaderBytes[176:178])[0]
    stdHeader['over_travel']=struct.unpack(SHORT,stdHeaderBytes[178:180])[0]
    stdHeader['ensemble_x']=struct.unpack(INT,stdHeaderBytes[180:184])[0]
    stdHeader['ensemble_y']=struct.unpack(INT,stdHeaderBytes[184:188])[0]
    stdHeader['inline_number']=struct.unpack(INT,stdHeaderBytes[188:192])[0]
    stdHeader['crossline_number']=struct.unpack(INT,stdHeaderBytes[192:196])[0]
    stdHeader['shotpoint_number']=struct.unpack(INT,stdHeaderBytes[196:200])[0]
    stdHeader['scalar_for_shotpoint_number']=struct.unpack(SHORT,stdHeaderBytes[200:202])[0]
    stdHeader['trace_value_units']=struct.unpack(SHORT,stdHeaderBytes[202:204])[0]
    stdHeader['transduction_constant_mantissa']=struct.unpack(INT,stdHeaderBytes[204:208])[0]
    stdHeader['transduction_constant_exponent']=struct.unpack(SHORT,stdHeaderBytes[208:210])[0]
    stdHeader['transduction_units']=struct.unpack(SHORT,stdHeaderBytes[210:212])[0]
    stdHeader['device_trace_identifier']=struct.unpack(SHORT,stdHeaderBytes[212:214])[0]
    stdHeader['scalar_applied_to_times']=struct.unpack(SHORT,stdHeaderBytes[214:216])[0]
    stdHeader['source_type']=struct.unpack(SHORT,stdHeaderBytes[216:218])[0]
    stdHeader['source_energy_direction_vertical']=struct.unpack(SHORT,stdHeaderBytes[218:220])[0]
    stdHeader['source_energy_direction_crossline']=struct.unpack(SHORT,stdHeaderBytes[220:222])[0]
    stdHeader['source_energy_direction_inline']=struct.unpack(SHORT,stdHeaderBytes[222:224])[0]
    stdHeader['source_measurement_mantissa']=struct.unpack(INT,stdHeaderBytes[224:228])[0]
    stdHeader['source_measurement_exponent']=struct.unpack(SHORT,stdHeaderBytes[228:230])[0]
    stdHeader['source_measurement_unit']=struct.unpack(SHORT,stdHeaderBytes[230:232])[0]
    return stdHeader

'''
 data_format : support for most of the SEG-Y data formats
'''
def data_format(header, formats):
    itype = header['data_sample_format']
    ENDIAN = formats['ENDIAN']
    SAMPLE_TYPE = list(">f")
    BYTES = 4
    bFail = True
    if ENDIAN=='little':
        SAMPLE_TYPE[0]='<'
    if itype==1:
        # 4-byte IBM floating-point
        # ibm2ieee(sort.unpack('>L', f.read(4)))
        SAMPLE_TYPE[1]='L'
        BYTES = 4
        bFail=False
    if itype==2:
        # 4-byte two's complement integer
        SAMPLE_TYPE[1]='i'
        BYTES = 4
        bFail=False
    if itype==3:
        # 2-byte two's complement integer
        SAMPLE_TYPE[1]='h'
        BYTES = 2
        bFail=False
    if itype==4:
        print('4-byte fixed-point with gain (obsolete) - is unsupported')
        BYTES = 4
        bFail=True
    if itype==5:
        # 4-byte IEEE floating-point
        SAMPLE_TYPE[1]='f'
        BYTES = 4
        bFail=False
    if itype==6:
        # 8-byte IEEE floating-point
        SAMPLE_TYPE[1]='d'
        BYTES = 8
        bFail=False
    if itype==7:
        print('3-byte twos complement integer - is unsupported')
        BYTES = 3
        bFail=True
    if itype==8:
        # 1-byte, twos complement integer
        SAMPLE_TYPE[1]='b'
        BYTES = 1
        bFail=False
    if itype==9:
        # 8-byte, twos complement
        SAMPLE_TYPE[1]='q'
        BYTES = 8
        bFail=False
    if itype==10:
        # 4-byte, unsigned integer
        SAMPLE_TYPE[1]='I'
        BYTES = 4
        bFail=False
    if itype==11:
        # 2-byte, unsigned integer
        SAMPLE_TYPE[1]='H'
        BYTES = 2
        bFail=False
    if itype==12:
        # 8-byte, unsigned integer
        SAMPLE_TYPE[1]='Q'
        BYTES = 8
        bFail=False
    if itype==15:
        print("3-byte unsigned integer - not supported")
        BYTES = 3
        bFail=True
    if itype==16:
        # 1-byte unsigned integer
        SAMPLE_TYPE[1]='B'
        BYTES = 1
        bFail=False
    sample_type = "".join(SAMPLE_TYPE)
    #print(bFail,sample_type)

    return sample_type, BYTES


    
'''This script takes a SEGY flat file and turns it into processable chunks
The headers become JSON objects
It implements the Rev 2 standard from
https://seg.org/Portals/0/SEG/News%20and%20Resources/Technical%20Standards/seg_y_rev2_0-mar2017.pdf
'''
def SEGYingest(segyfilename, outdir, overwrite=True):
    directory_services.makedir(outdir)
    headerDir = os.path.join(outdir,'json')
    directory_services.makedir(headerDir)
    tmpdir = os.path.join(outdir,'tmp')
    directory_services.makedir(tmpdir)
    
    
    # The 1-second blocks of data are memory mapped arrays...
    memmapList = []
    tmpList = []

    # Open the file for binary read
    rawObj=open(segyfilename,'rb')

    # hard-coded constants - from the standard
    FULL_HEADER_BYTES = 3600
    STANDARD_TRACE_HEADER_SIZE = 240

    # interpret the file header
    fullHeader = rawObj.read(FULL_HEADER_BYTES)
    header, formats = file_header(fullHeader)

    
    ntraces = header['number_of_traces_in_file']
    itrace=0
    # Local variables for convenience
    nx = header['number_of_traces_in_file']
    nt = header['extended_samples_per_trace']
    prf = int(1000000.0/header['sample_interval'])
    print('nx',nx)
    print('nt',nt)
    print('prf',prf)
    #print (json.dumps(header, sort_keys=True, indent=4, separators=(',', ': ')))
    xaxis = numpy.zeros((nx,1),dtype=numpy.double)
    ntOld=nt

    
    # read each trace
    for a in range(ntraces):
        nt=ntOld
        #print(a)
        stdHeaderBytes = rawObj.read(STANDARD_TRACE_HEADER_SIZE)
        stdHeader= trace_header(stdHeaderBytes, formats)
        # 233-240 : either binary zeros or the eight character trace header name SEG00000.
        #           May be ASCII or EBCDIC text.

        #if header['max_additional_240byte_headers']>0:
        #    print("additional trace headers found")
        # NOTE: this is wrong... should be stdHeader['scalar_applied_to_elevations_and_depths']...
        xaxis[a]=(stdHeader['elevation_of_receiver_group']/stdHeader['scalar_applied_to_coordinates'])

        # determine the format of the numbers
        sample_type, BYTES = data_format(header, formats)

        nsamps = header['number_of_samples_per_data_trace']
        nsampsBig = header['extended_samples_per_trace']
        if (nsampsBig>nsamps):
            nsamps=nsampsBig
        traceBytes = rawObj.read(nsamps*BYTES)
        # interpret trace
        trace = numpy.frombuffer(traceBytes,dtype=sample_type)
        if trace.shape[0]<nt:
            nt=trace.shape[0]
        # Have to support itype==1 - this is a bit clunky so look at this for refactoring
        itype = header['data_sample_format']
        if itype==1:
            trace = numpy.double(ibm2ieee(trace))    
        if a==0:
            #---RESTART--- when doing lots of SEGYs in parallel the compute system might fail
            #              so we can do a quick check to see whether all the 1-second files for t
            #              this SEGY already exist            
            datestring = str(stdHeader['year'])+' '+str(stdHeader['day']).zfill(3)+' '+str(stdHeader['hour']).zfill(2)+':'+str(stdHeader['minute']).zfill(2)+':'+str(stdHeader['second']).zfill(2)
            print(datestring)
            print('time_basis_code:',str(stdHeader['time_basis_code']))
            datestamp = datetime.datetime.strptime(datestring,'%Y %j %H:%M:%S')
            unixtime = int(datestamp.timestamp())
            bDone = True
            nsecs = 0
            for mapid in range(0,nt,prf):
                if not os.path.exists(os.path.join(outdir,str(unixtime+nsecs)+'.npy')):
                    bDone = False
                nsecs += 1
            if (bDone==True):
                print('skipping ', segyfilename, ' as already ingested')
                rawObj.close()
                return
            #---END OF RESTART----------------------------------------------
            # first time through... so we write the full header
            headerDir = os.path.join(headerDir,str(unixtime))
            if not os.path.exists(headerDir):
                os.makedirs(headerDir)
            with open(os.path.join(headerDir,str(unixtime)+'.json'),'w') as outJson:
                outJson.write(json.dumps(header, sort_keys=True, indent=4, separators=(',', ': ')))
            # create a memap list (because the SEGY is the 'wrong' way around...
            nsecs=0
            for mapid in range(0, nt, prf):
                fileout = os.path.join(tmpdir,str(unixtime+nsecs)+'.npy')
                #fileout = os.path.join(outdir,str(unixtime+nsecs)+'.npy')
                print(fileout)
                memmapList.append(numpy.memmap(fileout, dtype=numpy.double, mode='w+', shape=(nx,prf)))
                tmpList.append(fileout)
                nsecs=nsecs+1
        # Write this data to all the memmapped files
        nsecs = 0
        #print("AZURE-issue: ",nt,prf,trace.shape)
        for mapid in range(0,nt,prf):
            (memmapList[nsecs])[a,0:prf] = trace[mapid:mapid+prf]
            nsecs = nsecs+1
        # Write out the header at the end - because the directory is not set until we know about unixtimestamp
        # We need pretty print, even though we are writing to a stream, because we expect human inspection of the headers.
        with open(os.path.join(headerDir,str(a)+'.json'),'w') as outJson:
            outJson.write(json.dumps(stdHeader, sort_keys=True, indent=4, separators=(',', ': ')))
    for mapped in memmapList:        
        mapped.flush()
    # convert to numpy.save() so that we get metadata...
    nsecs=0
    print('Converting temporary files to permanent')
    for mapped in memmapList:
        fileout = os.path.join(outdir,str(unixtime+nsecs)+'.npy')
        print(fileout)
        numpy.save(fileout, numpy.array(mapped))
        nsecs=nsecs+1
        del mapped
        # override lazy behaviour...
        mapped = None
    memmapList = None
    # remove temporary files...
    for filename in tmpList:
        os.remove(filename)
    # Assumption that the fibre depth points do not change over the measurement period
    xaxisfilename = os.path.join(outdir,'measured_depth.npy')
    if not os.path.exists(xaxisfilename):
        numpy.save(xaxisfilename, xaxis)
    rawObj.close()



# TEST CODE...
def main():
    # A test of the SEGYingest...
    segyfilename = "{PATH}\\SegyPhase\\demo_003_F0086_S20180828_140213.862+0000_0000.sgy"
    outdir = "{PATH}\\sgy"

    SEGYingest(segyfilename, outdir, overwrite=False)
    # For google cloud use tensorflow for numpy.save and numpy.load... possibly restricted to float32...
    # https://stackoverflow.com/questions/41633748/load-numpy-array-in-google-cloud-ml-job

if __name__ == "__main__":
    main()
