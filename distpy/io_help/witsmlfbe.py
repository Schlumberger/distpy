# (C) 2020, Schlumberger. Refer to LICENSE

#
# witsmlfbe : handles reading and writing of the WITSML FBE format
#             and also contains tools for converting between LAS 2.0
#             and WITSML
#

import xml.etree.ElementTree as ET
import os
import numpy
import datetime
from os import listdir
from os.path import isfile, join
from xml.dom import minidom

##############################################################################
# CONFIGURATION
#################################
inpath='{PATH}/LAS/'
outpath='{PATH}/FBE/'
#################################


##############################################################################
# CONSTANTS
FAILED_MESSGE = 'Expecting a LAS 2.0 file with FBE data'
#LAS 2.0 hard coded numbers...
#0123456789012345678901234567890123
#DATE. 2016-12-13 07:13:49+00:00 : DATE
LAS2_DATESTART = 6
LAS2_DATEEND = 32
LAS2_ENDOFDESC = 33
LAS2_UNITSTART = 32
LAS2_UNITEND = 34
LAS2_DATE_TAG = 'DATE.'
LAS2_CURVE_TAG = '~Curve Information'
LAS2_PARAMS_TAG = '~Params'
LAS2_ASCII_TAG = '~ASCII'

# Sufficient specification for FBE in WITSML...
# common tags
SCHEMA = '{http://www.witsml.org/schemas/131}'
WITSML_NAME_TAG = SCHEMA + 'name'
NO_ATTR = {}
# LAS 2.0 does not tell us about the box or fiber used
UNKNOWN = 'UNKNOWN'
UNKNOWN_UID = {'uid': UNKNOWN}
UNKNOWN_UIDREF = {'uidRef': UNKNOWN}
TRUE = {'uid': '1'}
# There is no unique identifier for the LAS 2.0 data so we use this fixed dummy value
WELLLOG_UNIQUE = '5B8B6FB8-AACA-11E8-9637-0CC47A31FD84'
WELLLOG_UIDREF = {'uidRef': WELLLOG_UNIQUE}
# dummy well name
WELLNAME = 'FBR01'

ROOT_TAG = '{http://www.witsml.org/schemas/131}WITSMLComposite'
ROOT_ATTR = {'version': '1.3.1.1', '{http://www.w3.org/2001/XMLSchema-instance}schemaLocation': 'http://www.witsml.org/schemas/131 WITSML_composite.xsd'}
#...  tag    =  {http://www.witsml.org/schemas/131}wellSet
#...  attrib =  {}
#...  text   =  
WELLSET_TAG = SCHEMA + 'wellSet'
WELLSET_ATTR = NO_ATTR
#...|...  tag    =  {http://www.witsml.org/schemas/131}well
#...|...  attrib =  {'uid': 'FBR01'}
#...|...  text   =  
WELL_TAG = SCHEMA + 'well'
WELL_ATTR = {'uid': WELLNAME}
#...|...|...  tag    =  {http://www.witsml.org/schemas/131}name
#...|...|...  attrib =  {}
#...|...|...  text   =  FBR01
#...|...|...  tag    =  {http://www.witsml.org/schemas/131}wellboreSet
#...|...|...  attrib =  {}
#...|...|...  text   =  
WELL_NAME_TAG = WITSML_NAME_TAG
WELL_NAME_ATTR = NO_ATTR
WELL_NAME_TEXT = WELLNAME
WELLBORESET_TAG = SCHEMA + 'wellboreSet'
WELLBORESET_ATTR = NO_ATTR
#...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}wellbore
#...|...|...|...  attrib =  {'uid': 'FBR01'}
#...|...|...|...  text   =  
WELLBORE_TAG = SCHEMA + 'wellbore'
WELLBORE_ATTR = WELL_ATTR
#...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}name
#...|...|...|...|...  attrib =  {}
#...|...|...|...|...  text   =  FBR01
#...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}dtsInstalledSystemSet
#...|...|...|...|...  attrib =  {}
#...|...|...|...|...  text   =  
WELLBORE_NAME_TAG = WELL_NAME_TAG
WELLBORE_NAME_ATTR = WELL_NAME_ATTR
WELLBORE_NAME_TEXT = WELLNAME
# this is for DTS...re-used for DAS
DTSINSTALLEDSYSTEMSET_TAG = SCHEMA + 'dtsInstalledSystemSet'
DTSINSTALLEDSYSTEMSET_ATTR = NO_ATTR
#...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}dtsInstalledSystem
#...|...|...|...|...|...  attrib =  {'uid': '0C-C4-7A-31-FD-84'}
#...|...|...|...|...|...  text   =



DTSINSTALLEDSYSTEM_TAG = SCHEMA + 'dtsInstalledSystem'
#...we don't know the exact box in this case because it is not recorded in LAS 2.0
DTSINSTALLEDSYSTEM_ATTR = UNKNOWN_UID

# for the DTS Installed System Set branch
#...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}name
#...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...  text   =  hDvs_01
#...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}fiberInformation
#...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...  text   =  
DTSINSTALLEDSYSTEM_NAME_TAG = WITSML_NAME_TAG
DTSINSTALLEDSYSTEM_NAME_ATTR = NO_ATTR
DTSINSTALLEDSYSTEM_NAME_TEXT = UNKNOWN
FIBERINFORMATION_TAG = SCHEMA + 'fiberInformation'
FIBERINFORMATION_ATTR = NO_ATTR

#...|...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}fiber
#...|...|...|...|...|...|...|...  attrib =  {'uid': '1'}
#...|...|...|...|...|...|...|...  text   =  
FIBER_TAG = SCHEMA + 'fiber'
FIBER_ATTR = TRUE
#...|...|...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}name
#...|...|...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...|...|...  text   =  FBR01
#...|...|...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}mode
#...|...|...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...|...|...  text   =  singlemode
FIBER_NAME_TAG = WITSML_NAME_TAG
FIBER_NAME_ATTR = NO_ATTR
FIBER_NAME_TEXT = WELLNAME
FIBERMODE_TAG = SCHEMA + 'mode'
FIBERMODE_ATTR = NO_ATTR
FIBERMODE_TEXT = UNKNOWN

#...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}dtsMeasurementSet
#...|...|...|...|...  attrib =  {}
#...|...|...|...|...  text   =  
DTSMEASUREMENTSET_TAG = SCHEMA + 'dtsMeasurementSet'
DTSMEASUREMENTSET_ATTR = NO_ATTR
#...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}dtsMeasurement
#...|...|...|...|...|...  attrib =  {'uid': '5B8B6FB8-AACA-11E8-9637-0CC47A31FD84'}
#...|...|...|...|...|...  text   =  
DTSMEASUREMENT_TAG = SCHEMA + 'dtsMeasurement'
DTSMEASUREMENT_ATTR = UNKNOWN_UID
#...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}name
#...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...  text   =  INT_FBE_SIG
#...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}installedSystemUsed
#...|...|...|...|...|...|...  attrib =  {'uidRef': '0C-C4-7A-31-FD-84'}
#...|...|...|...|...|...|...  text   =  hDvs_01
#...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}dataInWellLog
#...|...|...|...|...|...|...  attrib =  {'uidRef': '5B8B6FB8-AACA-11E8-9637-0CC47A31FD84'}
#...|...|...|...|...|...|...  text   =  5B8B6FB8-AACA-11E8-9637-0CC47A31FD84
#...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}connectedToFiber
#...|...|...|...|...|...|...  attrib =  {'uidRef': '1'}
#...|...|...|...|...|...|...  text   =  FBR01
DTSMEASUREMENT_NAME_TAG = WITSML_NAME_TAG
DTSMEASUREMENT_NAME_ATTR = NO_ATTR
DTSMEASUREMENT_NAME_TEXT = UNKNOWN
INSTALLEDSYSTEMUSED_TAG = SCHEMA + 'installedSystemUsed'
INSTALLEDSYSTEMUSED_ATTR = UNKNOWN_UIDREF
INSTALLEDSYSTEMUSED_TEXT = UNKNOWN
DATAINWELLLOG_TAG = SCHEMA + 'dataInWellLog'
DATAINWELLLOG_ATTR = WELLLOG_UIDREF
DATAINWELLOG_TEXT = WELLLOG_UNIQUE
#...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}wellLogSet
#...|...|...|...|...  attrib =  {}
#...|...|...|...|...  text   =  
WELLLOGSET_TAG = SCHEMA + 'wellLogSet'
WELLLOGSET_ATTR = NO_ATTR
#...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}wellLog
#...|...|...|...|...|...  attrib =  {'uid': '5B8B6FB8-AACA-11E8-9637-0CC47A31FD84'}
#...|...|...|...|...|...  text   = 
# wellLogSet
WELLLOG_TAG = SCHEMA + 'wellLog'
WELLLOG_ATTR = WELLLOG_UIDREF
#...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}name
#...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...  text   =  5B8B6FB8-AACA-11E8-9637-0CC47A31FD84
#...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}serviceCompany
#...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...  text   =  Schlumberger
#...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}creationDate
#...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...  text   =  2018-08-28T13:57:13+00:00
#...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}indexType
#...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...  text   =  depth
#...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}nullValue
#...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...  text   =  NULL
#...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}logCurveInfo
#...|...|...|...|...|...|...  attrib =  {'uid': '1'}
#...|...|...|...|...|...|...  text   =
# wellLog
WELLLOG_NAME_TAG = WITSML_NAME_TAG
WELLLOG_NAME_ATTR = NO_ATTR
WELLLOG_NAME_TEXT = WELLLOG_UNIQUE
SERVICECOMPANY_TAG = SCHEMA + 'serviceCompany'
SERVICECOMPANY_ATTR = NO_ATTR
SERVICECOMPANY_NAME = UNKNOWN
CREATIONDATE_TAG = SCHEMA + 'creationDate'
CREATIONDATE_ATTR = NO_ATTR
##############...LAS 2.0 data in here###########
CREATIONDATE_TEXT = '2018-08-28T13:57:13+00:00'
INDEXTYPE_TAG = SCHEMA + 'indexType'
INDEXTYPE_ATTR = NO_ATTR
INDEXTYPE_TEXT = 'depth'
NULLVALUE_TAG = SCHEMA + 'nullValue'
NULLVALUE_ATTR = NO_ATTR
NULLVALUE_TEXT = 'NULL'
# AT THIS POINT WE GET INTO LIST OF INFORMATION THAT LAS 2.0 provides...
LOGCURVEINFO_TAG = SCHEMA + 'logCurveInfo'
LOGCURVEINFO_ATTR = TRUE
#...|...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}mnemonic
#...|...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...|...  text   =  DEPTH
#...|...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}unit
#...|...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...|...  text   =  m
#...|...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}curveDescription
#...|...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...|...  text   =  depth along the fiber
MNEMONIC_TAG = SCHEMA + 'mnemonic'
MNEMONIC_ATTR = NO_ATTR
UNIT_TAG = SCHEMA + 'unit'
UNIT_ATTR = NO_ATTR
CURVEDESCRIPTION_TAG = SCHEMA + 'curveDescription'
CURVEDESCRIPTION_ATTR = NO_ATTR
#...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}blockInfo
#...|...|...|...|...|...|...  attrib =  {'uid': '1'}
#...|...|...|...|...|...|...  text   =  
BLOCKINFO_TAG = SCHEMA + 'blockInfo'
BLOCKINFO_ATTR = TRUE
#...|...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}indexType
#...|...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...|...  text   =  length
#...|...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}direction
#...|...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...|...  text   =  increasing
#...|...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}indexCurve
#...|...|...|...|...|...|...|...  attrib =  {'columnIndex': '1'}
#...|...|...|...|...|...|...|...  text   =  LENGTH
#...|...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}blockCurveInfo
#...|...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...|...  text   =
INDEXTYPE_TAG = SCHEMA + 'indexType'
INDEXTYPE_ATTR = NO_ATTR
DIRECTION_TAG = SCHEMA + 'direction'
DIRECTION_ATTR = NO_ATTR
INDEXCURVE_TAG = SCHEMA + 'indexCurve'
INDEXCURVE_ATTR = {'columnIndex': '1'}
BLOCKCURVEINFO_TAG = SCHEMA + 'blockCurveInfo'
BLOCKCURVEINFO_ATTR = NO_ATTR
#...|...|...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}curveId
#...|...|...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...|...|...  text   =  1
#...|...|...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}columnIndex
#...|...|...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...|...|...  text   =  1
CURVEID_TAG = SCHEMA + 'curveId'
CURVEID_ATTR = NO_ATTR
COLUMNINDEX_TAG = SCHEMA + 'columnIndex'
COLUMNINDEX_ATTR = NO_ATTR
#...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}logData
#...|...|...|...|...|...|...  attrib =  {}
#...|...|...|...|...|...|...  text   =  
LOGDATA_TAG = SCHEMA + 'logData'
LOGDATA_ATTR = NO_ATTR
#...|...|...|...|...|...|...|...  tag    =  {http://www.witsml.org/schemas/131}data
#...|...|...|...|...|...|...|...  attrib =  {'id': '1'}
#...|...|...|...|...|...|...|...  text   =  -2.02918,574.19,98.1762,25.1918,17.0701,27.9438
DATA_TAG = SCHEMA + 'data'
DATA_ATTR = {'id': '1'}

CONNECTEDTOFIBER_TAG = SCHEMA + 'connectedToFiber'
CONNECTEDTOFIBER_ATTR = TRUE

##################################################################################################
# FUNCTIONS:
##for testing
'''
 printsubbranch : used in testing whether the formed XML is correct
'''
def printsubbranch(elements, dots):
    for elem in elements:
        print(dots,' tag    = ',elem.tag)
        print(dots,' attrib = ',elem.attrib)
        print(dots,' text   = ',elem.text)
        more_elements = elem.findall("./")
        if len(more_elements)>0:
            printsubbranch(more_elements, dots+'|...')

'''
  wrong_file_type : only LAS 2.0 is supported
'''
def wrong_file_type(lines):
    if lines[1][:11] != 'VERS.   2.0':
        return True    
    return False

'''
 lineIndex : determine the line number in the original data, corresponding to
             a given XML tab
'''
def lineIndex(lines,tag):
    test_len=len(tag)
    icount=0
    for line in lines:
        if line[:test_len] == tag:
            return icount,line
        icount+=1
    return 1,'none'

'''
 generic_curves : WITSML headers for the FBE style (band-limited data)
'''
def generic_curves(low_freq,high_freq, prf):
    curves = []
    #band-00
    curve={}
    curve['mnemonic'] = 'band_0-nyquist'
    curve['unit'] = 'unitless'
    curve['description'] = 'from 0 Hz to Nyquist Hz (prf='+str(prf)+ ')'
    curves.append(curve)

    for a in range(len(low_freq)):
        curve={}
        if a<9:
            curve['mnemonic'] = 'band_0' +str(a+1)+'_'+str(low_freq[a])+'-'+str(high_freq[a])
        else:
            curve['mnemonic'] = 'band_' +str(a+1)+'_'+str(low_freq[a])+'-'+str(high_freq[a])
        curve['unit'] = 'unitless'
        curve['description'] = 'from '+str(low_freq[a])+ ' Hz to ' +str(high_freq[a])+' Hz (prf='+str(prf) + ')'
        curves.append(curve)
    return curves


'''
 set_unkown_constants : allows over-writing of the defaults 
'''
def set_unknown_constants(data_style):
    global UNKNOWN, UNKNOWN_UID, UNKNOWN_UIDREF, DTSINSTALLEDSYSTEM_ATTR, DTSINSTALLEDSYSTEM_NAME_TEXT
    global FIBERMODE_TEXT, DTSMEASUREMENT_ATTR, DTSMEASUREMENT_NAME_TEXT, INSTALLEDSYSTEMUSED_ATTR
    global INSTALLEDSYSTEMUSED_TEXT, SERVICECOMPANY_NAME
    # The option to update the UNKNOWN to be something more useful
    UNKNOWN = data_style
    UNKNOWN_UID = {'uid': data_style}
    UNKNOWN_UIDREF = {'uidRef': data_style}
    DTSINSTALLEDSYSTEM_ATTR = UNKNOWN_UID
    DTSINSTALLEDSYSTEM_NAME_TEXT = UNKNOWN
    FIBERMODE_TEXT = UNKNOWN
    DTSMEASUREMENT_ATTR = UNKNOWN_UID
    DTSMEASUREMENT_NAME_TEXT = UNKNOWN
    INSTALLEDSYSTEMUSED_ATTR = UNKNOWN_UIDREF
    INSTALLEDSYSTEMUSED_TEXT = UNKNOWN
    SERVICECOMPANY_NAME = UNKNOWN


'''
  las2fbe : Given a LAS file, create a WITSML format fbe for Techlog.

            NOTE: this is not a generic reader of the LAS format and it is also
                  not a generic writer for WITSML 1.3.1.1

            Only submit files in LAS which have the column format of
            depth followed by FBE bands. No other files are supported.
'''
def las2fbe(filename):
    # import the LAS as lines of text
    f=open(filename,"r")
    lines = f.readlines()
    f.close()

    root = ET.Element(ROOT_TAG,ROOT_ATTR)
    wellset = ET.SubElement(root, WELLSET_TAG, WELLSET_ATTR)
    well = ET.SubElement(wellset, WELL_TAG, WELL_ATTR)
    

    wellname = ET.SubElement(well, WELL_NAME_TAG, WELL_NAME_ATTR)
    wellname.text = WELL_NAME_TEXT
    
    wellboreSet = ET.SubElement(well, WELLBORESET_TAG, WELLBORESET_ATTR)
    # The following level down is the wellbore
    wellbore = ET.SubElement(wellboreSet, WELLBORE_TAG, WELLBORE_ATTR)
    # wellbore has 4 elements
    wellborename = ET.SubElement(wellbore, WELLBORE_NAME_TAG, WELLBORE_NAME_ATTR)
    wellborename.text = WELLBORE_NAME_TEXT
    dtsInstalledSystemSet = ET.SubElement(wellbore, DTSINSTALLEDSYSTEMSET_TAG, DTSINSTALLEDSYSTEMSET_ATTR)
    dtsMeasurementSet = ET.SubElement(wellbore, DTSMEASUREMENTSET_TAG, DTSMEASUREMENTSET_ATTR)
    wellLogSet = ET.SubElement(wellbore, WELLLOGSET_TAG, WELLLOGSET_ATTR)
    # ....dtsInstalledSystemSet has information on the DTS/DAS system used
    dtsInstalledSystem = ET.SubElement(dtsInstalledSystemSet, DTSINSTALLEDSYSTEM_TAG, DTSINSTALLEDSYSTEM_ATTR)
    # ....|....dtsInstalledSystem has information on the DTS/DAS system used
    dtsInstalledSystemName = ET.SubElement(dtsInstalledSystem, DTSINSTALLEDSYSTEM_NAME_TAG, DTSINSTALLEDSYSTEM_NAME_ATTR)
    dtsInstalledSystemName.text = DTSINSTALLEDSYSTEM_NAME_TEXT
    fiberInformation = ET.SubElement(dtsInstalledSystem, FIBERINFORMATION_TAG, FIBERINFORMATION_ATTR)
    # ....|....|....fiberInformation - including mode
    fiber = ET.SubElement(fiberInformation, FIBER_TAG, FIBER_ATTR)
    # ....|....|....|....fiber - the name and mode of the fiber
    fiberName = ET.SubElement(fiber, FIBER_NAME_TAG, FIBER_NAME_ATTR)
    fiberName.text = FIBER_NAME_TEXT
    mode = ET.SubElement(fiber, FIBERMODE_TAG, FIBERMODE_ATTR)
    mode.text = FIBERMODE_TEXT
    # ....dtsMeasurementSet
    dtsMeasurement = ET.SubElement(dtsMeasurementSet, DTSMEASUREMENT_TAG, DTSMEASUREMENT_ATTR)
    # ....|....dtsMeasurement
    dtsMeasurementName = ET.SubElement(dtsMeasurement, DTSMEASUREMENT_NAME_TAG,DTSMEASUREMENT_NAME_ATTR)
    dtsMeasurementName.text = DTSMEASUREMENT_NAME_TEXT
    installedSystemUsed = ET.SubElement(dtsMeasurement, INSTALLEDSYSTEMUSED_TAG,INSTALLEDSYSTEMUSED_ATTR)
    installedSystemUsed.text = INSTALLEDSYSTEMUSED_TEXT
    dataInWellLog = ET.SubElement(dtsMeasurement, DATAINWELLLOG_TAG, DATAINWELLLOG_ATTR)
    dataInWellLog.text = WELLLOG_UNIQUE
    connectedToFiber = ET.SubElement(dtsMeasurement, CONNECTEDTOFIBER_TAG, CONNECTEDTOFIBER_ATTR)
    # ....wellLogSet
    wellLog = ET.SubElement(wellLogSet, WELLLOG_TAG, WELLLOG_ATTR)
    # ....|....wellLog
    wellLogName = ET.SubElement(wellLog, WELLLOG_NAME_TAG, WELLLOG_NAME_ATTR)
    wellLogName.text = WELLLOG_NAME_TEXT
    serviceCompany = ET.SubElement(wellLog, SERVICECOMPANY_TAG, SERVICECOMPANY_ATTR)
    serviceCompany.text = SERVICECOMPANY_NAME
    creationDate = ET.SubElement(wellLog, CREATIONDATE_TAG, CREATIONDATE_ATTR)
    #############LAS 2.0 INFO IN HERE.......###################
    istart, line = lineIndex(lines,LAS2_DATE_TAG)
    creationDate.text = line[LAS2_DATESTART:LAS2_DATEEND]
    indexType = ET.SubElement(wellLog, INDEXTYPE_TAG, INDEXTYPE_ATTR)
    indexType.text = INDEXTYPE_TEXT
    nullValue = ET.SubElement(wellLog, NULLVALUE_TAG, NULLVALUE_ATTR)
    nullValue.text = NULLVALUE_TEXT

    istart, line = lineIndex(lines,LAS2_CURVE_TAG)
    iend, line = lineIndex(lines,LAS2_PARAMS_TAG)
    lognumber=1
    band=1
    # populate the column descriptions
    for line in lines[istart+1:iend-1]:
        logCurveInfo = ET.SubElement(wellLog, LOGCURVEINFO_TAG, {'uid' : str(lognumber)} )
        blockInfo = ET.SubElement(wellLog, BLOCKINFO_TAG, {'uid' : str(lognumber)})
        
        mnemonic = ET.SubElement(logCurveInfo, MNEMONIC_TAG, MNEMONIC_ATTR)
        unit = ET.SubElement(logCurveInfo, UNIT_TAG, UNIT_ATTR)
        curveDescription = ET.SubElement(logCurveInfo,CURVEDESCRIPTION_TAG, CURVEDESCRIPTION_ATTR)
        if line[:4] == 'DEPT':
            mnemonic.text = 'DEPTH'
            unit.text = line[LAS2_UNITSTART:LAS2_UNITEND]
            curveDescription.text = 'depth along the fiber'
            indexType = ET.SubElement(blockInfo, INDEXTYPE_TAG, INDEXTYPE_ATTR)
            indexType.text = 'length'
            direction = ET.SubElement(blockInfo, DIRECTION_TAG, DIRECTION_ATTR)
            direction.text = 'increasing'
            indexCurve = ET.SubElement(blockInfo, INDEXCURVE_TAG, INDEXCURVE_ATTR)
            indexCurve.text = str(int(1))
        else:
            textpart = 'BAND-'
            if band<10:
                textpart +='0'
            textpart+=str(int(band))
            mnemonic.text = textpart
            unit.text = 'unitless'
            curveDescription = 'Frequency Band Energy '+line[:LAS2_ENDOFDESC]
            blockCurveInfo = ET.SubElement(blockInfo, BLOCKCURVEINFO_TAG, BLOCKCURVEINFO_ATTR)
            curveId = ET.SubElement(blockCurveInfo, CURVEID_TAG, CURVEID_ATTR)
            curveId.text = str(int(band))
            columnIndex = ET.SubElement(blockCurveInfo, COLUMNINDEX_TAG, COLUMNINDEX_ATTR)
            columnIndex.text = curveId.text          
            band+=1
        lognumber+=1
    # populate the log data
    logData = ET.SubElement(wellLog, LOGDATA_TAG, LOGDATA_ATTR)
    istart, line = lineIndex(lines,LAS2_ASCII_TAG)
    for a in range(istart+1,len(lines)):
        line = lines[a]
        if len(line)>1:
            data = ET.SubElement(logData,DATA_TAG, DATA_ATTR)
            line = line.rstrip()
            line = line.lstrip()
            line = line.replace('  ',' ')            
            data.text = str(''.join(line.replace(" ",", ")))
        
        
    return root


'''
 witsml_fbe : Given the information, write the wistml.
'''
def witsml_fbe(datestring, xaxis, curves, curveData, data_style='UNKNOWN'):
    set_unknown_constants(data_style)
    #print('xaxis.shape',xaxis.shape)
    root = ET.Element(ROOT_TAG,ROOT_ATTR)
    wellset = ET.SubElement(root, WELLSET_TAG, WELLSET_ATTR)
    well = ET.SubElement(wellset, WELL_TAG, WELL_ATTR)
    

    wellname = ET.SubElement(well, WELL_NAME_TAG, WELL_NAME_ATTR)
    wellname.text = WELL_NAME_TEXT
    
    wellboreSet = ET.SubElement(well, WELLBORESET_TAG, WELLBORESET_ATTR)
    # The following level down is the wellbore
    wellbore = ET.SubElement(wellboreSet, WELLBORE_TAG, WELLBORE_ATTR)
    # wellbore has 4 elements
    wellborename = ET.SubElement(wellbore, WELLBORE_NAME_TAG, WELLBORE_NAME_ATTR)
    wellborename.text = WELLBORE_NAME_TEXT
    dtsInstalledSystemSet = ET.SubElement(wellbore, DTSINSTALLEDSYSTEMSET_TAG, DTSINSTALLEDSYSTEMSET_ATTR)
    dtsMeasurementSet = ET.SubElement(wellbore, DTSMEASUREMENTSET_TAG, DTSMEASUREMENTSET_ATTR)
    wellLogSet = ET.SubElement(wellbore, WELLLOGSET_TAG, WELLLOGSET_ATTR)
    # ....dtsInstalledSystemSet has information on the DTS/DAS system used
    dtsInstalledSystem = ET.SubElement(dtsInstalledSystemSet, DTSINSTALLEDSYSTEM_TAG, DTSINSTALLEDSYSTEM_ATTR)
    # ....|....dtsInstalledSystem has information on the DTS/DAS system used
    dtsInstalledSystemName = ET.SubElement(dtsInstalledSystem, DTSINSTALLEDSYSTEM_NAME_TAG, DTSINSTALLEDSYSTEM_NAME_ATTR)
    dtsInstalledSystemName.text = DTSINSTALLEDSYSTEM_NAME_TEXT
    fiberInformation = ET.SubElement(dtsInstalledSystem, FIBERINFORMATION_TAG, FIBERINFORMATION_ATTR)
    # ....|....|....fiberInformation - including mode
    fiber = ET.SubElement(fiberInformation, FIBER_TAG, FIBER_ATTR)
    # ....|....|....|....fiber - the name and mode of the fiber
    fiberName = ET.SubElement(fiber, FIBER_NAME_TAG, FIBER_NAME_ATTR)
    fiberName.text = FIBER_NAME_TEXT
    mode = ET.SubElement(fiber, FIBERMODE_TAG, FIBERMODE_ATTR)
    mode.text = FIBERMODE_TEXT
    # ....dtsMeasurementSet
    dtsMeasurement = ET.SubElement(dtsMeasurementSet, DTSMEASUREMENT_TAG, DTSMEASUREMENT_ATTR)
    # ....|....dtsMeasurement
    dtsMeasurementName = ET.SubElement(dtsMeasurement, DTSMEASUREMENT_NAME_TAG,DTSMEASUREMENT_NAME_ATTR)
    dtsMeasurementName.text = DTSMEASUREMENT_NAME_TEXT
    installedSystemUsed = ET.SubElement(dtsMeasurement, INSTALLEDSYSTEMUSED_TAG,INSTALLEDSYSTEMUSED_ATTR)
    installedSystemUsed.text = INSTALLEDSYSTEMUSED_TEXT
    dataInWellLog = ET.SubElement(dtsMeasurement, DATAINWELLLOG_TAG, DATAINWELLLOG_ATTR)
    dataInWellLog.text = WELLLOG_UNIQUE
    connectedToFiber = ET.SubElement(dtsMeasurement, CONNECTEDTOFIBER_TAG, CONNECTEDTOFIBER_ATTR)
    # ....wellLogSet
    wellLog = ET.SubElement(wellLogSet, WELLLOG_TAG, WELLLOG_ATTR)
    # ....|....wellLog
    wellLogName = ET.SubElement(wellLog, WELLLOG_NAME_TAG, WELLLOG_NAME_ATTR)
    wellLogName.text = WELLLOG_NAME_TEXT
    serviceCompany = ET.SubElement(wellLog, SERVICECOMPANY_TAG, SERVICECOMPANY_ATTR)
    serviceCompany.text = SERVICECOMPANY_NAME
    creationDate = ET.SubElement(wellLog, CREATIONDATE_TAG, CREATIONDATE_ATTR)
    #############LAS 2.0 INFO IN HERE.......###################
    creationDate.text = datestring
    indexType = ET.SubElement(wellLog, INDEXTYPE_TAG, INDEXTYPE_ATTR)
    indexType.text = INDEXTYPE_TEXT
    nullValue = ET.SubElement(wellLog, NULLVALUE_TAG, NULLVALUE_ATTR)
    nullValue.text = NULLVALUE_TEXT

    # The DEPTH part
    lognumber=1
    logCurveInfo = ET.SubElement(wellLog, LOGCURVEINFO_TAG, {'uid' : str(lognumber)} )
    blockInfo = ET.SubElement(wellLog, BLOCKINFO_TAG, {'uid' : str(lognumber)})
        
    mnemonic = ET.SubElement(logCurveInfo, MNEMONIC_TAG, MNEMONIC_ATTR)
    unit = ET.SubElement(logCurveInfo, UNIT_TAG, UNIT_ATTR)
    curveDescription = ET.SubElement(logCurveInfo,CURVEDESCRIPTION_TAG, CURVEDESCRIPTION_ATTR)
    mnemonic.text = 'DEPTH'
    unit.text = 'm'
    curveDescription.text = 'depth along the fiber'
    indexType = ET.SubElement(blockInfo, INDEXTYPE_TAG, INDEXTYPE_ATTR)
    indexType.text = 'length'
    direction = ET.SubElement(blockInfo, DIRECTION_TAG, DIRECTION_ATTR)
    direction.text = 'increasing'
    indexCurve = ET.SubElement(blockInfo, INDEXCURVE_TAG, INDEXCURVE_ATTR)
    indexCurve.text = str(int(1))
    lognumber+=1

    # The data
    band=1
    # populate the column descriptions
    for curve in curves:
        logCurveInfo = ET.SubElement(wellLog, LOGCURVEINFO_TAG, {'uid' : str(lognumber)} )
        blockInfo = ET.SubElement(wellLog, BLOCKINFO_TAG, {'uid' : str(lognumber)})
        
        mnemonic = ET.SubElement(logCurveInfo, MNEMONIC_TAG, MNEMONIC_ATTR)
        unit = ET.SubElement(logCurveInfo, UNIT_TAG, UNIT_ATTR)
        curveDescription = ET.SubElement(logCurveInfo,CURVEDESCRIPTION_TAG, CURVEDESCRIPTION_ATTR)
        #'BAND-00'
        mnemonic.text = curve['mnemonic']
        #'unitless'
        unit.text = curve['unit']
        #'Frequency Band Energy 0-Nyquist'
        curveDescription.text = curve['description']

        blockCurveInfo = ET.SubElement(blockInfo, BLOCKCURVEINFO_TAG, BLOCKCURVEINFO_ATTR)
        curveId = ET.SubElement(blockCurveInfo, CURVEID_TAG, CURVEID_ATTR)
        curveId.text = str(int(band))
        columnIndex = ET.SubElement(blockCurveInfo, COLUMNINDEX_TAG, COLUMNINDEX_ATTR)
        columnIndex.text = curveId.text          
        band+=1
        lognumber+=1
    # populate the log data
    logData = ET.SubElement(wellLog, LOGDATA_TAG, LOGDATA_ATTR)

    xaxis = numpy.squeeze(xaxis)
    # sometimes we have only a single column...sometimes many columns...
    if curveData.ndim>1:
        for a in range(curveData.shape[0]):
            line = curveData[a]
            data = ET.SubElement(logData,DATA_TAG, DATA_ATTR)
            data.text = str(xaxis[a]) + ', ' + ', '.join(str(val) for val in line)
    else:
        for a in range(curveData.shape[0]):
            line = curveData[a]
            data = ET.SubElement(logData,DATA_TAG, DATA_ATTR)
            data.text = str(xaxis[a]) + ', ' +  str(line)        
    return root

'''
  writeFBE : Separates the writing part, in case we want to
             support things other than files
'''
def writeFBE(pathstem,fileout, root):
    tree = ET.ElementTree(root)
    ET.register_namespace("", "http://www.witsml.org/schemas/131")
    #fileout = filename.replace('.las','.fbe')
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    print(os.path.join(pathstem,fileout+'.fbe'))
    with open(os.path.join(pathstem,fileout+'.fbe'), "w") as f:
        f.write(xmlstr)


'''
 recursive_elem_counter : loops over XML recursively accumulating data locations
'''
def recursive_elem_counter(elem, list_of_files, depth_axis):
    for subelem in elem:
        if subelem.tag==LOGCURVEINFO_TAG:
            for subsubelem in subelem:
                if subsubelem.tag==MNEMONIC_TAG:
                    list_of_files.append(subsubelem.text)
        if subelem.tag==DATA_TAG:
            values = subelem.text.split(',')
            fvalues = [float(f) for f in values]
            depth_axis.append(fvalues[0])
        recursive_elem_counter(subelem, list_of_files, depth_axis)

'''
 recursive_elem_ingest : loops over XML recursively reading data
'''
def recursive_elem_ingest(elem, list_of_memmaps, idx, xidx, time_axis):
    for subelem in elem:
        if subelem.tag==DATA_TAG:
            values = subelem.text.split(',')
            fvalues = [float(f) for f in values]
            for a in range(len(list_of_memmaps)):
                list_of_memmaps[a][xidx,idx]=fvalues[a+1]
            xidx=xidx+1
        if subelem.tag==CREATIONDATE_TAG:
            value = datetime.datetime.strptime(subelem.text,"%Y-%m-%dT%H:%M:%S+00:00")
            time_axis.append(value.timestamp())
        recursive_elem_ingest(subelem, list_of_memmaps, idx, xidx, time_axis)   

'''
 recursive_elem_accumulator : used in the average and std. dev. determination
                              over multiple WITSML files
'''
def recursive_elem_accumulator(elem, values,squares, xidx):
    for subelem in elem:
        if subelem.tag==DATA_TAG:
            valueSet = subelem.text.split(',')
            fvalues = [float(f) for f in valueSet]
            for a in range(values.shape[1]):
                values[xidx,a]=values[xidx,a]+fvalues[a]
                squares[xidx,a]=squares[xidx,a]+(fvalues[a]*fvalues[a])
            xidx=xidx+1
        recursive_elem_accumulator(subelem, values, squares, xidx)   

'''
 recursive_elem_egest: used in writing WITSML to an XML structure for later
                       export
'''
def recursive_elem_egest(elem, values, xidx):
    for subelem in elem:
        if subelem.tag==DATA_TAG:
            subelem.text = ', '.join(str(val) for val in values[xidx])                
            xidx=xidx+1
        recursive_elem_egest(subelem, values, xidx)   

'''
 averageFBE :  Accumulate average FBE results over long time periods
               ...we take a whole directory in and write a whole directory out(?)
               ...with additional curves corresponding to std. dev.
'''
def averageFBE(datafiles, basedir, dirout, dirout_std):
    DEPTH = 'DEPTH'
    TIME = 'TIME'
    TMP = 'tmp'
    
    # The number of time samples
    nt = len(datafiles)

    
    # First file gives us nx and the list of output files
    filename = os.path.join(basedir,datafiles[0])
    tree = ET.parse(filename)
    root = tree.getroot()
    list_of_files = []
    depth_axis_values  =[]
    time_axis_values = []
    recursive_elem_counter(root, list_of_files, depth_axis_values)
    nx = len(depth_axis_values)

    set_of_values = numpy.zeros((nx,len(list_of_files)),dtype=numpy.double)
    set_of_squares = numpy.zeros((nx,len(list_of_files)),dtype=numpy.double)
    meanOut = numpy.zeros((nx,len(list_of_files)),dtype=numpy.double)
    stdOut = numpy.zeros((nx,len(list_of_files)),dtype=numpy.double)
        
    # loop and write...
    idx = 0
    for fname in datafiles:
        tree = ET.parse(os.path.join(basedir,fname))
        root = tree.getroot()
        xidx=0
        # to here
        recursive_elem_accumulator(root, set_of_values, set_of_squares, xidx)
        idx = idx + 1
        meanOut = set_of_values/idx
        stdOut = numpy.sqrt((set_of_squares/idx)-(meanOut*meanOut))
        # The stdOut first column should be MD just as for the meanOut
        stdOut[:,0]=meanOut[:,0]
        xidx=0
        recursive_elem_egest(root, meanOut, xidx)
        writeFBE(dirout,fname[:-4], root)
        # reset the root...
        xidx=0
        recursive_elem_egest(root, stdOut, xidx)
        writeFBE(dirout_std,fname[:-4], root)
        
'''
 readFBE : read a directory of *.fbe files and turn them into a directory containing a depth axis, a time axis and a 2D array of values
           - a format that is much easier to work with in reprocessing and interpretation flows.
'''
def readFBE(datafiles,dirout):
    DEPTH = 'DEPTH'
    TIME = 'TIME'
    TMP = 'tmp'
    
    # The number of time samples
    nt = len(datafiles)

    
    # First file gives us nx and the list of output files
    filename = datafiles[0]
    tree = ET.parse(filename)
    root = tree.getroot()
    list_of_files = []
    depth_axis_values  =[]
    time_axis_values = []
    recursive_elem_counter(root, list_of_files, depth_axis_values)
    nx = len(depth_axis_values)

    # create memmap files - similar to the way we reprocess SGY to 1-second data chunks.
    memmapList = []
    tmpList = []
    tmpdir = os.path.join(dirout,TMP)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
        
    for fname in list_of_files:
        if not fname==DEPTH:
            fileout = os.path.join(tmpdir, fname)
            memmapList.append(numpy.memmap(fileout, dtype=numpy.double, mode='w+', shape=(nx,nt)))
            tmpList.append(fileout)
    for mapped in memmapList:
        mapped.flush()
        
    # loop and write...
    idx = 0
    for fname in datafiles:
        tree = ET.parse(fname)
        root = tree.getroot()
        xidx=0
        recursive_elem_ingest(root, memmapList, idx, xidx, time_axis_values)
        idx = idx + 1

    # create time and depth axis as arrays
    time_axis = numpy.asarray(time_axis_values, dtype=numpy.double)
    depth_axis = numpy.asarray(depth_axis_values, dtype=numpy.double)

    for mapped in memmapList:
        mapped.flush()

    print("max values in memmaps")
    for a in range(len(memmapList)):
        print(numpy.max(memmapList[a]))
        
    print('Converting temporary files to permanent')
    for a in range(len(memmapList)):
        fileout = os.path.join(dirout,list_of_files[a+1]+'.npy')
        print(fileout)
        numpy.save(fileout, numpy.array(memmapList[a]))
        del mapped
        # override lazy behaviour...
        mapped = None
    memmapList = None
    # remove temporary files...
    for filename in tmpList:
        os.remove(filename)
    # Assumption that the fibre depth points do not change over the measurement period
    xaxisfilename = os.path.join(dirout,'measured_depth.npy')
    if not os.path.exists(xaxisfilename):
        numpy.save(xaxisfilename, depth_axis)
    taxisfilename = os.path.join(dirout,'time.npy')
    numpy.save(taxisfilename, time_axis)
'''
####################################################################################################
# Executable code...

#################################
# First make sure that the output
# directory has been created
#################################
if not os.path.exists(outpath):
    os.makedirs(outpath)

#################################
# Process all the files with the
# .las extension
#################################
onlyfiles = [f for f in listdir(inpath) if isfile(join(inpath, f))]
for filename in onlyfiles:
    if filename[-4:]=='.las':
        print('Converting ',filename)
        # we have a las file to convert
        root=las2fbe(inpath+filename)
        writeFBE(outpath,filename,root)
        
'''
def runtest():
    import os
    os.chdir("E:\\2019scr0001\\src2")
    dirout = "E:\\2019scr0001\\interpretation\\vlf"
    dirin = "E:\\2019scr0001\\results\\vlf"
    from witsml_fbe import readFBE
    readFBE(dirin,dirout)
    

if __name__ == "__main__":
    runtest()

