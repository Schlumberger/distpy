from __future__ import division, absolute_import, print_function

__all__ = ['io_help','calc','controllers','ingesters','workers']

__copyright__ = '(C) 2020, Schlumberger. Refer to LICENSE'
__license__ = '../LICENSE'
__status__ = 'Pre-alpha'
# version tracking
# 0.0.0 Baseline version containing the framework for the system.
# 0.1.0 Added the clip command, with a unit-test and documented in the Wiki.
# 0.2.0 Added the diff command, with a unit-test and documented in the Wiki.
# 0.3.0 Added the averageFBE command, with a unit-test and documented in the Wiki.
# 0.3.1 Minor bug fixes to make sure that the RAW-to-seismic integration tests runs
# 0.3.2 Minor bug-fixes to make sure that strainrate to WITSML processing works
# 0.3.3 Bug-fix to ensure directory creation when ingesting SEGY
# 1.0.0 change to the __init__.py format for Azure compatibility
# 1.1.0 post-processing and plotting capabilities added
# 1.2.0 source and reflection profiling
# 1.3.0 basic event detector using peak_to_peak(sta_lta(abs()))
# 1.3.1 attempt to improve accuracy of ibm2ieee conversion
# 1.4.0 velocity maps and masks to enable Velocity Band Energy workflows
# 1.4.1 faster calculation for backscatter-to-strainrate part. Performance tests.
# 1.4.2 - fix so that axes on the thumbnail plot are physical axes
#       - allowed "format" to specified in thumbnail command for jpg option
#       - introduced "real" command to allow the results of velocity masks to be consumed
#       - bug-fix to the velocity mask as the intepretation of the input keys was wrong
#       - sped up the calculation for the sta/lta and the peak-to-peak
#       - introduced wavelet based peak-counting
# 1.5.0 DTS support, in particular the steam-injection workflow starting from
#       CSV data
# 1.5.1 - Support for a scaling inside the thumbnail - this is the number of std devs to bracket the image's colorscale
# 1.5.2 - Bug-fix in the calculation of the diff-filter for poly-pulse processing, to avoid a possible divide-by-zero error.
# 1.6.0 Virtual sources, sweetness attribute and the gauge-length stacking
# 1.6.1 Clipping for data that has not got Time_axis defined
# 1.7.0 destripe filter and auto-docoumentation
# 1.8.0 read h5, read CSV...
# 1.9.0 hash command added for perceptual hash workflows
# 1.10.0 Support for proprietary extensions
# 1.10.1 Extra protection against divide by zero in sgy.py
# 1.11.0 Enabled the basic statistical summaries
# 1.12.0 Enabled the GPU options
# 1.13.0 Added an additional h5 reader option
# 1.14.0 Added the keras command
# 1.14.1 Corrected the auto-generated documentation on the keras command
# 1.15.0 Added the kmeans command
# 1.15.1 Corrected the auto-generated documentation on the kmeans command
__version__ = '1.15.1'

