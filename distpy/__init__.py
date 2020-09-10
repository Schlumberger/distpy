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
# 1.15.2 Corrected setup.py so that all necessary libraries are installed
# 1.16.0 Added the approximate very low frequency (VLF) calculation
# 1.16.1 Corrected the automatic documentation for the analytic signal command
# 1.16.2 corrected the time.npy output file from the CSV ingester.
# 1.17.0 linear transform and addition included to make custom filters easier to design
# 1.18.0 Generalized the kmeans clustering
# 1.19.0 CSV ingestion to WITSML
# 1.20.0 CSV ingestion from DTS-style (depth as rows, time as columns)
# 1.21.0 Improvement in data-handling of difference cases so that they can be plotted
# 1.22.0 Improvement in self documenting features
# 1.23.0 GPU write_numpy to make hybrid cloud easier as a deployment
# 1.23.1 Fix a bug in write_numpy
# 1.24.0 Allowed an external data-set to be used for min() and max() in the rescale command
# 1.24.1 Improved behaviour when an unknown command is tried
# 1.25.0 Added the extract command so that a single line can be taken
# 1.25.1 Reshape extract results to fit the pattern (n,1) of 2D arrays used by distpy
# 1.26.0 Select cluster using bounded_select, returns a mask
# 1.27.0 Peak broadening command
# 1.28.0 dip_filter, makes use of the existing convolve 2D filtering
# 1.29.0 Generic ingestion, and the in-memory option to support the CASE01 example
# 1.30.0 Generic templating via io_help.io_helpers.template(templateJSON={})
# 1.30.1 Fixed a bug in the parallel version introduced in 1.29.0
__version__ = '1.30.1'


