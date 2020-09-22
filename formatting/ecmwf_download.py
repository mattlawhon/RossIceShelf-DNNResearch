#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 00:14:25 2018

@author: Matt_Lawhon
"""

# Download function for the main ECMWF .nc file. 

# THIS HAS NOT BEEN SET UP FOR THE LAMONT PC COMPUTER
# To set this up, you will need to create a new environment in conda
# Follow instructions here: https://confluence.ecmwf.int/display/WEBAPI/Accessing+ECMWF+data+servers+in+batch
# and here: https://anaconda.org/conda-forge/ecmwf-api-client
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
    "area": "-75/-180/-90/179.25",
    "class": "ei",
    "dataset": "interim",
    "date": "1984-01-01/to/2018-04-30", # update date in the future
    "expver": "1",
    "grid": "0.75/0.75",
    "levtype": "sfc",
    "param": "151.128/165.128/166.128/167.128",
    "step": "0",
    "stream": "oper",
    "time": "00:00:00/06:00:00/12:00:00/18:00:00",
    'type': 'an',
    'resol': 'av',
    'format': "netcdf",
    'target': "/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project/Antarctica.nc",
})

#%%

# Download function for the 500 pressure level .nc file

# THIS HAS NOT BEEN SET UP FOR THE LAMONT PC COMPUTER
# To set this up, you will need to create a new environment in conda
# Follow instructions here: https://confluence.ecmwf.int/display/WEBAPI/Accessing+ECMWF+data+servers+in+batch
# and here: https://anaconda.org/conda-forge/ecmwf-api-client
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
    "area": "-75/-180/-90/179.25",
    "class": "ei",
    "dataset": "interim",
    "date": "1984-01-01/to/2018-04-30", # update date in the future
    "expver": "1",
    "grid": "0.75/0.75",
    "levelist": "500",
    "levtype": "pl",
    "param": "129.128/130.128",
    "step": "0",
    "stream": "oper",
    "time": "00:00:00/06:00:00/12:00:00/18:00:00",
    "type": "an",
    'format': "netcdf",
    "target": "/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project/Antarctica_PL.nc",
})