# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap 
import numpy as np
import os
import sys

if os.path.exists('D:\Visitor\Matt_work\Matt_AWS_Format'):
    direc = 'D:\Visitor\Matt_work\Matt_AWS_Format'
else:
    direc = '/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project'

try:
    from name_list_generator import create_lat_lon_list
except:
    sys.path.insert(0, direc)
    from name_list_generator import create_lat_lon_list

if not os.path.exists(os.path.join('/Users/Matt_Lawhon/Documents/Internships',
                                   'ANN Antarctica Project/name_list',
                                   'station_lat_lon_list.txt')):
    create_lat_lon_list()
    
station_dict = {}

with open(os.path.join('/Users/Matt_Lawhon/Documents/Internships',
                                   'ANN Antarctica Project/name_list',
                                   'station_lat_lon_list.txt')) as f:
    next(f)
    for line in f:
        (station, lat, lon) = line.split()
        station_dict[station] = [float(lat), float(lon)]

an_station_dict = {}
with open('/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project/corr_test.txt', 'r') as f:
    for line in f:
        (station, corr, corr_test, ecmwf_lat, ecmwf_lon) = line.split()

        an_station_dict[station] = station_dict[station]+ [float(corr), float(corr_test), 
                                                           float(ecmwf_lat), float(ecmwf_lon)]
#%%
        
plt.figure(figsize = (15, 15))

m = Basemap(projection='aeqd',llcrnrlon=90. ,llcrnrlat=-84.5,
            urcrnrlon=-154 ,urcrnrlat=-72, epsg = 3412)

cmap_corr = LinearSegmentedColormap.from_list('mycmap', [(.73, 0, .07), (.2, .6, .8)])
#m.etopo(scale=5, alpha=.5)
#m.shadedrelief(scale=5, alpha=.4)
#m.bluemarble(scale=5, alpha=.6);
m.bluemarble(scale=1, alpha=1);
#m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= True)
m.drawmeridians(np.arange(0,360,15),labels=[1,1,1,0], latmax = 90);
m.drawparallels(np.arange(-90,90,5),labels=[1,1,1,1]);
m.drawmapscale(120, -85.5, 175, -85, 100)


(x, y, ex, ey, color_c) = ([], [], [], [], [])
#
#['Alexander(TT!)', 'Elaine', 'Emilia', 'Emma', 'Gill', 'Lettau',
#                     'Margaret', 'Marilyn', 'Sabrina', 'Schwerdtfeger',  'Vito']
for station in an_station_dict:
    x.append(m(an_station_dict[station][1], an_station_dict[station][0])[0])
    y.append(m(an_station_dict[station][1], an_station_dict[station][0])[1])
    ex.append(m(an_station_dict[station][5], an_station_dict[station][4])[0])
    ey.append(m(an_station_dict[station][5], an_station_dict[station][4])[1])
    color_c.append(an_station_dict[station][2])
    if station in ['Alexander(TT!)','Marilyn', 'Sabrina', 'Vito']:
        plt.text(x[-1] - 15000, y[-1] + 25000, station)
    else:
        plt.text(x[-1] - 15000, y[-1] - 15000, station)

im = plt.scatter(x, y, marker = 'o', c = color_c, cmap = cmap_corr,  zorder=5, 
                 s = 150, label = 'AWS station')
plt.scatter(ex, ey, marker = 'o', c = 'white', edgecolor = 'black', s = 40, 
            zorder = 6, label = 'ECMWF approximation location')
plt.colorbar(im, label = 'Correlation (r value)')
plt.title('Accuracy of ECMWF on NN filled in AWS Wind Speed', y = 1.03)
plt.legend(loc = 1)
plt.savefig('/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project/polar map.pdf', 
            format = 'PDF', bbox_inches = 'tight')  

plt.show()
plt.close()
#%%