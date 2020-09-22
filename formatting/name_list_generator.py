# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:28:07 2018

@author: Indrani
"""


def create_name_list():
    import datetime
    import urllib.request
    import pandas as pd
    import os
    
    # Set directory
    if (os.path.exists('/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project')):
        directory = '/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project'
    else:
        directory = 'D:\Visitor\Matt_work\Matt_AWS_Format'
    
    # Convert a lat/lon string '76.56S' --> -76.56
    def lat_lon_conv(s, lat = False, lon = False):
        if s == '':
            return 0
        elif lat:
            return float(s[:-1]) if s[-1:].upper() == 'N' else -1*float(s[:-1])
        elif lon:
            return float(s[:-1]) if s[-1:].upper() == 'E' else -1*float(s[:-1])
            
    # Create the dataframe to store data   
    name_list = pd.DataFrame(columns = ['Station', 'Year', 'Month', 'File'])
    for year in range(1984, datetime.datetime.now().year + 1):
        
        # Downloads namelist file for the specified year if its not already downloaded.
        if not os.path.exists(os.path.join(directory, 'name_list', 'namelist%s.txt' % (str(year)[2:]))):
            urllib.request.urlretrieve('ftp://amrc.ssec.wisc.edu/pub/aws/10min/rdr/%s/namelist%s' 
                                       % (year, str(year)[2:]), os.path.join(directory, 
                                         'name_list', 'namelist%s.txt' % (str(year)[2:])))
    
        with open(os.path.join(directory, 'name_list', 'namelist%s.txt' % (str(year)[2:]))) as f:
            data = f.readlines()
            key_values = [[], [], [], [], [], []] # station, month, file_name, latitude, longitude, elevation
    
        for counter, line in enumerate(data):
            if (len((data[counter]).split())):
                if len(line.split()[1:-10]) > 1:
                    data[counter] = ' '.join([line.split()[0], ''.join(line.split()[1:-10])]+line.split()[-10:]) + '\n'
    
                key_values[0].append((data[counter]).split()[1:2][0])
                key_values[1].append(((data[counter]).split()[:1][0])[-6:-4])
                key_values[2].append((data[counter]).split()[:1][0])
                
                lat = (data[counter]).split('Lat :')[1].split('Long')[0].strip()
                lon = (data[counter]).split('Long :')[1].split('Elev')[0].strip()
                
                key_values[3].append(lat_lon_conv(lat, lat = True))
                key_values[4].append(lat_lon_conv(lon, lon = True))
                
                key_values[5].append((data[counter]).split()[-2:-1][0])
                
        with open(os.path.join(directory, 'name_list', 'namelist%s.txt' % (str(year)[2:])), 'w') as f:
            f.writelines(data)
    
        name_list = name_list.append(pd.DataFrame({'Station': key_values[0], 'Year': year,
                                                   'Month': key_values[1], 'File': key_values[2],
                                                   'Latitude': key_values[3], 'Longitude': key_values[4], 
                                                   'Elevation (m)': key_values[5]}),
                                     ignore_index = True, sort = False)
    
    name_list['Month'] = name_list['Month'].apply(lambda x: 1 if x == 'x1' else x).astype(float)
    name_list.sort_values(['Station', 'Year', 'Month'], inplace = True)
    name_list.reset_index(drop = True, inplace = True)
    name_list.to_csv(os.path.join(directory, 'name_list', 'full_name_list.txt'), sep = '\t', index = True)

def create_lat_lon_list():
    import numpy as np
    import pandas as pd
    import os
    
    if (os.path.exists('/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project')):
        directory = '/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project'
    else:
        directory = 'D:\Visitor\Matt_work\Matt_AWS_Format'
    
    if not os.path.exists(os.path.join(directory, 'name_list', 'full_name_list.txt')):
        create_name_list()
    
    lat_lon = pd.read_csv(os.path.join(directory, 'name_list', 'full_name_list.txt'), sep = '\t', index_col = 0)
    lat_lon['X'] = np.sin(np.radians(lat_lon['Latitude'] + 90)) * np.cos(np.radians(lat_lon['Longitude']+180))
    lat_lon['Y'] = np.sin(np.radians(lat_lon['Latitude'] + 90)) * np.sin(np.radians(lat_lon['Longitude']+180))
    lat_lon['Z'] = np.cos(np.radians(lat_lon['Latitude'] + 90))
    
    lat_lon_list = lat_lon.groupby('Station')['X', 'Y', 'Z'].mean()
    lat_lon_list['Latitude'] = np.degrees(np.arccos(lat_lon_list['Z']/np.sqrt(lat_lon_list['Y']**2 
                                                    + lat_lon_list['X']**2+ lat_lon_list['Z']**2))) - 90
    lat_lon_list['Longitude'] = (np.degrees(np.arctan2(lat_lon_list['Y'],
                                lat_lon_list['X'])) - 180).apply(lambda x: x if x > -180 else x + 360)
    lat_lon_list.drop(columns = ['X', 'Y', 'Z'], inplace = True)
    lat_lon_list = lat_lon_list.round(3)
    lat_lon_list.to_csv(os.path.join(directory, 'name_list', 'station_lat_lon_list.txt'), sep = '\t', index = True)


