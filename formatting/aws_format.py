#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 06:54:10 2018

@author: Matt_Lawhon
"""

def get_aws(station):
    
    import urllib.request 
    from urllib.error import URLError
    import pandas as pd
    import numpy as np
    import os
    import sys
    
    # sets directory based on if this file is being run on lamont desktop or matt's laptop 
    if os.path.exists('D:\Visitor\Matt_work\Matt_AWS_Format'):
        direc = 'D:\Visitor\Matt_work\Matt_AWS_Format'
    else:
        direc = '/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project'
     
    # Import name_list file (name_list_generator has to be run first)
    if not os.path.exists(os.path.join(direc, 'name_list', 'full_name_list.txt')):
        try:
            from name_list_generator import create_name_list
        except:
            sys.path.insert(0, direc)
            from name_list_generator import create_name_list
        finally:
            create_name_list()
    
        
    name_list = pd.read_csv(os.path.join(direc, 'name_list', 'full_name_list.txt'),
                            sep = '\t', index_col = 0)
    
    
    # Sets up dataframe which stations data will be added to
    station_df = name_list[name_list['Station'] == station]
    
    # Returns error if an invalid station was passed
    if (len(station_df) == 0):
        print('AWS Stations on server:', name_list.groupby('Station').first().index.values.tolist())
        raise FileNotFoundError("The Station '%s' does not exist in the online AWS directory" % station)
        
    df = pd.DataFrame(columns = ['Year', 'Month', 'Day', 'Julian Day', 'Time',
                     'Temperature (C)', 'Wind Speed (m/s)', 'Wind Direction', 
                     'U Wind Speed (m/s)', 'V Wind Speed (m/s)', 'Pressure (hPa)',
                     'Relative Humidity (%)','Delta-T (C)', 'Latitude', 
                     'Longitude', 'Elevation (m)'])
    column_order = df.columns.tolist()
    
    
    # Prepares directories and file paths for later file saving
    directory = os.path.join(direc, 'AWS', station)
    directory_raw_files = os.path.join(directory, 'raw monthly files')
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory_raw_files):
        os.makedirs(directory_raw_files)
    
    # i.e. for every file for the set station on the AWS server
    for index, row in station_df.iterrows():
        file_name =  os.path.join(directory_raw_files, '%s_%s-%s.txt' % (station, 
                                 int(row['Month']), row['Year']))
    
        # Downloads file if not already downloaded
        try:
            if not os.path.exists(file_name):
                urllib.request.urlretrieve('ftp://amrc.ssec.wisc.edu/pub/aws/10min/rdr/%s/%s' % 
                                           (row['Year'], row['File']), file_name)
            
            # Read in file and format nans and other values
            Temp = pd.read_csv(file_name, names = ['Julian Day','Time','Temperature (C)',
                              'Pressure (hPa)','Wind Speed (m/s)', 'Wind Direction',
                              'Relative Humidity (%)', 'Delta-T (C)'], delim_whitespace = True,
                               skiprows = 2).applymap(lambda x: np.nan if x == 444 else x)
            Temp['Year'] = int(row['Year'])
            Temp['Month'] = int(row['Month'])
            Temp['U Wind Speed (m/s)'] = np.product([Temp['Wind Speed (m/s)'], 
                                np.sin(np.radians(Temp['Wind Direction'])), -1])
            Temp['V Wind Speed (m/s)'] = np.product([Temp['Wind Speed (m/s)'], 
                                np.cos(np.radians(Temp['Wind Direction'])), -1])
            Temp['Julian Day'] = Temp['Julian Day'].astype(int)
            Temp['Day'] = (Temp['Julian Day'] - Temp['Julian Day'].iloc[0] + 1).astype(int)
            Temp['Time'] = Temp['Time'].astype(int)
            
            
            # Add in the Lat, Lon, and Elev of the station
            with open(file_name) as f:
                head = [(next(f)).split() for x in range(2)]
            Temp['Latitude'] = -1 * float((head[1][2])[:-1]) if \
                                (head[1][2])[-1:].upper() == 'S' else \
                                float((head[1][2])[:-1])
            Temp['Longitude'] = float((head[1][5])[:-1]) if \
                                (head[1][5])[-1:].upper() == 'E' else \
                                -1 * float((head[1][5])[:-1])
            Temp['Elevation (m)'] = float(head[1][8])
            
            # Final format and add to cumulative Dataframe
            Temp = Temp[column_order]
            Temp = Temp.round(2)
            df = df.append(Temp, ignore_index = True, sort = True)[column_order]
        except URLError:
            pass
        
    # Saves the raw data
    clean = False
    if not clean:
        df.replace(-0.0, 0.0, inplace = True)
        df.to_csv(os.path.join(directory, '%s_10-min_AWS_dataset.txt' % station),
                  sep = '\t', index = True, chunksize = 1000)
        clean = True
        
    # Removes wind values where the direction or speed stick to the same value for 
    # more than an hour since those are likely null values
    if clean:
        high_wind = pd.Series(df['Wind Speed (m/s)'].dropna()).value_counts()
        high_wind = high_wind[high_wind>100].index.values
        for elt in high_wind[high_wind - 15 > np.mean(high_wind)]:
            df[['Wind Speed (m/s)','U Wind Speed (m/s)', 'V Wind Speed (m/s)']] = \
            df.where(df['Wind Speed (m/s)'] != elt, np.nan)[['Wind Speed (m/s)', 'U Wind Speed (m/s)', 'V Wind Speed (m/s)']]
            
        df['Wind Direction'] = df.where(df['Wind Direction'] != 0, np.nan)['Wind Direction']
        (df['Temperature Validity'], df['Wind Validity']) = (False, False)

        for i in (range(6)):
            
            df['Wind count nan'] = np.nansum([df.isna()['Wind Direction'].shift(i) for i in range(i-5, i+1)], axis = 0)
            df['Wind Validity'] = (((np.nansum(np.equal([df['Wind Direction'] for i in range(6)],
                                                 [df['Wind Direction'].shift(i) for i in range(i-5, i+1)]), 
                                                axis = 0) + df['Wind count nan'] == 6) & 
                                                (df['Wind count nan'] < 4)) | df['Wind Validity'])
            
            df['Wind count nan'] = np.nansum([df.isna()['Wind Speed (m/s)'].shift(i) for i in range(i-5, i+1)], axis = 0)
            df['Wind Validity'] = (((np.nansum(np.equal([df['Wind Speed (m/s)'] for i in range(6)],
                                                 [df['Wind Speed (m/s)'].shift(i) for i in range(i-5, i+1)]), 
                                                axis = 0) + df['Wind count nan'] == 6) & 
                                                (df['Wind count nan'] < 2)) | df['Wind Validity'])
            
            df['Temperature Validity'] = ((np.nansum(np.equal([df['Temperature (C)'] for i in range(6)],
                                                 [df['Temperature (C)'].shift(i) for i in range(i-5, i+1)]), 
                                                axis = 0) == 6) | df['Temperature Validity']) 
            
            
        df[['Wind Direction', 'Wind Speed (m/s)', 'U Wind Speed (m/s)', 'V Wind Speed (m/s)']] = \
        df.where(~df['Wind Validity'], np.nan)[['Wind Direction','Wind Speed (m/s)', 'U Wind Speed (m/s)', 'V Wind Speed (m/s)']]
        
        df['Temperature (C)'] = df.where(~df['Temperature Validity'], np.nan)['Temperature (C)']
        df.drop(columns = ['Wind Validity', 'Wind count nan', 'Temperature Validity'], inplace = True)
        
        df.to_csv(os.path.join(directory, '%s_10-min_clean_AWS_dataset.txt' % station),
                  sep = '\t', index = True, chunksize = 1000)
    
    
    # For the cleaned cumulative file and the original cumulative file
    for clean in [False, True]:
        
        if not clean:
            full_df =  pd.read_csv(os.path.join(directory, '%s_10-min_AWS_dataset.txt' % station),
                                   sep = '\t', index_col = 0)
        if clean: 
            full_df =  pd.read_csv(os.path.join(directory, '%s_10-min_clean_AWS_dataset.txt' % station),
                                   sep = '\t', index_col = 0)
   
        # Dict with values to aid in groupby function
        time_conversion =  {'hourly': ['1-hour', full_df.index - full_df.index%6],
                            'six hourly': ['6-hour', full_df.index - full_df.index%36],
                            'daily': ['1-day', [full_df['Year'], full_df['Month'],
                                                full_df['Julian Day']]],
                            'monthly': ['1-month', [full_df['Year'], full_df['Month']]]}
                           
        # Set the time interval over which the dataframe will be calculated
        for time_interval in ['hourly', 'six hourly', 'daily', 'monthly']:
            
            # Dataframes containing the 'name' of the cumulative file
            constants = full_df.groupby(time_conversion[time_interval][1])['Year',
                                       'Month', 'Day', 'Julian Day', 'Latitude',
                                       'Longitude', 'Elevation (m)'].first()
            means = full_df.groupby(time_conversion[time_interval][1])[ 'Temperature (C)',
                                   'Wind Speed (m/s)', 'U Wind Speed (m/s)',
                                   'V Wind Speed (m/s)', 'Pressure (hPa)', 
                                   'Relative Humidity (%)', 'Delta-T (C)'].mean()
            means['Wind Direction'] = np.degrees(np.arctan2(means['U Wind Speed (m/s)'],
                                             means['V Wind Speed (m/s)'])) + 180
            maximums = full_df.groupby(time_conversion[time_interval][1])['Temperature (C)',
                                      'Wind Speed (m/s)', 'U Wind Speed (m/s)',
                                      'V Wind Speed (m/s)', 'Pressure (hPa)', 
                                      'Relative Humidity (%)', 'Delta-T (C)'].max()
            minimums = full_df.groupby(time_conversion[time_interval][1])['Temperature (C)',
                                      'Wind Speed (m/s)', 'U Wind Speed (m/s)',
                                      'V Wind Speed (m/s)', 'Pressure (hPa)', 
                                      'Relative Humidity (%)', 'Delta-T (C)'].min()
            missing = full_df[['Temperature (C)', 'Wind Speed (m/s)', 'Wind Direction',
                               'U Wind Speed (m/s)','V Wind Speed (m/s)',
                               'Pressure (hPa)', 'Relative Humidity (%)',
                               'Delta-T (C)']].isna().groupby(
                               time_conversion[time_interval][1]).sum()
            
            # Calculate date_time interval which will serve as the dataframe index
            if (time_interval == 'hourly' or time_interval == 'six hourly'):
                date_time = pd.to_datetime({'year': constants['Year'], 
                                            'month': constants['Month'], 
                                            'day': constants['Day'], 
                                            'hour': (constants.index%144)/6})
            else:
                date_time = pd.to_datetime({'year': constants['Year'], 
                                            'month': constants['Month'],
                                            'day': constants['Day']})
        
            # Rename the columns to reflect their quality
            means = means.add_prefix('Avg ')
            maximums = maximums.add_prefix('Max ')
            minimums = minimums.add_prefix('Min ')
            missing = missing.add_prefix('Number of Missing ')
            missing = missing.add_suffix(' Measurements')
            
            # Combine the constants, means, maximums, minimums and  missing
            df = pd.concat([constants, means, maximums, minimums, missing], axis = 1)
            
            # Specify column order
            df = df[['Year', 'Month', 'Day', 'Julian Day', 'Min Temperature (C)',
                     'Avg Temperature (C)', 'Max Temperature (C)',
                     'Number of Missing Temperature (C) Measurements',
                     'Min Wind Speed (m/s)', 'Avg Wind Speed (m/s)', 
                     'Max Wind Speed (m/s)', 
                     'Number of Missing Wind Speed (m/s) Measurements',
                     'Avg Wind Direction', 
                     'Number of Missing Wind Direction Measurements',
                     'Min U Wind Speed (m/s)',
                     'Avg U Wind Speed (m/s)', 'Max U Wind Speed (m/s)',
                     'Number of Missing U Wind Speed (m/s) Measurements',
                     'Min V Wind Speed (m/s)', 'Avg V Wind Speed (m/s)',
                     'Max V Wind Speed (m/s)', 
                     'Number of Missing V Wind Speed (m/s) Measurements',
                     'Min Pressure (hPa)', 'Avg Pressure (hPa)',
                     'Max Pressure (hPa)', 
                     'Number of Missing Pressure (hPa) Measurements',
                     'Min Relative Humidity (%)', 'Avg Relative Humidity (%)', 
                     'Max Relative Humidity (%)', 
                     'Number of Missing Relative Humidity (%) Measurements',
                     'Min Delta-T (C)', 'Avg Delta-T (C)', 'Max Delta-T (C)',
                     'Number of Missing Delta-T (C) Measurements', 'Latitude',
                     'Longitude', 'Elevation (m)']]
            
            # Formating 
            df.replace(-0.0, 0.0, inplace = True)
            df = df.round(2)
            df[df.columns[[7, 11, 13, 17, 21, 25, 29, 33]]] = df[df.columns[[7,
              11, 13, 17, 21, 25, 29, 33]]].astype(int)
            df.reset_index(drop = True, inplace = True)
            df.set_index(date_time, inplace = True)
            
            # Naming according to whether the data was taken from the cleaned cumulative file or not
            if not clean:
                df.to_csv(os.path.join(directory, '%s_%s_AWS_dataset.txt' % 
                                       (station, time_conversion[time_interval][0])), 
                                        sep = '\t', index = True, chunksize = 1000)
            else:
                df.to_csv(os.path.join(directory, '%s_%s_clean_AWS_dataset.txt' % 
                                       (station, time_conversion[time_interval][0])), 
                                        sep = '\t', index = True, chunksize = 1000)
              