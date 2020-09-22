
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 20:27:06 2018

@author: Matt_Lawhon
"""

def get_ecmwf(station):
    import pandas as pd
    import calendar
    from netCDF4 import Dataset
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
        
    # Method to add Julian Day to the ECMWF data set
    def atmospheric_julian_day(year, month, day):
        if (calendar.isleap(year)):
            month_list = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        else:
            month_list = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        return month_list[month-1] + day
    atmospheric_jd = np.vectorize(atmospheric_julian_day)
    
    # Method to find the avg Lat/Lon of AWS stations for calculating the approx ECMWF
    # Model of the station
    
    if not os.path.exists(os.path.join('/Users/Matt_Lawhon/Documents/Internships',
                                       'ANN Antarctica Project/name_list',
                                       'station_lat_lon_list.txt')):
        create_lat_lon_list()
    dataset = Dataset(os.path.join("/Users/Matt_Lawhon/Documents/Internships",
                                   "ANN Antarctica Project/Antarctica.nc"))
    station_dict = {}
    with open(os.path.join('/Users/Matt_Lawhon/Documents/Internships',
                                       'ANN Antarctica Project/name_list',
                                       'station_lat_lon_list.txt')) as f:
        next(f)
        for line in f:
            (key, lat, lon) = line.split()
            station_dict[key] = [float(lat), float(lon)]
    
    # Change Based on Station!
    if not (station in station_dict):
        print('AWS Stations on server:', list(station_dict.keys()))
        raise FileNotFoundError("The Station '%s' does not exist in the online AWS directory" % station)
    
    
    ML = True
    if (ML and station_dict[station][0] > -87.5):
        
        #import the surrounding 500 pressure level dataset
        surrounding = Dataset(os.path.join("/Users/Matt_Lawhon/Documents/Internships",
                                   "ANN Antarctica Project/Antarctica_PL.nc"))
        lat = surrounding.variables['latitude'][:]
        lon = surrounding.variables['longitude'][:]
        time = surrounding.variables['time'][:]
        z = surrounding.variables['z']
        t = surrounding.variables['t']
        units = {z : [0, 1], t : [-273.15, 1]}
        
        # Create point to the north by 2.5 degrees to the east by 2.5 degrees ...
        if (station_dict[station][0] > -87.5):
            
             #wont work for all boundary cases
            lon_pos_del = station_dict[station][1] + 2.5 if station_dict[station][1] + 2.5 <= 180 \
                                                         else station_dict[station][1] + 2.5 - 360
            lon_neg_del = station_dict[station][1] - 2.5 if station_dict[station][1] - 2.5 > -180 \
                                                         else station_dict[station][1] - 2.5 + 360
            
            station_lat_indexN = np.where((lat < station_dict[station][0] + 2.5 + .375) & 
                                          (lat > station_dict[station][0] + 2.5 - .375))[0]
            station_lon_indexN = np.where((lon < station_dict[station][1] + .375) & 
                                          (lon > station_dict[station][1] - .375))[0]
            station_lat_indexE = np.where((lat < station_dict[station][0] + .375) & 
                                          (lat > station_dict[station][0] - .375))[0]
            station_lon_indexE = np.where((lon < lon_pos_del + .375) & 
                                          (lon > lon_pos_del - .375))[0]
            station_lat_indexS = np.where((lat < station_dict[station][0] - 2.5 + .375) & 
                                          (lat > station_dict[station][0] - 2.5 - .375))[0]
            station_lon_indexS = np.where((lon < station_dict[station][1] + .375) & 
                                          (lon > station_dict[station][1] - .375))[0]
            station_lat_indexW = np.where((lat < station_dict[station][0] + .375) & 
                                          (lat > station_dict[station][0] - .375))[0]
            station_lon_indexW = np.where((lon < lon_neg_del  + .375) & 
                                          (lon > lon_neg_del - .375))[0]
        
        # If the point is near the south pole, just take points 2.5 north (in diff. directions)
        else: 
            station_lat_indexN = -87.75
            station_lon_indexN  = -180
            station_lat_indexE = -87.75
            station_lon_indexE  = -90
            station_lat_indexS = -87.75
            station_lon_indexS  = 0
            station_lat_indexW = -87.75
            station_lon_indexW  = 90
            
        # Create variables with whole time frame at particular point
        tN = np.ma.compressed((t[:, station_lat_indexN, station_lon_indexN].
                               mean(axis = (1, 2)))*units[t][1] + units[t][0])
        zN = np.ma.compressed((z[:, station_lat_indexN, station_lon_indexN].
                               mean(axis = (1, 2)))*units[z][1] + units[z][0])
        tS = np.ma.compressed((t[:, station_lat_indexS, station_lon_indexS].
                               mean(axis = (1, 2)))*units[t][1] + units[t][0]) 
        zS = np.ma.compressed((z[:, station_lat_indexS, station_lon_indexS].
                               mean(axis = (1, 2)))*units[z][1] + units[z][0])
        tE = np.ma.compressed((t[:, station_lat_indexE, station_lon_indexE].
                               mean(axis = (1, 2)))*units[t][1] + units[t][0]) 
        zE = np.ma.compressed((z[:, station_lat_indexE, station_lon_indexE].
                               mean(axis = (1, 2)))*units[z][1] + units[z][0])
        tW = np.ma.compressed((t[:, station_lat_indexW, station_lon_indexW].
                               mean(axis = (1, 2)))*units[t][1] + units[t][0]) 
        zW = np.ma.compressed((z[:, station_lat_indexW, station_lon_indexW].
                               mean(axis = (1, 2)))*units[z][1] + units[z][0])
        
        #creates date-time index
        date_time = pd.date_range(start="1984-01-01", periods = len(time), freq = '6H')
    
        # Create Dataframe, specify index/order
        #
        # 'Avg ' is to remind the viewer that ECMWF measurements are more 
        # representative of a six hour average than a single measurement
        # somehow valid for a six hour period
        surrounding_df = pd.DataFrame({'Avg Temperature N (C)': tN,
                                       'Avg Temperature E (C)': tE,
                                       'Avg Temperature S (C)': tS,
                                       'Avg Temperature W (C)': tW,
                                       'Avg Geopotential N (m^2/s^2)': zN,
                                       'Avg Geopotential E (m^2/s^2)': zE,
                                       'Avg Geopotential S (m^2/s^2)': zS,
                                       'Avg Geopotential W (m^2/s^2)': zW,
                                       'Latitude N': lat[station_lat_indexN].mean(), 
                                       'Longitude N': lon[station_lon_indexN].mean(),
                                       'Latitude E': lat[station_lat_indexE].mean(), 
                                       'Longitude E': lon[station_lon_indexE].mean(),
                                       'Latitude S': lat[station_lat_indexS].mean(), 
                                       'Longitude S': lon[station_lon_indexS].mean(),
                                       'Latitude W': lat[station_lat_indexW].mean(), 
                                       'Longitude W': lon[station_lon_indexW].mean()}, 
                                       columns = ['Avg Temperature N (C)',
                                                 'Avg Geopotential N (m^2/s^2)', 
                                                 'Latitude N', 
                                                 'Longitude N', 
                                                 'Avg Temperature E (C)', 
                                                 'Avg Geopotential E (m^2/s^2)', 
                                                 'Latitude E', 
                                                 'Longitude E', 
                                                 'Avg Temperature S (C)', 
                                                 'Avg Geopotential S (m^2/s^2)', 
                                                 'Latitude S', 
                                                 'Longitude S', 
                                                 'Avg Temperature W (C)', 
                                                 'Avg Geopotential W (m^2/s^2)', 
                                                 'Latitude W', 
                                                 'Longitude W'], index = date_time)
        #specify formatting 
        surrounding_df = surrounding_df.round(2)
        surrounding_df.replace(-0.0, 0.0, inplace = True)
    
    # Define main ECMWF dataset variables
    lat = dataset.variables['latitude'][:]
    lon = dataset.variables['longitude'][:]
    time = dataset.variables['time'][:]
    msl = dataset.variables['msl']
    u10 = dataset.variables['u10']
    v10 = dataset.variables['v10']
    t2m = dataset.variables['t2m']
    #d2m = dataset.variables['d2m'] 
    # http://andrew.rsmas.miami.edu/bmcnoldy/Humidity.html (used to calculate rel hum)
    
    # Set additive and multiplicative coefficients for unit conversion
    units = {msl : [0, .01], u10 : [0, 1], v10 : [0, 1], t2m : [-273.15, 1]}#, d2m : [-273.15, 1]}
    
    # Find closest ECMWF data point
    station_lat_index = np.where((lat < station_dict[station][0] + .375) & 
                                 (lat > station_dict[station][0] - .375))[0]
    station_lon_index = np.where((lon < station_dict[station][1] + .375) & 
                                 (lon > station_dict[station][1] - .375))[0]
    
    # Calculate the variables value at said ECMWF close point
    msl = np.ma.compressed((msl[:, station_lat_index, station_lon_index].
                            mean(axis = (1, 2)))*units[msl][1] + units[msl][0])
    u10 = np.ma.compressed((u10[:, station_lat_index, station_lon_index].
                            mean(axis = (1, 2)))*units[u10][1] + units[u10][0])
    v10 = np.ma.compressed((v10[:, station_lat_index, station_lon_index].
                            mean(axis = (1, 2)))*units[v10][1] + units[v10][0])
    t2m = np.ma.compressed((t2m[:, station_lat_index, station_lon_index].
                            mean(axis = (1, 2)))*units[t2m][1] + units[t2m][0])
    #d2m = np.ma.compressed((d2m[:, station_lat_index, station_lon_index].
    #                       mean(axis = (1, 2)))*units[d2m][1] + units[d2m][0])
    
    # Convert from U and V wind speed to wind speed and direction [0, 360) and datetime index
    wind_speed = np.sqrt(u10**2 + v10**2)
    wind_direction = np.degrees(np.arctan2(u10, v10)) + 180
    date_time = pd.date_range(start="1984-01-01", periods = len(time), freq = '6H')
    
    # Creates full dataframe
    #
    # 'Avg ' is to remind the viewer that ECMWF measurements are more 
    # representative of a six hour average than a single measurement
    # somehow valid for a six hour period
    full_df = pd.DataFrame({'Year': date_time.year, 'Month': date_time.month,
                               'Julian Day': atmospheric_jd(date_time.year,
                                                            date_time.month,
                                                            date_time.day),
                               'Avg Temperature (C)': t2m,
                               'Avg Wind Speed (m/s)': wind_speed, 
                               'Avg Wind Direction': wind_direction,
                               'Avg U Wind Speed (m/s)': u10, 
                               'Avg V Wind Speed (m/s)': v10,
                               'Avg Pressure (hPa)': msl,
                               'AWS Latitude': station_dict[station][0], 
                               'AWS Longitude': station_dict[station][1],
                               'Latitude': lat[station_lat_index].mean(),
                               'Longitude': lon[station_lon_index].mean()}, 
                                columns = ['Year', 'Month', 'Julian Day',
                                           'Avg Temperature (C)', 'Avg Wind Speed (m/s)',
                                           'Avg Wind Direction', 'Avg U Wind Speed (m/s)', 
                                           'Avg V Wind Speed (m/s)','Avg Pressure (hPa)',
                                           'AWS Latitude', 'AWS Longitude', 'Latitude',
                                           'Longitude'], index = date_time)
    
    # Formatting specifications
    full_df = full_df.round(2)
    full_df.replace(-0.0, 0.0, inplace = True)
    
    if not os.path.exists(os.path.join('/Users/Matt_Lawhon/Documents/Internships', 
                                       'ANN Antarctica Project', 'ECMWF', station)):
        os.makedirs(os.path.join('/Users/Matt_Lawhon/Documents/Internships', 
                                       'ANN Antarctica Project', 'ECMWF', station))
    
    # Naming based on whether the dataset contains the surrounding climate variables.
    if(ML):
        full_df = pd.merge(full_df, surrounding_df, left_index = True, right_index = True)
        full_df.to_csv(os.path.join('/Users/Matt_Lawhon/Documents/Internships', 
                                    'ANN Antarctica Project', 'ECMWF', station,
                                    '%s_6-hour_ECMWF_ML_Reformat.txt' % station), 
                        sep = '\t', index = True)
    else:
        full_df.to_csv(os.path.join('/Users/Matt_Lawhon/Documents/Internships', 
                                    'ANN Antarctica Project', 'ECMWF', station,
                                    '%s_6-hour_ECMWF_Reformat.txt' % station), 
                        sep = '\t', index = True)
        
    # Dict for changing the time interval of measurements in the dataframe
    time_conversion = {'daily': ['1-day', [full_df['Year'], full_df['Month'],
                                           full_df['Julian Day']], 'D'], 
                       'monthly': ['1-month', [full_df['Year'], full_df['Month']], 'MS']}
    
    for time_interval in ['daily', 'monthly']:
        
        # Creating min/max/mean... dataframes and datetime for index of the combined
        constants = full_df.groupby(time_conversion[time_interval][1])['Year',
                                   'Month', 'Julian Day', 'AWS Latitude', 'AWS Longitude',
                                   'Latitude', 'Longitude'].first()
        means = full_df.groupby(time_conversion[time_interval][1])['Avg Temperature (C)',
                               'Avg Wind Speed (m/s)', 'Avg U Wind Speed (m/s)', 
                               'Avg V Wind Speed (m/s)', 'Avg Pressure (hPa)'].mean()
        means['Avg Wind Direction'] = np.degrees(np.arctan2(means['Avg U Wind Speed (m/s)'],
                                                 means['Avg V Wind Speed (m/s)'])) + 180
        maximums = full_df.groupby(time_conversion[time_interval][1])['Avg Temperature (C)',
                                  'Avg Wind Speed (m/s)', 'Avg U Wind Speed (m/s)', 
                                  'Avg V Wind Speed (m/s)', 'Avg Pressure (hPa)'].max()
        minimums = full_df.groupby(time_conversion[time_interval][1])['Avg Temperature (C)',
                                  'Avg Wind Speed (m/s)', 'Avg U Wind Speed (m/s)', 
                                  'Avg V Wind Speed (m/s)', 'Avg Pressure (hPa)'].min()
        date_time = pd.date_range(start="1984-01-01", periods = len(means), 
                                  freq = time_conversion[time_interval][2])
        
        # Rename columns
        minimums.columns = minimums.columns.str.replace('Avg', 'Min')
        maximums.columns = maximums.columns.str.replace('Avg', 'Max')
        
        # Combine Dataframes, set datetime column as index, format and save to txt file
        df = pd.concat([constants, means, maximums, minimums], axis = 1)
        df.set_index(date_time, inplace = True)
        df = df[['Year', 'Month', 'Julian Day', 'Min Temperature (C)', #change month
                 'Avg Temperature (C)', 'Max Temperature (C)',
                 'Min Wind Speed (m/s)', 'Avg Wind Speed (m/s)',
                 'Max Wind Speed (m/s)', 'Avg Wind Direction',
                 'Min U Wind Speed (m/s)', 'Avg U Wind Speed (m/s)',
                 'Max U Wind Speed (m/s)', 'Min V Wind Speed (m/s)', 
                 'Avg V Wind Speed (m/s)', 'Max V Wind Speed (m/s)', 
                 'Min Pressure (hPa)', 'Avg Pressure (hPa)',
                 'Max Pressure (hPa)', 'AWS Latitude', 'AWS Longitude',
                 'Latitude', 'Longitude']]
        df = df.round(2)
        df.to_csv(os.path.join('/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project',
                               'ECMWF', station,
                               '%s_%s_ECMWF_dataset.txt' % (station, time_conversion[time_interval][0])), 
                               sep = '\t', index = True, chunksize = 1000)
    

