'''
Created May 2017

@author: Matthew Lawhon
'''

import pandas as pd
import os
import numpy as np
    
def normalize(df, cutoff_year = 0):
    df['Time'] = (df.index%4)/4
    normalize = df[['Year',  'Month', 'Julian Day', 'AWS Avg Temperature (C)', 'AWS Avg Pressure (hPa)', 'Time']]
    if (cutoff_year):
        norm_cutoff_year = (2014 - df['Year'].min())/(df['Year'].max() - df['Year'].min())
    normalize = (normalize-normalize.min())/(normalize.max()-normalize.min())
    normalize['AWS Avg Wind Speed (m/s)']  = df['AWS Avg Wind Speed (m/s)']
    
    if (cutoff_year):
        return normalize[normalize['Year'] > norm_cutoff_year] 
    else:
        return normalize


station_name = 'WindlessBight'

if os.path.isdir('D:\Visitor\Matt_work\Formatted_AWS\AWS_Holland\Ross Ice Shelf Data\WindlessBight01-17'):
    paths = {'Gill':'D:\Visitor\Matt_work\Formatted_AWS\Ross_Ice_Shelf_Data\Gill01-17',
             'WindlessBight':'D:\Visitor\Matt_work\Formatted_AWS\AWS_Holland\Ross Ice Shelf Data\WindlessBight01-17'}
    file_path_AWS = paths[station_name] #Change this file directory  AWS_Holland\Ross Ice Shelf Data\WindlessBight01-17
    file_path_ECMWF = 'D:\Visitor\Matt_work\Formatted_ECMWF' 
else:
    file_path_AWS = os.path.join('/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project', 'stations', station_name)
    file_path_ECMWF = '/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project'

AWS_10min_file = '%s_10-min_AWS_dataset.txt' % station_name 
AWS_1hr_file = '%s_1-hour_AWS_dataset.txt' % station_name
AWS_6hr_file = '%s_6-hour_AWS_dataset.txt' % station_name
AWS_24hr_file = '%s_1-day_AWS_dataset.txt' % station_name
clean_AWS_10min_file = '%s_10-min_clean_AWS_dataset.txt' % station_name 
clean_AWS_1hr_file = '%s_1-hour_clean_AWS_dataset.txt' % station_name
clean_AWS_6hr_file = '%s_6-hour_clean_AWS_dataset.txt' % station_name
clean_AWS_24hr_file = '%s_1-day_clean_AWS_dataset.txt' % station_name

AWS_10min = pd.read_table(os.path.join(file_path_AWS, AWS_10min_file), sep = '\t', index_col = 0)
AWS_1hr = pd.read_table(os.path.join(file_path_AWS, AWS_1hr_file), sep = '\t', index_col = 0)
AWS_6hr = pd.read_table(os.path.join(file_path_AWS, AWS_6hr_file), sep = '\t', index_col = 0)
AWS_24hr = pd.read_table(os.path.join(file_path_AWS, AWS_24hr_file), sep = '\t', index_col = 0)
clean_AWS_10min = pd.read_table(os.path.join(file_path_AWS, clean_AWS_10min_file), sep = '\t', index_col = 0)
clean_AWS_1hr = pd.read_table(os.path.join(file_path_AWS, clean_AWS_1hr_file), sep = '\t', index_col = 0)
clean_AWS_6hr = pd.read_table(os.path.join(file_path_AWS, clean_AWS_6hr_file), sep = '\t', index_col = 0)
clean_AWS_24hr = pd.read_table(os.path.join(file_path_AWS, clean_AWS_24hr_file), sep = '\t', index_col = 0)


ECMWF_6hr_file = '%s_6-hour_ECMWF_ML_Reformat.txt' % station_name

ECMWF_6hr = pd.read_table(os.path.join(file_path_ECMWF, ECMWF_6hr_file), sep = '\t', index_col = 0)


ecmwf = ECMWF_6hr.drop(columns = ECMWF_6hr.columns[[0, 1, 2, 9, 10]])
ecmwf = ecmwf.add_prefix('ECMWF ')
ecmwf['date_time'] = ecmwf.index
aws_w = clean_AWS_6hr
aws = pd.merge(aws_w[aws_w.columns[:4]], aws_w[aws_w.columns[4:]].add_prefix('AWS '), left_index = True, right_index = True )


                
df = pd.merge(aws, ecmwf, left_index = True, right_index = True)
df['Avg Wind Direction Difference'] = (df['AWS Avg Wind Direction'] - df['ECMWF Avg Wind Direction']).apply(lambda x:
                x if np.abs(x) < 180 else (360 - np.abs(x))*np.sign(x)*(-1))
df.name = station_name

def missing_val_cutoff(df, cutoff =  10):
    for metric in ['Wind Speed (m/s)', 'V Wind Speed (m/s)', 'U Wind Speed (m/s)',  'Pressure (hPa)', 'Temperature (C)']:
        df[['AWS Min %s' % metric, 'AWS Avg %s' % metric, 'AWS Max %s' % metric]] = df.where(
                df['AWS Number of Missing %s Measurements' % metric] < 36 - cutoff, np.nan)[[
                'AWS Min %s' % metric, 'AWS Avg %s' % metric, 'AWS Max %s' % metric]]
    df['AWS Avg Wind Direction'] = df.where(df['AWS Number of Missing Wind Direction Measurements'] < 36 - cutoff, np.nan)['AWS Avg Wind Direction']
    return df

df = missing_val_cutoff(df)
df.to_csv(os.path.join(file_path_ECMWF, '%s_combined_ml.txt' % df.name), sep = '\t', index = True)


#%%
import numpy as np

print(np.corrcoef(df[~np.isnan(df['AWS Avg Wind Speed (m/s)'])]['AWS Avg Wind Speed (m/s)'], 
                  df[~np.isnan(df['AWS Avg Wind Speed (m/s)'])]['ECMWF Avg Wind Speed (m/s)'])**2)
#%%
normalize.to_csv(os.path.join(file_path_AWS, 'Gill_Combined_Since-2015_normalized-toWholeDataset_6hr_ML_dataset.txt'), sep = '\t', index = True)
#df.to_csv(os.path.join(file_path_AWS, 'WindlessBight_Combined_6hr_Zeros-Removed_ML_dataset.txt'), sep = '\t', index = True)
#monthly_df.to_csv(os.path.join(file_path_AWS, 'WindlessBight_Combined_Monthly_ML_dataset.txt'), sep = '\t', index = True)
#monthly_df.to_csv(os.path.join(file_path_AWS, 'WindlessBight_Combined_Monthly_Zeros-Removed_ML_dataset.txt'), sep = '\t', index = True)

