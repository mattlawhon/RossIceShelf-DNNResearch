#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 10:09:07 2018

@author: Matt_Lawhon
"""


import sys
import numpy as np
import pandas as pd
#'Alexander(TT!)', 'Elaine', 'Emilia', 'Emma', 'Gill', 'Lettau',
#                     'Margaret', 'Marilyn', 'Sabrina', 'Schwerdtfeger',  'Vito'
import os

import sklearn.neural_network

station_list = ['Margaret', 'Marilyn']

for station_name in station_list:
    
    redo_aws = False
    redo_ecmwf = False
    
    
    if os.path.isdir('D:\Visitor\Matt_work\Formatted_AWS\AWS_Holland\Ross Ice Shelf Data\WindlessBight01-17'):
    #    file_path_AWS = paths[station_name] #Change this file directory  AWS_Holland\Ross Ice Shelf Data\WindlessBight01-17
        file_path_ECMWF = 'D:\Visitor\Matt_work\Formatted_ECMWF' 
    else:
        file_path_AWS = os.path.join('/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project', 'AWS', station_name)
        file_path_ECMWF = os.path.join('/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project', 
                                       'ECMWF', station_name)
        script_file_path = '/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project'
        graph_path = os.path.join('/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project/Graphs',
                                  station_name)
        combined_path = os.path.join('/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project', 'Combined')
        
    #    file_path = os.path.join('/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project/presentation',
    #                         station
    #if os.path.isdir('D:\Visitor\Matt_work\Formatted_AWS\AWS_Holland\Ross Ice Shelf Data\WindlessBight01-17'):
    #    file_path = os.path.join('D:\Visitor\Matt_work', 'Summary', 'Data Analysis')
    
    clean_AWS_6hr_file = '%s_6-hour_clean_AWS_dataset.txt' % station_name
    if (not os.path.exists(os.path.join(file_path_AWS, clean_AWS_6hr_file))) or redo_aws:
        try:
            from aws_format import get_aws
            get_aws(station_name)
        except FileNotFoundError:
            raise
        except:
            sys.path.insert(0, script_file_path)
            from aws_format import get_aws
            get_aws(station_name)
    
    clean_AWS_6hr = pd.read_table(os.path.join(file_path_AWS, clean_AWS_6hr_file), sep = '\t', index_col = 0)

    ECMWF_6hr_file = '%s_6-hour_ECMWF_ML_Reformat.txt' % station_name
    if (not os.path.exists(os.path.join(file_path_ECMWF, ECMWF_6hr_file))) or redo_ecmwf:
        try:
            get_ecmwf(station_name)
        except NameError:
            try:
                from ecmwf_format import get_ecmwf
                get_ecmwf(station_name)
            except FileNotFoundError:
                raise
            except ImportError:
                sys.path.insert(0, script_file_path)
                from ecmwf_format import get_ecmwf
                get_ecmwf(station_name)
        except FileNotFoundError:
            raise
    
    ECMWF_6hr = pd.read_table(os.path.join(file_path_ECMWF, ECMWF_6hr_file), sep = '\t', index_col = 0)
    
    if (redo_aws or redo_ecmwf) or (not (os.path.exists(os.path.join(combined_path, '%s_combined_ml.txt' % station_name)))):
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
        df.to_csv(os.path.join(combined_path, '%s_combined_ml.txt' % df.name), sep = '\t', index = True)
    
    df = pd.read_csv(os.path.join(combined_path, '%s_combined_ml.txt' % station_name), sep = '\t', index_col = 0)
    df.name = station_name

    print('%s Done!' % station_name)




corr_list = [[], [], [], [], []]

#raw = []
#['Alexander(TT!)', 'Elaine', 'Emilia', 'Emma', 'Gill',
#                     'Lettau', 'Margaret', 'Marilyn', 'Sabrina', 'Schwerdtfeger',
#                     'Vito']
for station_name in station_list:
    
    combined_path = os.path.join('/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project', 'Combined')
        
    df = pd.read_csv(os.path.join(combined_path, '%s_combined_ml.txt' % station_name), sep = '\t', index_col = 0)
    df.name = station_name
    trial_corr = [[], []]
    for trial in range(1):
#    woo = df[['AWS Avg Pressure (hPa)', 'ECMWF Avg Pressure (hPa)']].dropna()
#    raw.append(np.std(woo['ECMWF Avg Pressure (hPa)'] - woo['AWS Avg Pressure (hPa)']))
#    
        working_df = df[df.columns[[0, 1, 2, 3, 4, 5, 6, 22, 23, 24, 45, 46, 49, 50, 
                                    53, 54, 57, 58]]].dropna()
        working_df = ((working_df - working_df.min())/(working_df.max() - working_df.min()))
        working_df = pd.merge(working_df, pd.DataFrame(df['AWS Avg Wind Speed (m/s)']), 
                              right_index = True, left_index = True)
        
        train_test = working_df.dropna()
    #    raw.append((len(working_df) - len(train_test))/len(train_test))
        train_test = train_test.sample(frac = 1)
        train = train_test.head(int(np.round(.8*len(train_test))))
        test = train_test.tail(len(train_test) - int(np.round(.8*len(train_test))))
        to_fill_in = working_df[np.isnan(working_df['AWS Avg Wind Speed (m/s)'])]
        
        nn = sklearn.neural_network.MLPRegressor(tol =  0.0003, solver ='lbfgs', max_iter = 3300, 
                                                 learning_rate_init = 0.001, learning_rate = 'constant',
                                                 hidden_layer_sizes = (14, 16), alpha = 0.01, activation = 'relu')
        nn.fit(train[train.columns[:-1]], train[train.columns[-1:]].values.ravel())
         
        
#        trial_corr[1].append(np.corrcoef(nn.predict(test[test.columns[:-1]]), test[test.columns[-1:]].values.ravel())[0, 1])
        to_fill_in['AWS Avg Wind Speed (m/s)_x'] = nn.predict(to_fill_in[to_fill_in.columns[:-1]])
        
        df = pd.merge(df, pd.DataFrame(to_fill_in['AWS Avg Wind Speed (m/s)_x']), 
                      right_index = True, left_index = True, how='left')
        df.name = station_name
        df['AWS Avg Wind Speed (m/s)'] = df.where(~df['AWS Avg Wind Speed (m/s)'].isna(), df['AWS Avg Wind Speed (m/s)_x'], axis =0)['AWS Avg Wind Speed (m/s)']
        df.to_csv(os.path.join(combined_path, '%s_combined_ml_filled.txt' % df.name), sep = '\t', index = True)
#        full = (df[df.columns[[0, 1, 2, 3, 4, 5, 6, 22, 23, 24, 45, 46, 49, 50, 
#                                    53, 54, 57, 58]]].dropna()).append(to_fill_in['AWS Avg Wind Speed (m/s)'])
#        full = full.sort_index()
#        comparison = pd.merge(pd.DataFrame(full['AWS Avg Wind Speed (m/s)']), 
#                              pd.DataFrame(df['ECMWF Avg Wind Speed (m/s)']),
#                              left_index = True, right_index = True)
#        trial_corr[0].append(np.corrcoef(comparison['AWS Avg Wind Speed (m/s)'], comparison['ECMWF Avg Wind Speed (m/s)'])[0, 1])
#    corr_list[1].append(np.mean(trial_corr[0]))
#    corr_list[2].append(np.mean(trial_corr[1]))
#    print(np.abs(corr_list[1][-1] - trial_corr[0][-1]), corr_list[1][-1])
#    raw[0].append(df[['AWS Avg Wind Speed (m/s)', 'ECMWF Avg Wind Speed (m/s)']].corr().values[0, 1])
#    raw[1].append(df[['AWS Avg U Wind Speed (m/s)', 'ECMWF Avg U Wind Speed (m/s)']].corr().values[0, 1])
#    raw[2].append(df[['AWS Avg V Wind Speed (m/s)', 'ECMWF Avg V Wind Speed (m/s)']].corr().values[0, 1])
#    corr_list[0].append(station_name)
#    corr_list[3].append(df['ECMWF Latitude'].iloc[-1])
#    corr_list[4].append(df['ECMWF Longitude'].iloc[-1])
    print(station_name)
    
#print(np.nanmean(raw))
    #%%
with open('/Users/Matt_Lawhon/Documents/Internships/ANN Antarctica Project/corr_test.txt', 'w') as f:
    for i in range(len(corr_list[0])):
        for j in range(5):
            f.write(str(corr_list[j][i]) + ' ')
        f.write('\n')

#%%


from sklearn.model_selection import RandomizedSearchCV
nn = sklearn.neural_network.MLPRegressor()
random_grid = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
               'alpha': [0.00001, 0.00003, 0.0001, .0003, .001, .003, .01, .03],
               'hidden_layer_sizes': [(x, y) for x in [5, 9, 14, 17, 23, 28] for y in [2, 6, 11, 16, 20]],
               'learning_rate': ['constant', 'invscaling'],
               'learning_rate_init': [0.00003,0.0001,0.0003,0.001, 0.003, 0.01, 0.05, .1],
               'max_iter': [100, 330, 700, 1000, 2200, 3300, 5000, 10000],
               'solver': ['lbfgs', 'adam'],
               'tol': [0.00003,0.0001,0.0003,0.001, 0.003, 0.01, 0.05, .1]}
                
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = nn, 
                               param_distributions = random_grid, 
                               n_iter = 100, cv = 3, 
                               verbose=2, random_state=42)

rf_random.fit(train[train.columns[:-1]], train[train.columns[-1:]].values.ravel())
                
print(rf_random.best_params_)


'''
{'tol': 0.01, 'solver': 'lbfgs', 'max_iter': 700, 
'learning_rate_init': 0.05, 'learning_rate': 'constant',
'hidden_layer_sizes': (23, 6), 'alpha': 0.003, 'activation': 'relu'}
 
{'tol': 0.0003, 'solver': 'lbfgs', 'max_iter': 3300, 
'learning_rate_init': 0.003, 'learning_rate': 'constant',
'hidden_layer_sizes': (9, 16), 'alpha': 1e-05, 'activation': 'relu'}
#'''





