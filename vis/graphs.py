# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:35:05 2018

 Code for graphs created from the Gill Current file data frames



@author: Matt Lawhon
"""



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
import os
from matplotlib.colors import LinearSegmentedColormap 
from matplotlib.dates import date2num
import matplotlib.colors as colors
from math import sqrt

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def full_duration_plot(dfc, metric, file_path, pp, save = False):
    
    df = dfc.copy()
    size = (50, 10)
    year_diff = '%d-%d' % (np.nanmin(df['Year']), np.nanmax(df['Year']))
#    plt.rcParams['axes.facecolor'] = (.6, .6, .6)
    title = '%s by Year at %s Station %s' % (metric, dfc.name, year_diff)
    datetime = pd.to_datetime(df['date_time']).values
    
    for i in range(1, 12):
        df['dt_shift'] = date2num(pd.to_datetime(df['date_time']).shift(-i).values) - date2num(datetime)
        df = df.where(df['dt_shift'] < i/2, np.nan)

    if (metric == 'Pressure (hPa)'):
        f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize = (40, 8), 
                                    gridspec_kw = {'height_ratios':[15, 2]})

        ax.plot(date2num(datetime), df['ECMWF Avg %s' % metric], label = 'ECMWF Data', 
                alpha = .5, color = (0, .4, 1))
        ax.plot(date2num(datetime), df['AWS Avg %s' % metric], label = 'AWS Data',
                alpha = .5, color = (.9, .4, 0))
        ax.plot(date2num(datetime), df['ECMWF Avg %s' % metric].rolling(56).mean(), 
                color = (0, .8, 1), label = 'ECMWF Rolling Mean')
        ax.plot(date2num(datetime), df['AWS Avg %s' % metric].rolling(56, min_periods = 32).mean(),
                color = (1, .85, 0, 1), label = 'AWS Rolling Mean')
        ax2.plot(date2num(datetime), (df['AWS Avg %s' % metric] - df['ECMWF Avg %s' % metric]).rolling(56, min_periods = 32).mean(), 
                 label = 'Difference (AWS - ECMWF)', color = (.9,0.1, 0.1))
        ax.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        plt.tight_layout()
        ax.tick_params(labeltop=False) 
        ax.grid(color = (.5, .5, .5))
        ax.set_title(title)
        ax.set_xticks(date2num(pd.date_range(start = '%s-%s-%s'% (int(np.nanmin(df['Year'])), int(np.nanmin(df['Month'])), int(np.nanmin(df['Day']))), 
                                      end = '%s-%s-%s'% (int(np.nanmax(df['Year'])), int(np.nanmax(df['Month'])), int(np.nanmax(df['Day']))), freq = 'YS')))
        plt.figlegend(loc = (.0189, .822), facecolor = (1, 1, 1))
        f.text(0.002, 0.5, metric, ha='center', va='center', rotation='vertical')

    else:
        plt.figure(figsize = size)
        
        if (metric == 'Wind Direction'):
            AWS_wd = np.degrees(np.arctan2(df['AWS Avg U Wind Speed (m/s)'].rolling(56, min_periods = 32).mean(), 
                                           df['AWS Avg V Wind Speed (m/s)'].rolling(56, min_periods = 32).mean())) + 180
            AWS_mean = np.degrees(np.arctan2(np.nanmean(df['AWS Avg U Wind Speed (m/s)']), 
                                            np.nanmean(df['AWS Avg V Wind Speed (m/s)']))) + 180 
            ECMWF_wd = np.degrees(np.arctan2(df['ECMWF Avg U Wind Speed (m/s)'].rolling(56).mean(),
                                             df['ECMWF Avg V Wind Speed (m/s)'].rolling(56).mean())) + 180
            ECMWF_mean = np.degrees(np.arctan2(np.nanmean(df['ECMWF Avg U Wind Speed (m/s)']), 
                                            np.nanmean(df['ECMWF Avg V Wind Speed (m/s)']))) + 180 
            
            plt.plot(date2num(datetime), df['ECMWF Avg %s' % metric].apply(lambda x: x-360 if x > ECMWF_mean + 180 else x), 
                     label = 'ECMWF Data', alpha = .5, color = (0, .4, 1))
            plt.plot(date2num(datetime), df['AWS Avg %s' % metric].apply(lambda x: x-360 if x > AWS_mean + 180 else x), 
                     label = 'AWS Data', alpha = .5, color = (.9, .4, 0))
            
            plt.yticks(range(-180, 541, 45), (np.arange(-180, 541, 45).astype(str) + np.array(['° (S) ',
                       '° (SW)', '° (W) ', '° (NW)','° (N) ','° (NE)', '° (E) ', '° (SE)', '° (S) ',
                       '° (SW)', '° (W) ', '° (NW)','° (N) ','° (NE)', '° (E) ', '° (SE)', '° (S) '], dtype = object)))
    

            plt.plot(date2num(datetime), ECMWF_wd.apply(lambda x: x-360 if x > ECMWF_mean + 180 else x), color = (0, .8, 1), label = 'ECMWF Rolling Mean')
            plt.plot(date2num(datetime), AWS_wd.apply(lambda x: x-360 if x > AWS_mean + 180 else x), color = (1, .85, 0, 1), label = 'AWS Rolling Mean')
  

        else:
            plt.plot(date2num(datetime), df['ECMWF Avg %s' % metric], 
                     label = 'ECMWF Data', alpha = .5, color = (0, .4, 1))
            plt.plot(date2num(datetime), df['AWS Avg %s' % metric], 
                     label = 'AWS Data', alpha = .5, color = (.9, .4, 0))
            plt.plot(date2num(datetime), df['ECMWF Avg %s' % metric].rolling(56).mean(), 
                     color = (0, .8, 1), label = 'ECMWF Rolling Mean')
            plt.plot(date2num(datetime), df['AWS Avg %s' % metric].rolling(56, min_periods = 32).mean(), 
                     color = (1, .85, 0, 1), label = 'AWS Rolling Mean')
            plt.plot(date2num(datetime), (df['AWS Avg %s' % metric] - df['ECMWF Avg %s' % metric]).rolling(56, min_periods = 32).mean(), 
                     label = 'Difference (AWS - ECMWF)', color = (.9,0.1, 0.1))
            
        plt.title(title)
        plt.legend(loc = 2, facecolor = (1, 1, 1))
        plt.ylabel(metric) 

    plt.xticks(date2num(pd.date_range(start = '%s-%s-%s'% (int(np.nanmin(df['Year'])), int(np.nanmin(df['Month'])), int(np.nanmin(df['Day']))), 
                                      end = '%s-%s-%s'% (int(np.nanmax(df['Year'])), int(np.nanmax(df['Month'])), int(np.nanmax(df['Day']))), freq = 'YS')), 
                                       range(int(np.nanmin(df['Year'])), int(np.nanmax(df['Year'])+1)))
    
    plt.grid(color = (.5, .5, .5))
    plt.xlabel('Year', labelpad = 5)
    if save:
        if (metric == 'Pressure (hPa)'):
            f.savefig(os.path.join(file_path, (title+ '.pdf').replace('/', '÷')),
                      format = 'PDF', bbox_inches = 'tight')
        else:
            plt.savefig(os.path.join(file_path, (title+ '.pdf').replace('/', '÷')),
                        format = 'PDF', bbox_inches = 'tight')
        pp.savefig()
    else:
        plt.show()
    plt.close()
    
def kde_missing_vals(dfc, metric, file_path, pp, size = (12, 8), save = False):
    
    df = dfc.copy()
    year_diff = '%d-%d' % (df['Year'].iloc[0], df['Year'].iloc[-1])
    title = 'Estimation of Seasonality of Missing %s Values at %s Station %s' % (metric, dfc.name, year_diff)
    df['AWS Avg %s' % metric] = np.where(np.isnan(df['AWS Avg %s' % metric]), 0, 1)
    count_nan = (df.groupby('Julian Day')['AWS Avg %s' % metric].mean())
    y = np.polyval(np.polyfit(range(np.nanmin(df['Julian Day']), 
                                    np.nanmax(df['Julian Day']) + 1),
                              count_nan.values, deg = 50),
                              range(np.nanmin(df['Julian Day']), 
                                    np.nanmax(df['Julian Day']) + 1))
    y = np.where(y<1, y, 1)

#    plt.rcParams['axes.facecolor'] = (.6, .6, .6)
    plt.figure(figsize = size)
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Percent of %s Data Present' % metric)
    plt.axis([-9, 376, -4, 104])
    plt.yticks(range(0, 110, 10))

    plt.plot(range(np.nanmin(df['Julian Day']), np.nanmax(df['Julian Day']) + 1),
             y*100, color = (1, .85, 0))
    plt.fill_between(range(np.nanmin(df['Julian Day']), 
                           np.nanmax(df['Julian Day']) + 1),
                     y*100, color = (1, .85, 0))
    plt.xticks([1, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 366], ['      Jan',
               '     Feb', '      Mar', '      Apr', '      May', '      Jun', '      Jul',
               '      Aug','      Sep', '      Oct', '     Nov', '      Dec', '      '], ha = 'left')

    plt.grid(color = (.5, .5, .5))
    if save:
        plt.savefig(os.path.join(file_path, (title+ '.pdf').replace('/', '÷')),
                    format = 'PDF', bbox_inches = 'tight')
        pp.savefig()
    else:
        plt.show()
    plt.close()
   
    
def AvE_corr(dfc, metric, file_path, pp, size = (15, 12), save = False):
    df = dfc.copy()
    year_diff = '%d-%d' % (np.nanmin(df['Year']), np.nanmax(df['Year']))
    plt.figure(figsize = size)
    df = df[~np.isnan(df['AWS Avg %s' % metric])]
    x_min = np.nanmin(df['AWS Avg %s' % metric])-5
    x_max = np.nanmax(df['AWS Avg %s' % metric])+5
    cmap_JD = LinearSegmentedColormap.from_list('mycmap', [(.93, 0, .07), 'blue',(.93, 0, .07)])
    title = 'AWS %s vs. ECMWF %s Correlation at %s Station %s' % (metric, metric, dfc.name, year_diff)
#    plt.rcParams['axes.facecolor'] = (.6, .6, .6)
                
    if (metric == 'Wind Direction'):
       plt.xticks(range(0, 361, 45), (np.arange(0, 361, 45).astype(str) + np.array(['° (N) ','° (NE)', '° (E) ',
                                      '° (SE)', '° (S) ', '° (SW)', '° (W) ', '° (NW)','° (N) '], dtype = object)))
       plt.yticks(range(0, 361, 45), (np.arange(0, 361, 45).astype(str) + np.array(['° (N) ','° (NE)', '° (E) ',
                                      '° (SE)', '° (S) ', '° (SW)', '° (W) ', '° (NW)','° (N) '], dtype = object)))
       plt.axis([x_min-(x_max-x_min)/20, x_max+(x_max-x_min)/20, x_min-(x_max-x_min)/20, x_max+(x_max-x_min)/20])
       slope, intercept, r_value, p_value, std_err = stats.linregress(df['AWS Avg %s' % metric], 
                                                                      df['AWS Avg %s' % metric] - df['Avg %s Difference' % metric])
    
    else:
        plt.axis([x_min-(x_max-x_min)/10, x_max+(x_max-x_min)/10, x_min-(x_max-x_min)/10, x_max+(x_max-x_min)/10])
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['AWS Avg %s' % metric], 
                                                                       df['ECMWF Avg %s' % metric])

    plt.plot([x_min, x_max], [x_min*slope + intercept, x_max*slope + intercept],
             label = 'LSRL, r = %s' % str(round(r_value, 3)), color = (.5, .5, .5), linewidth = 2)
    plt.plot([-100, 1200], [-100, 1200], color = (.5, .5, .5), linewidth = 1)
    plt.scatter(df['AWS Avg %s' % metric], df['ECMWF Avg %s' % metric], 
                c = df['Julian Day'],cmap=cmap_JD, s= 1)
    plt.xlabel('AWS Avg %s' % (metric))
    plt.colorbar(label = 'Julian Day')
    plt.ylabel('ECMWF %s' % (metric))
    plt.title(title)
    plt.grid(color = (.5, .5, .5))
    plt.legend(loc = 2, facecolor = (0.9, 0.9, 0.9))
    if save:
        plt.savefig(os.path.join(file_path, (title+ '.pdf').replace('/', '÷')),
                    format = 'PDF', bbox_inches = 'tight')
        pp.savefig()
    else:
        plt.show()
    plt.close()

def wspeed_wdir(dfc, file_path, pp, size = (13, 10), save = False):
    df = dfc.copy()
    year_diff = '%d-%d' % (np.nanmin(df['Year']), np.nanmax(df['Year']))
        
    cmap_JD = LinearSegmentedColormap.from_list('mycmap', [(.93, 0, .07), 'blue',(.93, 0, .07)])
    title = 'AWS Avg Wind Speed vs. AWS Avg Wind Direction at %s Station %s' % (dfc.name, year_diff)
    plt.figure(figsize = size)
#    plt.rcParams['axes.facecolor'] = (.6, .6, .6) 
    plt.scatter(df['AWS Avg Wind Speed (m/s)'], df['AWS Avg Wind Direction'], s= 1, 
                c = df['Julian Day'], cmap = cmap_JD)
    plt.colorbar(label = 'Julian Day')
    plt.grid(color = (.5, .5, .5))
    plt.yticks(range(0, 361, 30))
    plt.yticks(range(0, 361, 45), (np.arange(0, 361, 45).astype(str) + np.array(['° (N) ','° (NE)',
                                   '° (E) ','° (SE)', '° (S) ', '° (SW)', '° (W) ', '° (NW)','° (N) '], dtype = object)))
    plt.xlabel('AWS Avg Wind Speed (m/s)')
    plt.ylabel('AWS Avg Wind Direction')
    plt.title(title)
    if save:
        plt.savefig(os.path.join(file_path, title + '.pdf'), 
                    format = 'PDF', bbox_inches = 'tight')
        pp.savefig()
    else:
        plt.show()
    plt.close() 
    
    title = 'AWS Avg U Wind Speed vs. AWS Avg V Wind Speed at %s Station %s' % (dfc.name, year_diff)
    plt.figure(figsize = size)
    plt.scatter(df['AWS Avg U Wind Speed (m/s)'], df['AWS Avg V Wind Speed (m/s)'], s=1, 
                c = df['Julian Day'], cmap = cmap_JD)
    plt.grid(color = (.5, .5, .5))
    plt.colorbar(label = 'Julian Day')
    plt.xlabel('AWS Avg U Wind Speed (m/s from W)')
    plt.ylabel('AWS Avg V Wind Speed (m/s from S)')
    plt.title(title)
    if save:
        plt.savefig(os.path.join(file_path,(title+ '.pdf').replace('/', '÷')),
                    format = 'PDF', bbox_inches = 'tight')
        pp.savefig()
    else:
        plt.show()
    plt.close()

def wspeed_wdirEC(dfc, file_path, pp, size = (13, 10), save = False):
    df = dfc.copy()
    year_diff = '%d-%d' % (np.nanmin(df['Year']), np.nanmax(df['Year']))
        
    cmap_JD = LinearSegmentedColormap.from_list('mycmap', [(.93, 0, .07), 'blue',(.93, 0, .07)])
    title = 'ECMWF Avg Wind Speed vs. ECMWF Avg Wind Direction at %s Station %s' % (dfc.name, year_diff)
    plt.figure(figsize = size)
#    plt.rcParams['axes.facecolor'] = (.6, .6, .6) 
    plt.scatter(df['ECMWF Avg Wind Speed (m/s)'], df['ECMWF Avg Wind Direction'], s= 1, 
                c = df['Julian Day'], cmap = cmap_JD)
    plt.colorbar(label = 'Julian Day')
    plt.grid(color = (.5, .5, .5))
    plt.yticks(range(0, 361, 30))
    plt.yticks(range(0, 361, 45), (np.arange(0, 361, 45).astype(str) + np.array(['° (N) ','° (NE)',
                                   '° (E) ','° (SE)', '° (S) ', '° (SW)', '° (W) ', '° (NW)','° (N) '], dtype = object)))
    plt.xlabel('ECMWF Avg Wind Speed (m/s)')
    plt.ylabel('ECMWF Avg Wind Direction')
    plt.title(title)
    if save:
        plt.savefig(os.path.join(file_path, title + '.pdf'), 
                    format = 'PDF', bbox_inches = 'tight')
        pp.savefig()
    else:
        plt.show()
    plt.close() 
    
    title = 'ECMWF Avg U Wind Speed vs. ECMWF Avg V Wind Speed at %s Station %s' % (dfc.name, year_diff)
    plt.figure(figsize = size)
    plt.scatter(df['ECMWF Avg U Wind Speed (m/s)'], df['ECMWF Avg V Wind Speed (m/s)'], s=1, 
                c = df['Julian Day'], cmap = cmap_JD)
    plt.grid(color = (.5, .5, .5))
    plt.colorbar(label = 'Julian Day')
    plt.xlabel('ECMWF Avg U Wind Speed (m/s from W)')
    plt.ylabel('ECMWF Avg V Wind Speed (m/s from S)')
    plt.title(title)
    if save:
        plt.savefig(os.path.join(file_path,(title+ '.pdf').replace('/', '÷')),
                    format = 'PDF', bbox_inches = 'tight')
        pp.savefig()
    else:
        plt.show()
    plt.close()
    
    
def monthly_mean(dfc, metric, file_path, pp, size = (12, 8), save = False, whole = False):
    if not np.isnan(np.nanmean(dfc['AWS Avg %s' % metric])):
        df = dfc.copy()
        year_diff = '%d-%d' % (np.nanmin(df['Year']), np.nanmax(df['Year']))
        title = 'Monthly Average %s at %s Station %s' % (metric, dfc.name, year_diff)
#        plt.rcParams['axes.facecolor'] = (.6, .6, .6)
        plt.figure(figsize = (12, 8))
        percent_missing = 100 - df.groupby('Month')['AWS Number of Missing %s Measurements' % metric].mean()/.36
        if whole:
            missing_max = np.max((100 - df.groupby('Month')[[col for col in df.columns if 'Missing' in col]].mean()/.36).values)/100
            missing_min = np.min((100 - df.groupby('Month')[[col for col in df.columns if 'Missing' in col]].mean()/.36).values)/100
            missing_del = missing_max - missing_min
       
            cmap_missing = truncate_colormap(LinearSegmentedColormap.from_list('mycmap', [(1, 0, 0), (0, 1, 0)]),
                                             (np.min(percent_missing)/100 - missing_min)/missing_del,
                                             (np.max(percent_missing)/100 - missing_min)/missing_del)
        else:
            cmap_missing = LinearSegmentedColormap.from_list('mycmap', [(.9, 0, 0), (0, .9, 0)])
        
        if (metric == 'Wind Direction'):
            V = df.groupby('Month')['AWS Avg V Wind Speed (m/s)']
            U = df.groupby('Month')['AWS Avg U Wind Speed (m/s)']
            
            plt.errorbar(V.mean().index, np.degrees(np.arctan2(U.mean().values, V.mean().values))+180,
                         yerr = 3*np.degrees(np.arctan2(U.std().values, V.std().values))/np.sqrt(df.groupby('Month').size()),
                         capsize = 8, zorder = 4, label = 'ECMWF Avg '+metric+' Standard Error 99.7% CI', c = (1, .85, 0))
            im = plt.scatter(V.mean().index, np.degrees(np.arctan2(U.mean().values, V.mean().values))+180,
                        c = percent_missing, edgecolors = (.2, .2, .2), cmap = cmap_missing,
                        zorder = 5, s = 100, label = '')
            
            ymin = float(np.min(np.degrees(np.arctan2(U.mean().values, V.mean().values))+160))
            ymax = float(np.max(np.degrees(np.arctan2(U.mean().values, V.mean().values))+200))
            
            V = df.groupby('Month')['ECMWF Avg V Wind Speed (m/s)']
            U = df.groupby('Month')['ECMWF Avg U Wind Speed (m/s)']
            
            plt.errorbar(V.mean().index, np.degrees(np.arctan2(U.mean().values, V.mean().values))+180,
                         yerr = 3*np.degrees(np.arctan2(U.std().values, V.std().values))/np.sqrt(df.groupby('Month').size()),
                         capsize = 8, zorder = 2, label = 'ECMWF Avg '+metric+' Standard Error 99.7% CI')
            plt.scatter(V.mean().index, np.degrees(np.arctan2(U.mean().values, V.mean().values))+180,
                        c = (.4, .4, .4), zorder = 3, s = 50, label = '')
            
            
                     
            ymin = np.nanmin([np.min(np.degrees(np.arctan2(U.mean().values, V.mean().values))+160), ymin])
            ymax = np.nanmax([np.max(np.degrees(np.arctan2(U.mean().values, V.mean().values))+200), ymax])
            
            plt.yticks(np.arange(0, 361, 22.5), (np.arange(0, 361, 22.5).astype(str) + np.array(['° (N)  ','° (NNE)','° (NE) ', '° (ENE)',
                                           '° (E)  ','° (ESE)', '° (SE) ','° (SSE)', '° (S)  ','° (SSW)', '° (SW) ','° (WSW)',
                                           '° (W)  ', '° (WNW)', '° (NW)', '° (NNW)','° (N) '], dtype = object)))
        
            plt.axis([.5, 12.5, ymin, ymax])        
        else:
            plt.errorbar(df.groupby('Month')['AWS Avg %s' % metric].mean().index,
                         df.groupby('Month')['AWS Avg %s' % metric].mean(), capsize = 8,
                         yerr = 3* df.groupby('Month')['AWS Avg %s' % metric].std()/np.sqrt(df.groupby('Month').size()),
                         zorder = 4, label = 'AWS Avg '+metric+' Standard Error 99.7% CI', c = (1, .85, 0)) #Update this yellow color.
            im = plt.scatter(df.groupby('Month')['AWS Avg %s' % metric].mean().index,
                        df.groupby('Month')['AWS Avg %s' % metric].mean(), 
                        c = percent_missing, edgecolors = (.2, .2, .2), cmap = cmap_missing, 
                        zorder = 5, s = 100, label = '')
            if (('ECMWF Avg %s' % metric) in df.columns):
                plt.errorbar(df.groupby('Month')['ECMWF Avg %s' % metric].mean().index,
                             df.groupby('Month')['ECMWF Avg %s' % metric].mean(), capsize = 8,
                             yerr = 3* df.groupby('Month')['ECMWF Avg %s' % metric].std()/np.sqrt(df.groupby('Month').size()),
                             zorder = 2, label = 'ECMWF Avg '+metric+' Standard Error 99.7% CI')
                plt.scatter(df.groupby('Month')['ECMWF Avg %s' % metric].mean().index,
                            df.groupby('Month')['ECMWF Avg %s' % metric].mean(), c= (.4, .4, .4),
                            zorder = 3, s = 50, label = '')
            
            plt.xlim(.5, 12.5)
            ymin, ymax = plt.ylim()
            plt.ylim(ymin, ymax + (ymax-ymin)*.1)
            
        plt.legend(loc = 2, facecolor = (1, 1, 1))
        plt.grid(zorder = 1, color = (.5, .5, .5))
        plt.ylabel(metric)
        plt.colorbar(im, label = 'Percent of %s Measurements not null' % metric)
        plt.xlabel('Month')    
        plt.xticks(np.arange(.5, 12.5), ['     Jan','     Feb', '     Mar', '      Apr',
                       '     May', '     Jun', '      Jul','     Aug','     Sep', '     Oct',
                       '     Nov', '     Dec', '      '], ha = 'left')
        plt.title(title)
        plt.legend(loc = 2, facecolor = (1, 1, 1))
        if save:
            plt.savefig(os.path.join(file_path,(title+ '.pdf').replace('/', '÷')), 
                        format = 'PDF', bbox_inches = 'tight')
            pp.savefig()
        else:
            plt.show()
        plt.close()                   



#%%
import sys
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#'Alexander(TT!)', 'Elaine', 'Emilia', 'Emma', 'Gill', 'Lettau',
#                     'Margaret', 'Marilyn', 'Sabrina', 'Schwerdtfeger',  'Vito'
import os
for station_name in ['Margaret', 'Marilyn']:
    
    redo_aws = False
    redo_ecmwf = False
    save_boolean = True
    
    
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

    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    if os.path.exists(os.path.join(graph_path, '%s Graphs.pdf' % station_name)):
        os.remove(os.path.join(graph_path, '%s Graphs.pdf' % station_name))
        
    pp = PdfPages(os.path.join(graph_path, '%s Graphs.pdf' % station_name))
#    ['Wind Speed (m/s)', 'V Wind Speed (m/s)', 'U Wind Speed (m/s)', 
#                   'Wind Direction',  'Pressure (hPa)', 'Temperature (C)', 
#                   'Relative Humidity (%)', 'Delta-T (C)']
#    
    for metric in ['Wind Speed (m/s)', 'V Wind Speed (m/s)', 'U Wind Speed (m/s)', 
                   'Wind Direction',  'Pressure (hPa)', 'Temperature (C)', 
                   'Relative Humidity (%)', 'Delta-T (C)']:
        monthly_mean(df, metric, graph_path,pp, save = save_boolean)
        kde_missing_vals(df, metric, graph_path, pp, save = save_boolean)
        if metric != 'Relative Humidity (%)' and metric != 'Delta-T (C)':
            full_duration_plot(df, metric, graph_path, pp, save = save_boolean)
            AvE_corr(df, metric, graph_path, pp, save = save_boolean)
#    
    wspeed_wdir(df, graph_path, pp, save = save_boolean)
    wspeed_wdirEC(df, graph_path, pp, save = save_boolean)
    
    pp.close()
#
    print('%s Done!' % station_name)
#%%
def monthly_mean(dfc, metric, file_path, pp, size = (12, 8), save = False, whole = False):
    if not np.isnan(np.nanmean(dfc['AWS Avg %s' % metric])):
        df = dfc.copy()
        year_diff = '%d-%d' % (np.nanmin(df['Year']), np.nanmax(df['Year']))
        title = 'Monthly Average %s at %s Station %s' % (metric, dfc.name, year_diff)
        plt.figure(figsize = (12, 8))
        percent_missing = 100 - df.groupby('Month')['AWS Number of Missing %s Measurements' % metric].mean()/.36
        if whole:
            missing_max = np.max((100 - df.groupby('Month')[[col for col in df.columns if 'Missing' in col]].mean()/.36).values)/100
            missing_min = np.min((100 - df.groupby('Month')[[col for col in df.columns if 'Missing' in col]].mean()/.36).values)/100
            missing_del = missing_max - missing_min
       
            cmap_missing = truncate_colormap(LinearSegmentedColormap.from_list('mycmap', [(1, 0, 0), (0, 1, 0)]),
                                             (np.min(percent_missing)/100 - missing_min)/missing_del,
                                             (np.max(percent_missing)/100 - missing_min)/missing_del)
        else:
            cmap_missing = LinearSegmentedColormap.from_list('mycmap', [(.9, 0, 0), (0, .9, 0)])
        
        if (metric == 'Wind Direction'):
            V = df.groupby('Month')['AWS Avg V Wind Speed (m/s)']
            U = df.groupby('Month')['AWS Avg U Wind Speed (m/s)']
            
            plt.errorbar(V.mean().index, np.degrees(np.arctan2(U.mean().values, V.mean().values))+180,
                         yerr = 3*np.degrees(np.arctan2(U.std().values, V.std().values))/np.sqrt(df.groupby('Month').size()),
                         capsize = 6, zorder = 4, label = 'ECMWF Avg '+metric+' Standard Error 99.7% CI', c = (.3, .2, 0))
            plt.scatter(V.mean().index, np.degrees(np.arctan2(U.mean().values, V.mean().values))+180,
                        c = (.2, .2, .2), zorder = 5, s = 30, label = '')
            
            ymin = float(np.min(np.degrees(np.arctan2(U.mean().values, V.mean().values))+160))
            ymax = float(np.max(np.degrees(np.arctan2(U.mean().values, V.mean().values))+200))
            
            V = df.groupby('Month')['ECMWF Avg V Wind Speed (m/s)']
            U = df.groupby('Month')['ECMWF Avg U Wind Speed (m/s)']
            
            plt.errorbar(V.mean().index, np.degrees(np.arctan2(U.mean().values, V.mean().values))+180,
                         yerr = 3*np.degrees(np.arctan2(U.std().values, V.std().values))/np.sqrt(df.groupby('Month').size()),
                         capsize = 6, zorder = 2, label = 'ECMWF Avg '+metric+' Standard Error 99.7% CI')
            plt.scatter(V.mean().index, np.degrees(np.arctan2(U.mean().values, V.mean().values))+180,
                        c = (.2, .2, .2), zorder = 3, s = 30, label = '')
            
            ymin = np.nanmin([np.min(np.degrees(np.arctan2(U.mean().values, V.mean().values))+160), ymin])
            ymax = np.nanmax([np.max(np.degrees(np.arctan2(U.mean().values, V.mean().values))+200), ymax])
            
            plt.yticks(np.arange(0, 361, 22.5), (np.arange(0, 361, 22.5).astype(str) + np.array(['° (N)  ','° (NNE)','° (NE) ', '° (ENE)',
                                           '° (E)  ','° (ESE)', '° (SE) ','° (SSE)', '° (S)  ','° (SSW)', '° (SW) ','° (WSW)',
                                           '° (W)  ', '° (WNW)', '° (NW)', '° (NNW)','° (N) '], dtype = object)))
        
            plt.axis([.5, 12.5, ymin, ymax])        
        else:
            plt.errorbar(df.groupby('Month')['AWS Avg %s' % metric].mean().index,
                         df.groupby('Month')['AWS Avg %s' % metric].mean(), capsize = 6,
                         yerr = 3* df.groupby('Month')['AWS Avg %s' % metric].std()/np.sqrt(df.groupby('Month').size()),
                         zorder = 4, label = 'AWS Avg '+metric+' Standard Error 99.7% CI', c = (.3, .2, 0))
            plt.scatter(df.groupby('Month')['AWS Avg %s' % metric].mean().index,
                        df.groupby('Month')['AWS Avg %s' % metric].mean(),  c = (.2, .2, .2),
                        zorder = 5, s = 30, label = '')
            if (('ECMWF Avg %s' % metric) in df.columns):
                plt.errorbar(df.groupby('Month')['ECMWF Avg %s' % metric].mean().index,
                             df.groupby('Month')['ECMWF Avg %s' % metric].mean(), capsize = 6,
                             yerr = 3* df.groupby('Month')['ECMWF Avg %s' % metric].std()/np.sqrt(df.groupby('Month').size()),
                             zorder = 2, label = 'ECMWF Avg '+metric+' Standard Error 99.7% CI')
                plt.scatter(df.groupby('Month')['ECMWF Avg %s' % metric].mean().index,
                            df.groupby('Month')['ECMWF Avg %s' % metric].mean(), c = (.2, .2, .2),
                            zorder = 3, s = 30, label = '')
            
            plt.xlim(.5, 12.5)
            ymin, ymax = plt.ylim()
            plt.ylim(ymin, ymax + (ymax-ymin)*.1)
            
        plt.legend(loc = 2)
        plt.grid(zorder = 1, color = (.2, .2, .2))
        plt.ylabel(metric)
        plt.xlabel('Month')    
        plt.xticks(np.arange(.5, 12.5), ['     Jan','     Feb', '     Mar', '      Apr',
                       '     May', '     Jun', '      Jul','     Aug','     Sep', '     Oct',
                       '     Nov', '     Dec', '      '], ha = 'left')
        plt.title(title)
        if save:
            plt.savefig(os.path.join(file_path,(title+ '.pdf').replace('/', '÷')), 
                        format = 'PDF', bbox_inches = 'tight')
            pp.savefig()
        else:
            plt.show()
        plt.close()  
        
        
def kde_missing_vals(dfc, metric, file_path, pp, size = (12, 8), save = False):
    df = dfc.copy()
    year_diff = '%d-%d' % (df['Year'].iloc[0], df['Year'].iloc[-1])
    title = 'Estimation of Seasonality of Missing %s Values at %s Station %s' % (metric, dfc.name, year_diff)
    df['AWS Avg %s' % metric] = np.where(np.isnan(df['AWS Avg %s' % metric]), 0, 1)
    count_nan = (df.groupby('Julian Day')['AWS Avg %s' % metric].mean())
    y = np.polyval(np.polyfit(range(np.nanmin(df['Julian Day']), 
                                    np.nanmax(df['Julian Day']) + 1),
                              count_nan.values, deg = 50),
                              range(np.nanmin(df['Julian Day']), 
                                    np.nanmax(df['Julian Day']) + 1))
    y = np.where(y<1, y, 1)

    plt.figure(figsize = size)
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Percent of %s Data Present' % metric)
    plt.axis([-9, 376, -4, 104])
    plt.yticks(range(0, 110, 10))
    plt.grid(zorder = 100, color = (.2, .2, .2))
    plt.plot(range(np.nanmin(df['Julian Day']), np.nanmax(df['Julian Day']) + 1),
             y*100, color = (1, .85, 0), zorder = 0)
    plt.fill_between(range(np.nanmin(df['Julian Day']), 
                           np.nanmax(df['Julian Day']) + 1),
                     y*100, color = (1, .85, 0), zorder = 0)
    plt.xticks([1, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 366], ['      Jan',
               '     Feb', '      Mar', '      Apr', '      May', '      Jun', '      Jul',
               '      Aug','      Sep', '      Oct', '     Nov', '      Dec', '      '], ha = 'left')


    if save:
        plt.savefig(os.path.join(file_path, (title+ '.pdf').replace('/', '÷')),
                    format = 'PDF', bbox_inches = 'tight')
        pp.savefig()
    else:
        plt.show()
    plt.close()


def wspeed_wdir(dfc, file_path, pp, size = (13, 10), save = False):
    df = dfc.copy().sample(frac = 1)
    year_diff = '%d-%d' % (np.nanmin(df['Year']), np.nanmax(df['Year']))
        
    cmap_JD = LinearSegmentedColormap.from_list('mycmap', [(.93, 0, .07), (.07, .07, 1),(.93, 0, .07)])
    title = 'AWS Avg Wind Speed vs. AWS Avg Wind Direction at %s Station %s' % (dfc.name, year_diff)
    plt.figure(figsize = size)
#    plt.rcParams['axes.facecolor'] = (.6, .6, .6) 
    plt.scatter(df['AWS Avg Wind Speed (m/s)'], df['AWS Avg Wind Direction'], s= 1, 
                c = df['Julian Day'], cmap = cmap_JD)
    plt.colorbar(label = 'Julian Day')
    plt.grid(color = (0.2, .2, .2))
    plt.yticks(range(0, 361, 30))
    plt.yticks(range(0, 361, 45), (np.arange(0, 361, 45).astype(str) + np.array(['° (N) ','° (NE)',
                                   '° (E) ','° (SE)', '° (S) ', '° (SW)', '° (W) ', '° (NW)','° (N) '], dtype = object)))
    plt.xlabel('AWS Avg Wind Speed (m/s)')
    plt.ylabel('AWS Avg Wind Direction')
    plt.title(title)
    if save:
        plt.savefig(os.path.join(file_path, title + '.pdf'), 
                    format = 'PDF', bbox_inches = 'tight')
        pp.savefig()
    else:
        plt.show()
    plt.close() 
    
    title = 'AWS Avg U Wind Speed vs. AWS Avg V Wind Speed at %s Station %s' % (dfc.name, year_diff)
    plt.figure(figsize = size)
    plt.scatter(df['AWS Avg U Wind Speed (m/s)'], df['AWS Avg V Wind Speed (m/s)'], s=1, 
                c = df['Julian Day'], cmap = cmap_JD)
    plt.grid(color = (0.2, 0.2, 0.2))
    plt.colorbar(label = 'Julian Day')
    plt.xlabel('AWS Avg U Wind Speed (m/s from W)')
    plt.ylabel('AWS Avg V Wind Direction (m/s from S)')
    plt.title(title)
    if save:
        plt.savefig(os.path.join(file_path,(title+ '.pdf').replace('/', '÷')),
                    format = 'PDF', bbox_inches = 'tight')
        pp.savefig()
    else:
        plt.show()
    plt.close()
    
def full_duration_plot(dfc, metric, file_path, pp, save = False):
    
    df = dfc.copy()
    size = (50, 10)
    year_diff = '%d-%d' % (np.nanmin(df['Year']), np.nanmax(df['Year']))
#    plt.rcParams['axes.facecolor'] = (.6, .6, .6)
    title = '%s by Year at %s Station %s' % (metric, dfc.name, year_diff)
    datetime = pd.to_datetime(df['date_time']).values
    
    for i in range(1, 12):
        df['dt_shift'] = date2num(pd.to_datetime(df['date_time']).shift(-i).values) - date2num(datetime)
        df = df.where(df['dt_shift'] < i/2, np.nan)

    if (metric == 'Pressure (hPa)'):
        f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize = (40, 8), 
                                    gridspec_kw = {'height_ratios':[15, 2]})

        ax.plot(date2num(datetime), df['ECMWF Avg %s' % metric], label = 'ECMWF Data', 
                alpha = .5, color = (0, .4, 1))
        ax.plot(date2num(datetime), df['AWS Avg %s' % metric], label = 'AWS Data',
                alpha = .5, color = (.9, .4, 0))
        ax.plot(date2num(datetime), df['ECMWF Avg %s' % metric].rolling(56).mean(), 
                color = (0, .8, 1), label = 'ECMWF Rolling Mean')
        ax.plot(date2num(datetime), df['AWS Avg %s' % metric].rolling(56, min_periods = 32).mean(),
                color = (1, .85, 0, 1), label = 'AWS Rolling Mean')
        ax2.plot(date2num(datetime), (df['AWS Avg %s' % metric] - df['ECMWF Avg %s' % metric]).rolling(56, min_periods = 32).mean(), 
                 label = 'Difference (AWS - ECMWF)', color = (.9,0.1, 0.1))
        ax.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        plt.tight_layout()
        ax.tick_params(labeltop=False) 
        ax.grid(color = (.2, .2, .2))
        ax.set_title(title)
        ax.set_xticks(date2num(pd.date_range(start = '%s-%s-%s'% (int(np.nanmin(df['Year'])), int(np.nanmin(df['Month'])), int(np.nanmin(df['Day']))), 
                                      end = '%s-%s-%s'% (int(np.nanmax(df['Year'])), int(np.nanmax(df['Month'])), int(np.nanmax(df['Day']))), freq = 'YS')))
        plt.figlegend(loc = (.0189, .822))
        f.text(0.002, 0.5, metric, ha='center', va='center', rotation='vertical')

    else:
        plt.figure(figsize = size)
        
        if (metric == 'Wind Direction'):
            AWS_wd = np.degrees(np.arctan2(df['AWS Avg U Wind Speed (m/s)'].rolling(56, min_periods = 32).mean(), 
                                           df['AWS Avg V Wind Speed (m/s)'].rolling(56, min_periods = 32).mean())) + 180
            AWS_mean = np.degrees(np.arctan2(np.nanmean(df['AWS Avg U Wind Speed (m/s)']), 
                                            np.nanmean(df['AWS Avg V Wind Speed (m/s)']))) + 180 
            ECMWF_wd = np.degrees(np.arctan2(df['ECMWF Avg U Wind Speed (m/s)'].rolling(56).mean(),
                                             df['ECMWF Avg V Wind Speed (m/s)'].rolling(56).mean())) + 180
            ECMWF_mean = np.degrees(np.arctan2(np.nanmean(df['ECMWF Avg U Wind Speed (m/s)']), 
                                            np.nanmean(df['ECMWF Avg V Wind Speed (m/s)']))) + 180 
            
            plt.plot(date2num(datetime), df['ECMWF Avg %s' % metric].apply(lambda x: x-360 if x > ECMWF_mean + 180 else x), 
                     label = 'ECMWF Data', alpha = .5, color = (0, .4, 1))
            plt.plot(date2num(datetime), df['AWS Avg %s' % metric].apply(lambda x: x-360 if x > AWS_mean + 180 else x), 
                     label = 'AWS Data', alpha = .5, color = (.9, .4, 0))
            
            plt.yticks(range(-180, 541, 45), (np.arange(-180, 541, 45).astype(str) + np.array(['° (S) ',
                       '° (SW)', '° (W) ', '° (NW)','° (N) ','° (NE)', '° (E) ', '° (SE)', '° (S) ',
                       '° (SW)', '° (W) ', '° (NW)','° (N) ','° (NE)', '° (E) ', '° (SE)', '° (S) '], dtype = object)))
    

            plt.plot(date2num(datetime), ECMWF_wd.apply(lambda x: x-360 if x > ECMWF_mean + 180 else x), color = (0, .8, 1), label = 'ECMWF Rolling Mean')
            plt.plot(date2num(datetime), AWS_wd.apply(lambda x: x-360 if x > AWS_mean + 180 else x), color = (1, .85, 0, 1), label = 'AWS Rolling Mean')
  

        else:
            plt.plot(date2num(datetime), df['ECMWF Avg %s' % metric], 
                     label = 'ECMWF Data', alpha = .3, color = (0, .8, 1))
            plt.plot(date2num(datetime), df['AWS Avg %s' % metric], 
                     label = 'AWS Data', alpha = .3, color = (1, .85, 0))
            plt.plot(date2num(datetime), df['ECMWF Avg %s' % metric].rolling(56).mean(), 
                     color = (0, .4, 1), label = 'ECMWF Rolling Mean')
            plt.plot(date2num(datetime), df['AWS Avg %s' % metric].rolling(56, min_periods = 32).mean(), 
                     color = (.9, .4, 0), label = 'AWS Rolling Mean')
            plt.plot(date2num(datetime), (df['AWS Avg %s' % metric] - df['ECMWF Avg %s' % metric]).rolling(56, min_periods = 32).mean(), 
                     label = 'Difference (AWS - ECMWF)', color = (.9,0.1, 0.1))
            
        plt.title(title)
        plt.legend(loc = 2)
        plt.ylabel(metric) 

    plt.xticks(date2num(pd.date_range(start = '%s-%s-%s'% (int(np.nanmin(df['Year'])), int(np.nanmin(df['Month'])), int(np.nanmin(df['Day']))), 
                                      end = '%s-%s-%s'% (int(np.nanmax(df['Year'])), int(np.nanmax(df['Month'])), int(np.nanmax(df['Day']))), freq = 'YS')), 
                                       range(int(np.nanmin(df['Year'])), int(np.nanmax(df['Year'])+1)))
    
    plt.grid(color = (.2, .2, .2))
    plt.xlabel('Year', labelpad = 5)
    if save:
        if (metric == 'Pressure (hPa)'):
            f.savefig(os.path.join(file_path, (title+ '.pdf').replace('/', '÷')),
                      format = 'PDF', bbox_inches = 'tight')
        else:
            plt.savefig(os.path.join(file_path, (title+ '.pdf').replace('/', '÷')),
                        format = 'PDF', bbox_inches = 'tight')
        pp.savefig()
    else:
        plt.show()
    plt.close()

    
def AvE_corr(dfc, metric, file_path, pp, size = (15, 12), save = False):
    df = dfc.copy()
    year_diff = '%d-%d' % (np.nanmin(df['Year']), np.nanmax(df['Year']))
    plt.figure(figsize = size)
    df = df[~np.isnan(df['AWS Avg %s' % metric])]
    x_min = np.nanmin(df['AWS Avg %s' % metric])-5
    x_max = np.nanmax(df['AWS Avg %s' % metric])+5
    cmap_JD = LinearSegmentedColormap.from_list('mycmap', [(.93, .1, .17), (0, .7, 1),(.93, .1, .17)])
    title = 'AWS %s vs. ECMWF %s Correlation at %s Station %s' % (metric, metric, dfc.name, year_diff)
#    plt.rcParams['axes.facecolor'] = (.6, .6, .6)
                
    if (metric == 'Wind Direction'):
       plt.xticks(range(0, 361, 45), (np.arange(0, 361, 45).astype(str) + np.array(['° (N) ','° (NE)', '° (E) ',
                                      '° (SE)', '° (S) ', '° (SW)', '° (W) ', '° (NW)','° (N) '], dtype = object)))
       plt.yticks(range(0, 361, 45), (np.arange(0, 361, 45).astype(str) + np.array(['° (N) ','° (NE)', '° (E) ',
                                      '° (SE)', '° (S) ', '° (SW)', '° (W) ', '° (NW)','° (N) '], dtype = object)))
       plt.axis([x_min-(x_max-x_min)/20, x_max+(x_max-x_min)/20, x_min-(x_max-x_min)/20, x_max+(x_max-x_min)/20])
       slope, intercept, r_value, p_value, std_err = stats.linregress(df['AWS Avg %s' % metric], 
                                                                      df['AWS Avg %s' % metric] - df['Avg %s Difference' % metric])
    
    else:
        plt.axis([x_min-(x_max-x_min)/10, x_max+(x_max-x_min)/10, x_min-(x_max-x_min)/10, x_max+(x_max-x_min)/10])
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['AWS Avg %s' % metric], 
                                                                       df['ECMWF Avg %s' % metric])

    plt.plot([x_min, x_max], [x_min*slope + intercept, x_max*slope + intercept],
             label = 'LSRL, r = %s' % str(round(r_value, 3)), color = (0, 0, 0), linewidth = 2)
    plt.plot([-100, 1200], [-100, 1200], color = (.2, .2, .2), linewidth = 1)
    plt.scatter(df['AWS Avg %s' % metric], df['ECMWF Avg %s' % metric], 
                c = df['Julian Day'],cmap=cmap_JD, s= 1)
    plt.xlabel('AWS Avg %s' % (metric))
    plt.colorbar(label = 'Julian Day')
    plt.ylabel('ECMWF %s' % (metric))
    plt.title(title)
    plt.grid(color = (0.2, 0.2, 0.2))
    plt.legend(loc = 2)
    if save:
        plt.savefig(os.path.join(file_path, (title+ '.pdf').replace('/', '÷')),
                    format = 'PDF', bbox_inches = 'tight')
        pp.savefig()
    else:
        plt.show()
    plt.close()