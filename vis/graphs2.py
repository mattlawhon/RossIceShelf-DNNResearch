#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:07:02 2018

@author: Matt_Lawhon
"""


def monthly_wind_std(dfc, metric, file_path, size = (40, 10), save = False):
    
    df = dfc.copy()
    year_diff = '%d-%d' % (np.nanmin(df['Year']), np.nanmax(df['Year']))
    cmap_missing = LinearSegmentedColormap.from_list('mycmap', [(.9, 0, 0), (0, .9, 0)])
#    plt.rcParams['axes.facecolor'] = (.8, .8, .8)
    title = 'AWS Monthly Average %s with Standard Deviation at %s Station %s' % (metric, dfc.name, year_diff)
    plt.figure(figsize = (40, 10))
    
    monthly = df.groupby(['Year', 'Month'])['AWS Avg %s' % metric]
    err = monthly.std().values
    y = monthly.mean().values
    datetime = pd.to_datetime({'year': df.groupby(['Year', 'Month'])['Year'].first(), 
                                               'month': df.groupby(['Year', 'Month'])['Month'].first(),
                                               'day': 1}).values
    
    if (metric == 'Wind Direction'):
        plt.yticks(range(-90, 451, 45), (np.arange(-90, 451, 45).astype(str) + np.array(['° (W) ', '° (NW)', '° (N) ','° (NE)',
                                       '° (E) ','° (SE)', '° (S) ', '° (SW)', '° (W) ', '° (NW)','° (N) ', '° (NE)',
                                       '° (E) ',], dtype = object)))
        y =  np.degrees(np.arctan2(df.groupby(['Year', 'Month'])['AWS Avg U Wind Speed (m/s)'].mean().values,
                                   df.groupby(['Year', 'Month'])['AWS Avg V Wind Speed (m/s)'].mean().values)) + 180
        err =  np.degrees(np.arctan2(df.groupby(['Year', 'Month'])['AWS Avg U Wind Speed (m/s)'].std().values,
                                   df.groupby(['Year', 'Month'])['AWS Avg V Wind Speed (m/s)'].std().values))
        
        
        

    plt.errorbar(date2num(datetime), y, yerr = err,
                 capsize = 5, c = (0, 0, 0), ecolor = (.2, .2, .1), zorder = 2)
    plt.scatter(date2num(datetime), y, 
                c = 100 - df.groupby(['Year', 'Month'])['AWS Number of Missing %s Measurements' % metric].mean()/.36,
                cmap = cmap_missing, zorder =3, edgecolors = (0.2, .2, .2), s = 50)

    plt.xlabel('Year')
    plt.xticks(date2num(pd.date_range(start = '%s-%s-%s'% (np.nanmin(df['Year']), np.nanmin(df['Month']), np.nanmin(df['Day'])), 
                                      end = '%s-%s-%s'% (np.nanmax(df['Year']), np.nanmax(df['Month']), np.nanmax(df['Day'])), freq = 'YS')), 
                                       range(np.nanmin(df['Year']), np.nanmax(df['Year'])+1))
    plt.colorbar(label = 'Percent of Wind Data Present')
    plt.ylabel(metric)
    plt.title(title)
    
    plt.grid(color = (1, 1, 1), zorder = 1, linewidth = 2)
    if save:
        plt.savefig(os.path.join(file_path, (title+ '.pdf').replace('/', '÷')), format = 'PDF', bbox_inches = 'tight')
    else:
        plt.show()
    plt.close()


#%%
fig, axes = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (10, 10))

for ab in axes:
    for a in ab:
        a.axis([0, 24, 0, 360])
        a.grid()
        a.set_yticks(range(0, 361,30 ))
        a.set_xticks(range(0, 25,4 ))
        

axes[0][1].scatter(summer_night['AWS Avg Wind Speed'], summer_night['AWS Avg Wind Direction'], s = .7 )
axes[0][1].set(title = 'Summer Night')
axes[0][0].scatter(summer_day['AWS Avg Wind Speed'], summer_day['AWS Avg Wind Direction'], s = .7 )
axes[0][0].set(title = 'Summer Day', ylabel = 'Wind Direction' )
axes[1][1].scatter(winter_night['AWS Avg Wind Speed'], winter_night['AWS Avg Wind Direction'], s = .7 )
axes[1][1].set(title = 'Winter Night', xlabel = 'Wind Speed (m/s)' )
axes[1][0].scatter(winter_day['AWS Avg Wind Speed'], winter_day['AWS Avg Wind Direction'], s = .7 )
axes[1][0].set(title = 'Winter Day', xlabel = 'Wind Speed (m/s)', ylabel = 'Wind Direction' )

plt.show()
# wind speed v direction summer/winter day/night

#%%
fig, axes = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (10, 10))

slope, intercept, r_value, p_value, std_err = stats.linregress(summer_night['AWS Avg Wind Speed'], summer_night['ECMWF Avg Wind Speed'])
axes[0][1].scatter(summer_night['AWS Avg Wind Speed'], summer_night['ECMWF Avg Wind Speed'], s = .7, label = 'r value = %f' % r_value)
axes[0][1].set(title = 'Summer Night')
axes[0][1].plot([0, 24], [intercept, 24*slope + intercept], c = (.7, .2, .2))

slope, intercept, r_value, p_value, std_err = stats.linregress(summer_day['AWS Avg Wind Speed'], summer_day['ECMWF Avg Wind Speed'])
axes[0][0].scatter(summer_day['AWS Avg Wind Speed'], summer_day['ECMWF Avg Wind Speed'], s = .7, label = 'r value = %f' % r_value)
axes[0][0].set(title = 'Summer Day', ylabel = 'ECMWF Wind Speed (m/s)' )
axes[0][0].plot([0, 24], [intercept, 24*slope + intercept], c = (.7, .2, .2))

slope, intercept, r_value, p_value, std_err = stats.linregress(winter_night['AWS Avg Wind Speed'], winter_night['ECMWF Avg Wind Speed'])
axes[1][1].scatter(winter_night['AWS Avg Wind Speed'], winter_night['ECMWF Avg Wind Speed'], s = .7, label = 'r value = %f' % r_value)
axes[1][1].set(title = 'Winter Night', xlabel = 'AWS Wind Speed (m/s)' )
axes[1][1].plot([0, 24], [intercept, 24*slope + intercept], c = (.7, .2, .2))

slope, intercept, r_value, p_value, std_err = stats.linregress(winter_day['AWS Avg Wind Speed'], winter_day['ECMWF Avg Wind Speed'])
axes[1][0].scatter(winter_day['AWS Avg Wind Speed'], winter_day['ECMWF Avg Wind Speed'], s = .7, label = 'r value = %f' % r_value)
axes[1][0].set(title = 'Winter Day', xlabel = ' AWS Wind Speed (m/s)', ylabel = 'ECMWF Wind Speed (m/s)' )
axes[1][0].plot([0, 24], [intercept, 24*slope + intercept], c = (.7, .2, .2))

for ab in axes:
    for a in ab:
        a.axis([0, 24, 0, 24])
        a.grid()
        a.set_yticks(range(0, 25, 4))
        a.set_xticks(range(0, 25, 4))
        a.legend()
        
plt.show()
#  AWS wind speed v ECMWF wind speed summer/winter day/night
#%%
fig, axes = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (12, 10))



slope, intercept, r_value, p_value, std_err = stats.linregress(summer_night['AWS Avg Wind Direction'], summer_night['AWS Avg Wind Direction'] - summer_night['Wind Direction Difference'])
axes[0][1].scatter(summer_night['AWS Avg Wind Direction'], summer_night['ECMWF Avg Wind Direction'], s = .7, label = 'r value = %f' % r_value, c =summer_night['AWS Avg Wind Speed'] )
axes[0][1].set(title = 'Summer Night')
axes[0][1].plot([0, 360], [intercept, 360*slope + intercept], c = (.9, .2, .2))

slope, intercept, r_value, p_value, std_err = stats.linregress(summer_day['AWS Avg Wind Direction'], summer_day['AWS Avg Wind Direction'] - summer_day['Wind Direction Difference'])
axes[0][0].scatter(summer_day['AWS Avg Wind Direction'], summer_day['ECMWF Avg Wind Direction'], s = .7, label = 'r value = %f' % r_value, c =summer_day['AWS Avg Wind Speed'])
axes[0][0].set(title = 'Summer Day', ylabel = 'ECMWF Wind Direction' )
axes[0][0].plot([0, 360], [intercept, 360*slope + intercept], c = (.9, .2, .2))

slope, intercept, r_value, p_value, std_err = stats.linregress(winter_night['AWS Avg Wind Direction'], winter_night['AWS Avg Wind Direction'] - winter_night['Wind Direction Difference'])
axes[1][1].scatter(winter_night['AWS Avg Wind Direction'], winter_night['ECMWF Avg Wind Direction'], s = .7, label = 'r value = %f' % r_value, c =winter_night['AWS Avg Wind Speed'])
axes[1][1].set(title = 'Winter Night', xlabel = 'AWS Wind Direction' )
axes[1][1].plot([0, 360], [intercept, 360*slope + intercept], c = (.9, .2, .2))
slope, intercept, r_value, p_value, std_err = stats.linregress(winter_day['AWS Avg Wind Direction'], winter_day['AWS Avg Wind Direction'] - winter_day['Wind Direction Difference'])
axes[1][0].scatter(winter_day['AWS Avg Wind Direction'], winter_day['ECMWF Avg Wind Direction'], s = .7, label = 'r value = %f' % r_value, c =winter_day['AWS Avg Wind Speed'])
axes[1][0].set(title = 'Winter Day', xlabel = ' AWS Wind Direction', ylabel = 'ECMWF Wind Direction' )
axes[1][0].plot([0, 360], [intercept, 360*slope + intercept], c = (.9, .2, .2))

fig.subplots_adjust(right=0.8)
im = plt.gca().get_children()[0]
cax = fig.add_axes([0.85,0.15,0.05,0.7]) 
fig.colorbar(im, cax=cax)


for ab in axes:
    for a in ab:
        a.axis([0, 360, 0, 360])
        a.grid()
        a.set_yticks(range(0, 361, 60))
        a.set_xticks(range(0, 361, 60))
        a.legend()

cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

        
plt.show()
#%%
fig, axes = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (12, 12))

slope, intercept, r_value, p_value, std_err = stats.linregress(summer_night['AWS Avg Temp'], summer_night['ECMWF Avg Temp'])
axes[0][1].scatter(summer_night['AWS Avg Temp'], summer_night['ECMWF Avg Temp'], s = .7, label = 'r value = %f' % r_value)
axes[0][1].set(title = 'Summer Night')
axes[0][1].plot([-70, 5], [-70*slope + intercept, 5*slope + intercept], c = (.7, .2, .2))

slope, intercept, r_value, p_value, std_err = stats.linregress(summer_day['AWS Avg Temp'], summer_day['ECMWF Avg Temp'])
axes[0][0].scatter(summer_day['AWS Avg Temp'], summer_day['ECMWF Avg Temp'], s = .7, label = 'r value = %f' % r_value)
axes[0][0].set(title = 'Summer Day', ylabel = 'ECMWF Temp' )
axes[0][0].plot([-70, 5], [-70*slope + intercept, 5*slope + intercept], c = (.7, .2, .2))

slope, intercept, r_value, p_value, std_err = stats.linregress(winter_night['AWS Avg Temp'], winter_night['ECMWF Avg Temp'])
axes[1][1].scatter(winter_night['AWS Avg Temp'], winter_night['ECMWF Avg Temp'], s = .7, label = 'r value = %f' % r_value)
axes[1][1].set(title = 'Winter Night', xlabel = 'AWS Temp' )
axes[1][1].plot([-70, 5], [-70*slope + intercept, 5*slope + intercept], c = (.7, .2, .2))

slope, intercept, r_value, p_value, std_err = stats.linregress(winter_day['AWS Avg Temp'], winter_day['ECMWF Avg Temp'])
axes[1][0].scatter(winter_day['AWS Avg Temp'], winter_day['ECMWF Avg Temp'], s = .7, label = 'r value = %f' % r_value)
axes[1][0].set(title = 'Winter Day', xlabel = ' AWS Temp', ylabel = 'ECMWF Temp' )
axes[1][0].plot([-70, 5], [-70*slope + intercept, 5*slope + intercept], c = (.7, .2, .2))

for ab in axes:
    for a in ab:
        a.axis([-70, 5, -70, 5])
        a.grid()
        a.set_yticks(range(-70, 10, 5))
        a.set_xticks(range(-70, 10, 5))
        a.legend(loc = 2)
        
plt.show()
#%%
# Regression plot for wind with color for unmeasured metric
df_wind = wind_reformat(df, missing_val_cutoff = 0)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_wind['AWS Avg Wind Direction'] , df_wind['ECMWF Avg Wind Direction'] )
plt.figure(figsize = (15,10))
plt.scatter(df_wind['AWS Avg Wind Direction'] , df_wind['ECMWF Avg Wind Direction'], c = df_wind['AWS Avg Wind Speed'],  
            cmap=truncate_colormap(plt.get_cmap('plasma'), .2, 1), s= 20, label = '')
plt.plot([0, 360], [intercept, intercept + slope*360], label = 'lsrl, r = %s' % str(round(r_value, 5)), color = (.1, .1 ,.1), linewidth = 3)
plt.xlabel('AWS Avg Wind Speed')
plt.legend()
plt.colorbar(label = 'Wind Direction')
plt.ylabel('ECMWF Wind Speed')
plt.title('AWS Avg Wind Speed vs. ECMWF Wind Speed')
plt.grid()
plt.legend(loc = 2)
plt.show()



#%%
dfc = wind_reformat(df, season = 'winter')

plt.figure(figsize = (20, 10))
plt.hist(dfc[((dfc['AWS Avg Wind Speed'])== 0)]['ECMWF Avg Temp'], bins = np.arange(-70, 6, 5))
plt.grid()
plt.xlabel('AWS Temp')
plt.title('Winter ECMWF Temp while AWS measures 0 m/s')
plt.xticks(np.arange(-70, 6, 5))
plt.show()

#%%
avg_direction = [np.nanmean(summer_night['U Wind Speed (m/s)']), 
                 np.nanmean(summer_night['V Wind Speed (m/s)']),
                 np.nanmean(summer_day['U Wind Speed (m/s)']), 
                 np.nanmean(summer_day['V Wind Speed (m/s)']),
                 np.nanmean(winter_night['U Wind Speed (m/s)']), 
                 np.nanmean(winter_night['V Wind Speed (m/s)']),
                 np.nanmean(winter_day['U Wind Speed (m/s)']), 
                 np.nanmean(winter_day['V Wind Speed (m/s)'])]

print('\nPrevailing Summer Night Wind Direction = %s, \n'
      'Prevailing Summer Day Wind Direction = %s, \n'
      'Prevailing Winter Night Wind Direction = %s, \n'
      'Prevailing Winter Day Wind Direction = %s. \n'
      % (np.round(np.degrees(np.arctan2(avg_direction[0],avg_direction[1]) + np.pi), decimals = 2),
        np.round(np.degrees(np.arctan2(avg_direction[2],avg_direction[3]) + np.pi), decimals = 2),
        np.round(np.degrees(np.arctan2(avg_direction[4],avg_direction[5]) + np.pi), decimals = 2),
        np.round(np.degrees(np.arctan2(avg_direction[6],avg_direction[7]) + np.pi), decimals = 2)))
#%%
df_wind = wind_reformat(df, missing_val_cutoff = 26)
#df_wind = AWS_10min_df.copy()
#df_wind = df_wind[((df_wind['Wind Speed (m/s)']>3))]# | (~(df_wind['Wind Direction']%180 == 0)))]
Jan = df_wind[df_wind['Julian Day']<=31]
Feb = df_wind[(df_wind['Julian Day']>31)&(df_wind['Julian Day']<=59)]
Mar = df_wind[(df_wind['Julian Day']>59)&(df_wind['Julian Day']<=90)]
Apr = df_wind[(df_wind['Julian Day']>90)&(df_wind['Julian Day']<=120)]
May = df_wind[(df_wind['Julian Day']>120)&(df_wind['Julian Day']<=151)]
Jun = df_wind[(df_wind['Julian Day']>151)&(df_wind['Julian Day']<=181)]
Jul = df_wind[(df_wind['Julian Day']>181)&(df_wind['Julian Day']<=212)]
Aug = df_wind[(df_wind['Julian Day']>212)&(df_wind['Julian Day']<=243)]
Sep = df_wind[(df_wind['Julian Day']>243)&(df_wind['Julian Day']<=273)]
Oct = df_wind[(df_wind['Julian Day']>273)&(df_wind['Julian Day']<=304)]
Nov = df_wind[(df_wind['Julian Day']>304)&(df_wind['Julian Day']<=334)]
Dec = df_wind[df_wind['Julian Day']>334]

Annual = [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec]

#%%
print('\nJan Avg Wind Speed = %f m/s, std = %f'
      '\nFeb Avg Wind Speed = %f m/s, std = %f'
      '\nMar Avg Wind Speed = %f m/s, std = %f'
      '\nApr Avg Wind Speed = %f m/s, std = %f'
      '\nMay Avg Wind Speed = %f m/s, std = %f'
      '\nJun Avg Wind Speed = %f m/s, std = %f'
      '\nJul Avg Wind Speed = %f m/s, std = %f'
      '\nAug Avg Wind Speed = %f m/s, std = %f'
      '\nSep Avg Wind Speed = %f m/s, std = %f'
      '\nOct Avg Wind Speed = %f m/s, std = %f'
      '\nNov Avg Wind Speed = %f m/s, std = %f'
      '\nDec Avg Wind Speed = %f m/s, std = %f'
      % (np.nanmean(Jan['Wind Speed (m/s)']), np.nanstd(Jan['Wind Speed (m/s)']),
         np.nanmean(Feb['Wind Speed (m/s)']), np.nanstd(Feb['Wind Speed (m/s)']),
         np.nanmean(Mar['Wind Speed (m/s)']), np.nanstd(Mar['Wind Speed (m/s)']),
         np.nanmean(Apr['Wind Speed (m/s)']), np.nanstd(Apr['Wind Speed (m/s)']),
         np.nanmean(May['Wind Speed (m/s)']), np.nanstd(May['Wind Speed (m/s)']),
         np.nanmean(Jun['Wind Speed (m/s)']), np.nanstd(Jun['Wind Speed (m/s)']),
         np.nanmean(Jul['Wind Speed (m/s)']), np.nanstd(Jul['Wind Speed (m/s)']),
         np.nanmean(Aug['Wind Speed (m/s)']), np.nanstd(Aug['Wind Speed (m/s)']),
         np.nanmean(Sep['Wind Speed (m/s)']), np.nanstd(Sep['Wind Speed (m/s)']),
         np.nanmean(Oct['Wind Speed (m/s)']), np.nanstd(Oct['Wind Speed (m/s)']),
         np.nanmean(Nov['Wind Speed (m/s)']), np.nanstd(Nov['Wind Speed (m/s)']),
         np.nanmean(Dec['Wind Speed (m/s)']), np.nanstd(Dec['Wind Speed (m/s)'])))

#%%
for i in Annual:
    plt.hist(i[~np.isnan(i['AWS Avg Wind Speed'])]['ECMWF Avg Wind Speed'], bins = range(0, 15))
    plt.title('AWS Avg Wind Speed')
    plt.show()
#%%
print('\nJan Avg Wind Speed = %f m/s, std = %f'
      '\nFeb Avg Wind Speed = %f m/s, std = %f'
      '\nMar Avg Wind Speed = %f m/s, std = %f'
      '\nApr Avg Wind Speed = %f m/s, std = %f'
      '\nMay Avg Wind Speed = %f m/s, std = %f'
      '\nJun Avg Wind Speed = %f m/s, std = %f'
      '\nJul Avg Wind Speed = %f m/s, std = %f'
      '\nAug Avg Wind Speed = %f m/s, std = %f'
      '\nSep Avg Wind Speed = %f m/s, std = %f'
      '\nOct Avg Wind Speed = %f m/s, std = %f'
      '\nNov Avg Wind Speed = %f m/s, std = %f'
      '\nDec Avg Wind Speed = %f m/s, std = %f'
      % (np.nanmean(Jan['AWS Avg Wind Speed']), np.nanstd(Jan['AWS Avg Wind Speed']),
         np.nanmean(Feb['AWS Avg Wind Speed']), np.nanstd(Feb['AWS Avg Wind Speed']),
         np.nanmean(Mar['AWS Avg Wind Speed']), np.nanstd(Mar['AWS Avg Wind Speed']),
         np.nanmean(Apr['AWS Avg Wind Speed']), np.nanstd(Apr['AWS Avg Wind Speed']),
         np.nanmean(May['AWS Avg Wind Speed']), np.nanstd(May['AWS Avg Wind Speed']),
         np.nanmean(Jun['AWS Avg Wind Speed']), np.nanstd(Jun['AWS Avg Wind Speed']),
         np.nanmean(Jul['AWS Avg Wind Speed']), np.nanstd(Jul['AWS Avg Wind Speed']),
         np.nanmean(Aug['AWS Avg Wind Speed']), np.nanstd(Aug['AWS Avg Wind Speed']),
         np.nanmean(Sep['AWS Avg Wind Speed']), np.nanstd(Sep['AWS Avg Wind Speed']),
         np.nanmean(Oct['AWS Avg Wind Speed']), np.nanstd(Oct['AWS Avg Wind Speed']),
         np.nanmean(Nov['AWS Avg Wind Speed']), np.nanstd(Nov['AWS Avg Wind Speed']),
         np.nanmean(Dec['AWS Avg Wind Speed']), np.nanstd(Dec['AWS Avg Wind Speed'])))


d
#%%

dfc = df.copy()
dfc['AWS Avg Wind Speed'] = dfc['AWS Avg Wind Speed'].apply(lambda x: np.nan if x == 0 else x)
print(dfc.columns)
plt.figure(figsize = (40, 10))
plt.errorbar(time_frame(dfc['Year'], dfc['Julian Day']), dfc['AWS Avg Wind Speed'], yerr = dfc['std'], capsize = 5, ecolor = (.8, .3, .3), marker = 'o')
plt.plot([0, 6000], [0, 0], c = (0, 0, 0), linewidth = 1)
plt.axis([0, 6000, -3, 20])
plt.xticks(np.arange(0, 6000, 365.25), range(2001, 2018))
plt.yticks(range(-2, 20, 1))
plt.xlabel('Year')
plt.ylabel('Wind Speed (m/s)')
plt.title('AWS Monthly Average Wind Speeds')
plt.grid(True, which = 'both')
plt.show()



