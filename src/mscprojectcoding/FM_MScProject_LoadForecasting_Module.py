#!/usr/bin/env python
# coding: utf-8

# module containing my frequently used functions


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# override some default print parameters
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth',-1)

#np.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=2000)
np.set_printoptions(edgeitems=10)


# general functions

# change num lines to print of a numpy array without changing default
def fullprint(*args, **kwargs):
  from pprint import pprint
  import numpy as np
  opt = np.get_printoptions()
  np.set_printoptions(threshold=np.inf)
  pprint(*args, **kwargs)
  np.set_printoptions(**opt)

def pltDefaults(small,med,large):
    import matplotlib.pyplot as plt

    SMALL_SIZE = small  # eg 8
    MEDIUM_SIZE = med  # eg 10
    BIGGER_SIZE = large # eg 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# track memory usage
# http://practicalpython.blogspot.com/2017/03/monitoring-memory-usage-in-jupyter.html
def show_mem_usage():
    '''Displays memory usage from inspection
    of global variables in this notebook'''
    gl = sys._getframe(1).f_globals
    vars= {}
    for k,v in list(gl.items()):
        # for pandas dataframes
        if hasattr(v, 'memory_usage'):
            mem = v.memory_usage(deep=True)
            if not np.isscalar(mem):
                mem = mem.sum()
            vars.setdefault(id(v),[mem]).append(k)
        # work around for a bug
        elif isinstance(v,pd.Panel):
            v = v.values
        vars.setdefault(id(v),[sys.getsizeof(v)]).append(k)
    total = 0
    for k,(value,*names) in vars.items():
        if value>1e6:
            print(names,"%.3fMB"%(value/1e6))
        total += value
    print("%.3fMB"%(total/1e6))
    
# get NaN stats of DataFrame by column
def get_NaN_stats(df):
    list_NaN_stats = list(df.isnull().sum(axis=0).items())  # count of null/NaN/etc by column
    #print(df[df.isnull().any(axis=1)])   # print rows containing NaNs
    return df[df.isnull().any(axis=1)]
 
# get unique userID (same meterID, tarrif, Acorn group etc)
def get_uniques(df_in,cols):
    df_out = pd.DataFrame()
    uniques = df_in.drop_duplicates(subset=cols)   # takes 1st line of each new block of rows where a new subset
    df_out= pd.concat([df_out,uniques])   
    return df_out

# search to find ordinal row number of data series for specified uID(s)
# arg1 = uID : list of users from
# arg2 = series of (all) user_id
# return row number
def getRowN_of_uID(uID,all_uIDs):
    rowN=[0]*len(uID)
    cnt1=0
    for i in uID:
        cnt1 += 1
        cnt2=0
        for j in all_uIDs:
            cnt2 += 1
            if j == i:
                print("Row# for uID",i,"is : ",cnt2-1)
                rowN[cnt1-1]=cnt2-1
                break
    print("row numbers of uID: ",rowN)
    return(rowN)


# get range of data for each groupby block
def get_data_range(df,groupby_col,data_col):
    df_col_range = pd.DataFrame()
    min_val = df.groupby(groupby_col)[data_col].min()
    max_val = df.groupby(groupby_col)[data_col].max() 
    delta = max_val - min_val
    tmp  = pd.concat([min_val, max_val], axis=1,ignore_index = False)
    df_col_range = pd.concat([df_col_range,tmp], axis=0,ignore_index = False)
    return df_col_range

#get piecewise linear function as numpy array
def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*(x-x0)+y0, lambda x:k2*(x-x0)+y0])  # lambda function

# function to round UP to an integer - can pass negative decimals to round by > multiple of 1, 10 etc
def round_up(n, decimals=0):
    import math
    multiplier = 10 ** decimals 
    return math.ceil(n * multiplier) / multiplier


# function to round DOWN to an integer - can pass negative decimals to round by > multiple of 1, 10 etc
def round_down(n, decimals=0):
    import math
    multiplier = 10 ** decimals 
    return math.floor(n * multiplier) / multiplier

# set color map from colmap name and value range
def setColMap(colmapName,minV,maxV):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors2
    import matplotlib.cm as cmx

    cmOcc = plt.cm.get_cmap(colmapName)                      # get a colour bar object
    cNorm = colors2.Normalize(minV, maxV)      # map minVal,maxVal range to cNorm 0-1 colour range indices
    scalarMap3 = cmx.ScalarMappable(norm=cNorm, cmap=cmOcc)  # match cNorm normalised data (hrs) to selected colour bar
    scalarMap3._A = []  
    return(cmOcc,cNorm,scalarMap3)  

# function to get max/min for axis limits
def get_limits(data, decimals=0):
    import math
    
    if(isinstance(data,pd.Series)):
        data = data.values  
    
    multiplier = 10 ** decimals 
    return (int(math.floor(data.min() * multiplier)/multiplier),int(math.ceil(data.max()* multiplier)/multiplier))

# function to find all values of a dictionary key where nested lists and dictionaries

def findkeys(node, kv):
    if isinstance(node, type([])):
        for i in node:
            for x in findkeys(i, kv):
                yield x
    elif isinstance(node, dict):
        if kv in node:
            yield node[kv]
        for j in node.values():
            for x in findkeys(j, kv):
                yield x
                
# funtion to flatten a nested list with up to 3 levels of nesting allowed.
   
def removeNestings(l): 
    output = []
    for i in l: 
        if type(i) == list: 
            for j in i:
                if type(j) == list:
                    for k in j:
                        if type(k) == list:
                            print("more than 3 nest levels not allowed \n")
                            break
                        else:
                            output.append(k) 
                else:
                    output.append(j) 
        else: 
            output.append(i) 
    return(output)

# function to return max,min of numpy array - option to get for each column or row or Total
def get_NPminmax_values(data,**kwargs):
    if 'ax' in kwargs:
        ax = kwargs.get('ax')
        ax=int(ax)
    else:
        ax = None
        
    min_=np.amin(data,axis=ax)
    max_=np.amax(data,axis=ax)
    
    return(min_,max_)


#  plot load (eg aggregate) np array by weekday/sat/sun and hr of day

def PlotLoadByDaysOfWeek(load, YaxisLabel, yLim):
    fig = plt.figure(figsize=(12, 3))
    axs = []
    for j in range(3):
        axs.append(fig.add_subplot(1, 3, j + 1))  # add 1*3 subplot array to figure

    start = 0
    end = 24
    weekdays, saturdays, sundays = [], [], []
    for j in range(365):
        dayConsumption = load[start:end]  # iterate through DOY
        # get the day of week
        day_of_week = dayConsumption.index[0]
        day_of_week = day_of_week.dayofweek
        if day_of_week in [0, 1, 2, 3, 4]:
            axs[0].scatter(np.arange(1, 25), dayConsumption, color='k', alpha=0.1)  # plot on left
            weekdays.append(dayConsumption)  # append the 24 hourly values for day j if weekday
            axs[0].set_ylabel(YaxisLabel)
            axs[0].set_title('Weekdays')
        elif day_of_week in [5]:
            axs[1].scatter(np.arange(1, 25), dayConsumption, color='g', alpha=0.1)  # plot on centre
            saturdays.append(dayConsumption)
            axs[1].set_title('Saturdays')
        else:
            axs[2].scatter(np.arange(1, 25), dayConsumption, color='b', alpha=0.1)  # plot on right
            sundays.append(dayConsumption)
            axs[2].set_title('Sundays')
        start += 24
        end += 24  # day iterator
    for ax in axs:
        ax.set_xlim([1, 24])
        ax.set_ylim([0, yLim])
        ax.set_xlabel('hr')

    axs[0].scatter(np.arange(1, 25), np.mean(weekdays, axis=0),
                   color='r')  # plot mean of weekday consumption at each hr
    axs[1].scatter(np.arange(1, 25), np.mean(saturdays, axis=0), color='r')
    axs[2].scatter(np.arange(1, 25), np.mean(sundays, axis=0), color='r')
    return (fig)


def plotSectionOfLoadProf(load_df, startIndex, nDays, labelTitle, smplsPerDay, *args, **kwargs):
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator, MaxNLocator)
    import matplotlib.lines as mlines
    import matplotlib.ticker as ticker  # import the full shebang

    if (isinstance(load_df, pd.Series)):
        load_df = pd.DataFrame(load_df)
        print(load_df.shape)
    timePlot = np.arange(1, load_df.shape[0] + 1)  # index series for x axis

    if 'subPlotSize' in kwargs:
        subPlotSize = kwargs.get('subPlotSize')
    else:
        subPlotSize = (12, 4)

    fig = plt.figure(figsize=(subPlotSize[0], subPlotSize[1]))
    axs = []

    if 'myStyles' in kwargs:
        myStyles = kwargs.get('myStyles')
        lineCols = myStyles['lineColors']
        lineStyles = myStyles['lineStyles']
        if 'alpha' in myStyles.keys():
            alpha = myStyles['alpha']
        else:
            alpha = [1, 1, 1, 1, 1, 1]
    else:
        lineCols = ['k', 'r', 'g', 'c', 'm', 'y']
        lineStyles = ['solid', 'solid', 'solid', 'solid']
        alpha = [1, 1, 1, 1, 1, 1]

    lw_load = 2

    # select  window indices
    window = [startIndex,
              startIndex + (smplsPerDay * nDays)]  # start and end index on load array eg 1hr or 30min indices
    dayLabels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']  # for axis labelling

    indexVals = load_df.index  # index = hr in datetime64ns format : generally 1st value will be at midnight
    proxys = []  # for legend

    # find the first tick  (will be at start of window usually)
    for j, ind in enumerate(indexVals):  # counter j, index ind  (tuple)
        if j >= window[0]:
            if ind.hour == 0:
                startTick = 0 + j  # find first midnight in range
                print("index data:{0} hr:{1},startTick:{3},Window:{2},Length(hrs) {4}".format \
                          (ind, ind.hour, window, startTick, (window[1] - window[0]) / (smplsPerDay / 24)))
                break
        else:
            startTick = 0

    axs.append(fig.add_subplot(1, 1, 1))

    # plot load profiles in load_df columns
    for colName in load_df.columns:
        plotVals = load_df[colName].values
        idx = load_df.columns.to_list().index(colName)
        l, = axs[0].plot(timePlot, plotVals, color=lineCols[idx % len(lineCols)],
                         linestyle=lineStyles[idx % len(lineStyles)], \
                         lw=lw_load, alpha=alpha[
                idx % len(alpha)])  # returns list of lines representing the plotted data; ',' turns off warning
        proxys.append(l)

    # allow for plotting second load df eg aggregate on 2nd y axis
    if 'load_df2' in kwargs:
        load_df2 = kwargs.get('load_df2')
        axs1 = axs[0].twinx()
        for colName2 in load_df2.columns:
            plotVals2 = load_df2[colName2].values
            idx2 = load_df2.columns.to_list().index(colName2)
            l, = axs1.plot(timePlot, plotVals2, color='k', linestyle='solid', \
                           lw=lw_load)  # returns list of lines representing the plotted data; ',' turns off warning
            proxys.append(l)
            axs1.set_ylabel('Agg Load (kWh)')

    axs[0].set_title(labelTitle)
    axs[0].set_xlim(window)

    # set tick mark positions and spacing, labels
    axs[0].set_xticks(np.arange(startTick + int(smplsPerDay / 2), window[1] + int(smplsPerDay / 2),
                                smplsPerDay))  # major ticks (by default)
    axs[0].set_xticks(np.arange(startTick, window[1], smplsPerDay), minor=True)
    tickLabs = []
    for ti in np.arange(startTick, window[1], smplsPerDay):
        tickLabs.append(str(indexVals[ti].dayofyear) + ' ' + str(dayLabels[indexVals[ti].dayofweek]))
        # tickLabs.append(str(dayLabels[indexVals[ti].dayofweek]) + ',' + str(indexVals[ti].dayofyear/(smplsPerDay / 24)))
        # tickLabs.append(indexVals[window[0]+ti].dayofyear)
        # print(ti,[indexVals[window[0]+ti].dayofweek])
    axs[0].xaxis.set_minor_locator(
        ticker.MultipleLocator(12))  # minor tick location every 12 hrs (but not shown or labelled)
    axs[0].grid(b=True, which='major', axis='x', alpha=0.3)  # keep v faint grid at 12 noon where ticklabels are
    axs[0].grid(b=True, which='minor', axis='x', alpha=0.9)  # want prominent grid lines at midnight

    axs[0].set_xticklabels(tickLabs)

    if 'xLabel' in args:
        axs[0].set_xlabel('Day of Year              Day of Week', fontsize=14, labelpad=-10)

    # axs[0].set_xlabel('Day Of Week, Day of Year')
    axs[0].set_ylabel('Load (kWh)')

    if 'yLim' not in kwargs:
        yLim = [0.0, 1.0]
    else:
        yLim = kwargs.get('yLim')
    axs[0].set_ylim(yLim)

    # add legend
    legLabs = []
    for colName in load_df.columns.values:
        legLabs.append(colName)
    ncol = load_df.shape[1]

    if 'load_df2' in kwargs:
        for colName in load_df2.columns.values:
            legLabs.append(colName)
        ncol += load_df2.shape[1]

    plt.legend(proxys, legLabs, ncol=ncol, handletextpad=0.2)  # , fontsize=10

    fig.tight_layout()
    return (fig, axs);


#  following functions for persistence models  

# standard naive persistance model scatter plot of hourly load data
# pdLoadSeries has datetime index

def persistScatterPlot(pdLoadSeries,smplsPerDay,nDays,**kwargs):         
    T_to_F = [1,2,3,4]   

    fig = plt.figure(figsize=(12,4.5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # loop through the days
    for day in range(nDays-1): 
        startIndex = (day+1)*smplsPerDay               
        dayOfWeek = pdLoadSeries.index[startIndex].dayofweek

        if dayOfWeek in T_to_F:               # plot load (Mon to Thu) v load(Tue to Fri) ?
            ax1.scatter(pdLoadSeries[startIndex-smplsPerDay:startIndex],                        pdLoadSeries[startIndex:startIndex+smplsPerDay], color='k')
        else:                                 # plot load (Mon to Thu) v load(Sat to Mon) ?
            ax2.scatter(pdLoadSeries[startIndex-smplsPerDay:startIndex],                        pdLoadSeries[startIndex:startIndex+smplsPerDay], color='g')
            if day>6:
                ax3.scatter(pdLoadSeries[startIndex-(smplsPerDay*7):startIndex-(smplsPerDay*6)],                         pdLoadSeries[startIndex:startIndex+smplsPerDay], color='b')               
                
    ax1.set_title('Load Tues to Fri - 24h lag')
    ax2.set_title('Load Sat to Mon - 24h lag')
    ax3.set_title('Load Sat to Mon - 168h lag')
    ax1.set_xlabel('load (kW)'), ax1.set_ylabel('load (kW)')
    ax2.set_xlabel('load (kW)'), ax2.set_ylabel('load (kW)')
    ax3.set_xlabel('load (kW)'), ax3.set_ylabel('load (kW)')

    SupTitle = kwargs.get('SupTitle', None)
    SaveFigTitle = kwargs.get('SaveFigTitle', None)

    fig.suptitle('Persistance Models: '+str(SupTitle))

    fig.tight_layout()
    fig.subplots_adjust(top=0.83)
    
    fname = str(SaveFigTitle)+'.png'
    fig.savefig(fname, dpi=300, format='png',  bbox_inches='tight')


# Generate persistance model plot where input : Day Of Week pre-spliced xs,ys
# i/p data in list format?
# kwargs:
# supTitle
# uID
# SaveFigTitle

def peristModelPlot(lDataTtoF, lDataStoM, lDataStoM_7d, **kwargs):
    fig = plt.figure(figsize=(12, 4.5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.scatter(lDataTtoF[0, :], lDataTtoF[1, :], color='k', alpha=0.2)
    ax2.scatter(lDataStoM[0, :], lDataStoM[1, :], color='g', alpha=0.2)
    ax3.scatter(lDataStoM_7d[0, :], lDataStoM_7d[1, :], color='b', alpha=0.2)

    ax1.set_title('Load Tues to Fri - 24h lag')
    ax2.set_title('Load Sat to Mon - 24h lag')
    ax3.set_title('Load Sat to Mon - 168h lag')
    ax1.set_xlabel('load (kW)'), ax1.set_ylabel('load (kW)')
    ax2.set_xlabel('load (kW)')  # , ax2.set_ylabel('load (kW)')
    ax3.set_xlabel('load (kW)')  # , ax3.set_ylabel('load (kW)')

    if 'supTitle' in kwargs:
        supTitle = kwargs.get('supTitle')

        if supTitle == 'uID':
            uID = kwargs.get('uID')
            fig.suptitle('Persistance Models: uID:' + str(uID))
        else:
            fig.suptitle('Persistance Models: ' + str(supTitle))

    fig.tight_layout()
    fig.subplots_adjust(top=0.83)

    if 'SaveFigTitle' in kwargs:
        SaveFigTitle = kwargs.get('SaveFigTitle', None)

        if not SaveFigTitle == 'None':
            fname = str(SaveFigTitle) + '.png'
            fig.savefig(fname, dpi=300, format='png', bbox_inches='tight')

    return (fig, ax1, ax2, ax3)


# Splice out a) xs series b) ys series from load data series (aggregate or uID) and create 2 row
# output data array for input to persistance model plots

# use list append method instead of declaring np array and filling as previous - avoids need to get rid of 0s
# this version is generalised so can use for weekdays [T to Fri]/ weekends[Sat to Mon] and
# differnt lag amounts (t-24,t-168, etc)


def persistDataArrayVersion2(pdLoadSeries, smplsPerDay, nDays, dayClass=[1, 2, 3, 4], lagD=1, refDay=0):
    xs = []
    ys = []
    np.set_printoptions(threshold=200)

    for day in range(nDays - 2):  # (day of startIndex) is day+1 hence range(nDays-2);day=lagged;day+1=unlagged;
        startIndex = (day + 1) * smplsPerDay
        dayOfWeek = pdLoadSeries.index[startIndex].dayofweek  # day of the week with Monday=0

        if ((dayOfWeek in dayClass) & (day >= refDay)):
            xs.append(pdLoadSeries[startIndex - (smplsPerDay * lagD): \
                                   startIndex - (smplsPerDay * (lagD - 1))].to_list())  # lagged (1 day earlier)
            ys.append(pdLoadSeries[startIndex:startIndex + smplsPerDay].to_list())  # un-lagged (present)

    a = np.array(xs).reshape([1, len(xs) * smplsPerDay])[0]
    b = np.array(ys).reshape([1, len(ys) * smplsPerDay])[0]

    loadDataArray = np.zeros([2, len(xs) * smplsPerDay])
    loadDataArray = np.vstack((a, b))

    return (loadDataArray)


# This function supercedes persistDataArrayV2 above : set window half width = 0 for default case.

# inputs:
# load series, smplsPerDay, nDays, winbdowHalfWidth (smpls)
# dayClass, lagD and refDay required to splice by weekday v weekend, t-24 v t-168 models etc

# output:
# load data array where row1 = xs, row2 = ys

def persistDataWindowed(loadSeries,smplsPerDay=24,nDays=365,windowHW=1,dayClass=[1,2,3,4],lagD=1,refDay=0): 
    dIdx=[]
    for i in range(windowHW,(windowHW*-1)-1,-1):
        dIdx.append(i)
   
    xs=[]
    ys=[]
    
    
    for day in range(1,nDays-2):          # start at 1 not 0 to prevent searches before start of day0 ;  
        startIndex = (day+1)*smplsPerDay               
        dayOfWeek = loadSeries.index[startIndex].dayofweek   #day of the week with Monday=0
      
        if ((dayOfWeek in dayClass) & (day >= refDay)): 
            x=np.zeros(smplsPerDay)
            y=np.zeros(smplsPerDay)

            for i in range(smplsPerDay):   # loop through hrs of day; compute delta(i) = load(t) - (load(t-(25,24,23)) etc
                delta=[]

                for cnt,k in enumerate(dIdx):          # loop through window elements
                    delta.append(loadSeries[startIndex+i]-loadSeries[startIndex-(smplsPerDay*lagD)-k+i])                      

                delta =  [abs(ele) for ele in delta]
                xIdxMin = np.argmin(delta)     # find index of min delta value
                x[i]=loadSeries[startIndex-(smplsPerDay*lagD)+i+(xIdxMin-windowHW)]  
                y[i]=loadSeries[startIndex+i]

            #print("startIndex: ",startIndex,"day: ",day,"dayOfWeek: ",dayOfWeek)
            #print("x:",x)
            #print("y:",y)


            xs.append(x)   # append data for day
            ys.append(y)    
            #print("xs:",xs)
            #print("ys:",ys)

    a=np.array(xs).reshape([1,len(xs)*smplsPerDay])[0]
    b=np.array(ys).reshape([1,len(ys)*smplsPerDay])[0]

    loadDataArray = np.zeros([2,len(xs)*smplsPerDay])  
    loadDataArray = np.vstack((a,b))
   
        
    return(loadDataArray)


# Function to calculate linear
# regression and R2
# using Sklearn to calculate r2 - much faster than other methods tested

# input xs,ys series , returns r2 and mean sq error
# kwargs :
# print (flag to print outputs),
# plot (flag to plot output regression and data),
# colour,
# figure obj name, can define figure beforehand - if wish to append subplot to predefined fig
# inputAx  : existing single axes objcet eg subplot to use
# subplotPos, subplotTitle
#
def lrmodel_r2(xs, ys, **kwargs):
    from sklearn import linear_model
    from sklearn import metrics
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error

    # need to reshape features and targets to 2D array for linear_model
    xs = xs.reshape(-1, 1)
    ys = ys.reshape(-1, 1)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(xs, ys)

    # Make predictions using the testing set
    ys_pred = regr.predict(xs)

    r2 = r2_score(ys, ys_pred)
    mse = mean_squared_error(ys, ys_pred)

    if (kwargs.get('print') == 'y'):
        print('Coefficients: ', regr.coef_)
        print('Mean squared error: ', mse)
        print('Coefficient of determination: ', r2)

    if (kwargs.get('plot') == 'y'):
        cl = kwargs.get('cl')
        if (kwargs.get('inputFigObject') == 'None'):
            fig = plt.figure(figsize=(5, 4))
            ax1 = fig.add_subplot(111)
        else:
            fig = kwargs.get('inputFigObject')
            subplotPos = kwargs.get('subplotPos', 1)
            # print("Input fig object : subplot to be created at position ",subplotPos)

            if 'inputAx' in kwargs:
                # print("input axs object required")
                ax1 = kwargs.get('inputAx')
            else:
                # print("creating new subplot axes")
                ax1 = fig.add_subplot(1, 3, subplotPos)

        ax1.scatter(xs, ys, color=cl, edgecolor=cl, alpha=0.3, facecolor='None')
        ax1.plot(xs, ys_pred, color=cl, linewidth=5)
        ax1.annotate('R2 value:' + str(round(r2, 4)), xy=(0.05, 0.90), xycoords='axes fraction', fontsize=15)

        subplotTitle = kwargs.get('subplotTitle', '')
        ax1.set_title(str(subplotTitle))
        ax1.set_xlabel('load (kW)'), ax1.set_ylabel('load (kW)')

    return (r2, mse, ys_pred)


# general wrapper to use lrmodel_r2 function
def r2_get(loadDataArray,**kwargs):
    xs=loadDataArray[0]
    ys=loadDataArray[1]
    r2,mse,yPred = lrmodel_r2(xs,ys,**kwargs)
    return(r2)           # r2 = float number


# Function to CALC and ADD R2 regression LINE to existing subplot
# using Sklearn to calculate r2 - much faster than other methods tested
# inputs
# input xs,ys series
# colour,
# figure obj name, can define figure beforehand - if wish to append subplot to predefined fig
# inputAx  : existing single axes objcet eg subplot to use

def r2RegPlot(xs, ys, inputFigObj,inputAx,cl):
    from sklearn import linear_model
    from sklearn import metrics
    from sklearn.metrics import r2_score

    # need to reshape features and targets to 2D array for linear_model
    xs = xs.reshape(-1, 1)
    ys = ys.reshape(-1, 1)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(xs, ys)

    # Make predictions using the testing set
    ys_pred = regr.predict(xs)
    r2 = r2_score(ys, ys_pred)


    inputAx.plot(xs, ys_pred, color=cl, linewidth=4)
    inputAx.annotate('R2 value:' + str(round(r2, 4)), xy=(0.05, 0.90), xycoords='axes fraction', fontsize=15)

    return(inputAx)

# function to plot histograms of r2 values on same figure where input is:
# dataframe with r2 data for each of the 3 persistance models 
# (now generalised for multiple window widths) i.e. (#windows * 3) columns
# 1. TtoF (t   v   t-(24  +/- window half width)), 
# 2. StoM (t   v   t-(24  +/- window half width)), 
# 3. StoM (t   v   t-(168  +/- window half width)) 

def r2_histogram(r2df,**kwargs):  #kwargs =  fig name, plot_title
    fig = plt.figure(figsize=(25,15))
    ax1 = fig.add_subplot(111)
    
    pltDefaults(10,15,20)

    names=np.array(r2df.columns).reshape(-1,3)

    r2Bins = np.arange(0,1.05,0.05) 
    r2Vals = r2df[names[0][0]].values
    hist,edges = np.histogram(r2Vals, r2Bins, density=False) 
    widths = edges[1:]-edges[0:-1]  # bin edges
    bin_centres = 0.5*(edges[1:]+edges[:-1])

    colors=['k','b','g','c','m','y']         
    lineStyles=['solid','dashed','dashdot','dotted']   
    r2LabelsL=['Tue to Fri t24','Sat to Mon t24','Sat to Mon t168']
    r2LabelsS=['TtoF24_','StoM24_','StoM168_']
    windows=[0,1,2,3,4,5,6,12]
    
    cnt=0
    for name in names: 
        lSt=lineStyles[cnt % 4]        # set line style

        if cnt in [0,1,2,3]:
            cls=colors[0:3]
        else:
            cls=colors[3:6]
            
        for j in range(3):
            if (j==0):
                r2Vals = r2df[name[j]].values; cl=cls[j]; r2Label=r2LabelsS[j]+str(windows[cnt])
            elif (j==1):
                r2Vals = r2df[name[j]].values; cl=cls[j]; r2Label=r2LabelsS[j]+str(windows[cnt])
            else: 
                r2Vals = r2df[name[j]].values; cl=cls[j]; r2Label=r2LabelsS[j]+str(windows[cnt])

            hist,edges = np.histogram(r2Vals, r2Bins, density=False)  #data, bins, False=>Hist contain #samples in each bin.
            ax1.scatter(bin_centres,hist,color=cl,edgecolor=cl, alpha = 0.3, facecolor= 'None') 

            ax1.plot(bin_centres,hist,label=r2Label,color=cl,linestyle=lSt,lw=2)
            
            if (cnt<1):
                ax1.bar(edges[0:-1]+widths/2,hist,width=widths,edgecolor='k', alpha = 0.3, facecolor= 'None')
        cnt+=1

    ax1.set_xlabel('R2 Value')
    ax1.set_ylabel('# of users')
    ax1.set_ylim([0,200])
    ax1.set_xlim([0,1.01])
    ax1.set_xticks(np.arange(0,1.01,0.1))
    ax1.tick_params(labeltop=False, labelright=True)
    
    pltTitle = kwargs.get('pltTitle')
    ax1.set_title('R2 Values for Hourly Data Persistance Models: '+str(pltTitle),fontsize=20)
    ax1.grid(which='both')

    plt.legend(ncol=len(names),fontsize=16)
    
  
    SaveFigTitle = kwargs.get('SaveFigTitle', None)

    if not SaveFigTitle == 'None':
        fname = str(SaveFigTitle)+'.png'
        fig.savefig(fname, dpi=300, format='png',  bbox_inches='tight')


# similar to r2_histogram BUT only plots a few histogram profiles (and greys out the rest)
# function to plot histograms of r2 values on same figure where input is:
# dataframe with r2 data for each of the 3 persistance models
# (now generalised for multiple window widths) i.e. (#windows * 3) columns
# 1. TtoF (t   v   t-(24  +/- window half width)),
# 2. StoM (t   v   t-(24  +/- window half width)),
# 3. StoM (t   v   t-(168  +/- window half width))

def r2_histogramV2(r2df,**kwargs):  #kwargs =  fig name, plot_title
    fig = plt.figure(figsize=(15,6))
    ax1 = fig.add_subplot(111)

    pltDefaults(10,15,20)

    names=np.array(r2df.columns).reshape(-1,3)

    r2Bins = np.arange(0,1.05,0.05)
    r2Vals = r2df[names[0][0]].values
    hist,edges = np.histogram(r2Vals, r2Bins, density=False)
    widths = edges[1:]-edges[0:-1]  # bin edges
    bin_centres = 0.5*(edges[1:]+edges[:-1])

    colors=['r','b','g','k','k','k','r','b','g']          # colors[0:2] for winodws 0 to 3; colors[3:5] for windows 4 to 12
    lineStyles=['solid','dashed','dashdot','dotted']      # windows [0,1,2,3],[4,5,6,12]
    r2LabelsL=['TueToFri_t24_W','SatToMon_t24_W','SatToMon_t168_W']
    r2LabelsS=['TtoF24_','StoM24_','StoM168_']
    #r2LabelsS=['Tue_to_Fri24_','Sun_to_Mon24_','Sun_to_Mon168_']
    windows=[0,1,2,3,4,5,6,12]

    cnt=0
    for name in names:             # loop through all windows
        lSt=lineStyles[cnt % 4]        # set line style, recyle every 5th group

        if cnt in [0]:            #,1,2,3]:
            cls=colors[0:3]
        elif cnt in [4]:
            cls=colors[6:9]
        else:
            cls=colors[3:6]

        for j in range(3):         # loop through TtoF24,StoM24,StoM168
            if (cnt%4)==0:
                if (j==0):
                    r2Vals = r2df[name[j]].values; cl=cls[j]; r2Label=r2LabelsL[j]+str(windows[cnt])
                elif (j==1):
                    r2Vals = r2df[name[j]].values; cl=cls[j]; r2Label=r2LabelsL[j]+str(windows[cnt])
                else:
                    r2Vals = r2df[name[j]].values; cl=cls[j]; r2Label=r2LabelsL[j]+str(windows[cnt])
            else:
                if (j==0):
                    r2Vals = r2df[name[j]].values; cl=cls[j]; r2Label=r2LabelsS[j]+str(windows[cnt])
                elif (j==1):
                    r2Vals = r2df[name[j]].values; cl=cls[j]; r2Label=r2LabelsS[j]+str(windows[cnt])
                else:
                    r2Vals = r2df[name[j]].values; cl=cls[j]; r2Label=r2LabelsS[j]+str(windows[cnt])

            hist,edges = np.histogram(r2Vals, r2Bins, density=False)  #data, bins, False=>Hist contain #samples in each bin.
            #ax1.scatter(bin_centres,hist,color=cl,edgecolor=cl, alpha = 0.001, facecolor= 'None')   # plots point at centre of bar

            if cnt in [0]:
                ax1.plot(bin_centres,hist,label=r2Label,color=cl,linestyle=lSt,lw=3,alpha=0.7)       # plot line histogram through points
            elif cnt in [4]:
                ax1.plot(bin_centres,hist,label=r2Label,color=cl,linestyle='dotted',lw=3,alpha=0.7)       # plot line histogram through points
            else:
                ax1.plot(bin_centres,hist,color=cl,linestyle=lSt,lw=1,alpha=0.2)  #label=r2Label,

            #if (cnt<1):
            #    ax1.bar(edges[0:-1]+widths/2,hist,width=widths,edgecolor='k', alpha = 0.001, facecolor= 'None')
        cnt+=1

    ax1.set_xlabel('R2 Value',fontsize=24)
    ax1.set_ylabel('# of users',fontsize=24)
    ax1.set_ylim([0,200])
    ax1.set_xlim([0,1.01])
    ax1.set_xticks(np.arange(0,1.01,0.1))
    ax1.tick_params(labeltop=False, labelright=False,labelsize=18)

    pltTitle = kwargs.get('pltTitle')
    #ax1.set_title('R2 Values for Hourly Data Persistance Models: '+str(pltTitle),fontsize=20)
    ax1.grid(which='both')

    plt.legend(ncol=2,fontsize=18,columnspacing=1)


    SaveFigTitle = kwargs.get('SaveFigTitle', None)
    print(SaveFigTitle)
    if not SaveFigTitle == 'None':
        fname = '../project_data/FiguresLCL/'+str(SaveFigTitle)+'.png'
        fig.savefig(fname, dpi=300, format='png',  bbox_inches='tight')


# Error calculation functions
# FUNCTION TO CALULATE APE
# kwargs: minLim (default 0.001) - to avoid divide by 0

def mape_calc(loadSeries,modelSeries,**kwargs):
    if(isinstance(loadSeries,pd.Series)):
        loadSeries = loadSeries.values
    if(isinstance(modelSeries,pd.Series)):
        modelSeries = modelSeries.values

    if len(modelSeries) != len(loadSeries):
        print("model and load series have different lengths")
        return()

    # code to replace v small value with minLim
    if 'minLim' in kwargs:
        minLim=kwargs.get('minLim')
    else:
        minLim=0.001

    loadSeries[loadSeries < minLim] = minLim

    # warn if denominator is 0 or v small
    if (np.amin(loadSeries) < 0.001):
        print("WARNING: at least one denominator value < 0.001, error values whould be reviewed  ")

    return(np.abs(modelSeries-loadSeries)/np.abs(loadSeries))  


# FUNCTION TO CALULATE CV (Coefficient of Variance)
def cv_calc(loadSeries,modelSeries):
    if(isinstance(loadSeries,pd.Series)):
        loadSeries = loadSeries.values
    if(isinstance(modelSeries,pd.Series)):
        modelSeries = modelSeries.values

    if len(modelSeries) != len(loadSeries):
        print("model and load series have different lengths")
        return()
    
    return(np.sqrt(np.mean(np.power(modelSeries-loadSeries,2)))/np.mean(loadSeries))  

# FUNCTION TO CALULATE RMSE 
def rmse_calc(loadSeries,modelSeries):
    if(isinstance(loadSeries,pd.Series)):
        loadSeries = loadSeries.values
    if(isinstance(modelSeries,pd.Series)):
        modelSeries = modelSeries.values

    if len(modelSeries) != len(loadSeries):
        print("model and load series have different lengths")
        return()
    
    return(np.sqrt(np.mean(np.power(modelSeries-loadSeries,2))))

# FUNCTION TO CALULATE range normalised  RMSE 
def nrmse_calc(loadSeries,modelSeries):
    if(isinstance(loadSeries,pd.Series)):
        loadSeries = loadSeries.values
    if(isinstance(modelSeries,pd.Series)):
        modelSeries = modelSeries.values

    if len(modelSeries) != len(loadSeries):
        print("model and load series have different lengths")
        return()
    
    return(np.sqrt(np.mean(np.power(modelSeries-loadSeries,2)))/(np.amax(loadSeries)-np.amin(loadSeries)))


# FUNCTION TO CALULATE range normalised  RMSE using method of Humeau et al 2013 (normalize using L2 norm of series)
def nrmse_calcV2(loadSeries, modelSeries):
    if (isinstance(loadSeries, pd.Series)):
        loadSeries = loadSeries.values
    if (isinstance(modelSeries, pd.Series)):
        modelSeries = modelSeries.values

    if len(modelSeries) != len(loadSeries):
        print("model and load series have different lengths")
        return ()

    return (np.sqrt(np.mean(np.power(modelSeries - loadSeries, 2))) /  np.sqrt(np.mean(np.power(loadSeries, 2)))    )


# FUNCTION TO CALULATE MAE (mean average error)
def mae_calc(loadSeries,modelSeries):
    if(isinstance(loadSeries,pd.Series)):
        loadSeries = loadSeries.values
    if(isinstance(modelSeries,pd.Series)):
        modelSeries = modelSeries.values

    if len(modelSeries) != len(loadSeries):
        print("model and load series have different lengths")
        return()
    
    return(np.abs(modelSeries-loadSeries))

# define function to compute errors : need previous error functions defined first

# yData = load series (pd.Series or np array)
# models = dictionary of (1 or more) model forecast series (pd.Series or ndarray)
# args: nrmseNorm : use L2 normalization
# kwargs: minLim - used by mape_calc

def get_errors(yData,models,*args,**kwargs):
    if(isinstance(yData,pd.Series)):
        yData = yData.values

    modelNames = list(models.keys())
    nModels = len(modelNames)
    
    APEs = np.zeros((len(yData),nModels))
    CVs = np.zeros((len(yData),nModels))
    RMSEs = np.zeros((len(yData),nModels))
    NRMSEs = np.zeros((len(yData),nModels))
    AEs = np.zeros((len(yData),nModels))
    
    for i, (modName,modSeries) in enumerate(models.items()):
        if(isinstance(modSeries,pd.Series)):
            modSeries = modSeries.values
            
        #next bit required to handle minMaxScaler outputs
        if len(modSeries.shape) > 1:
            modSeries = np.squeeze(modSeries)

        if 'minLim' in kwargs:
            minLim = kwargs.get('minLim')
        else:
            minLim = 0.001
        APEs[:,i] = mape_calc(yData,modSeries,minLim=minLim)
        CVs[:,i] = cv_calc(yData,modSeries)
        RMSEs[:,i] = rmse_calc(yData,modSeries)
        if 'nrmseNorm' in args:
            NRMSEs[:, i] = nrmse_calcV2(yData, modSeries)
        else:
            NRMSEs[:,i] = nrmse_calc(yData,modSeries)
        AEs[:,i] = mae_calc(yData,modSeries)
    MAPEs = np.mean(APEs, axis=0)
    MCVs = np.mean(CVs, axis=0)
    MRMSEs = np.mean(RMSEs, axis=0)
    MNRMSEs = np.mean(NRMSEs, axis=0)
    MAEs = np.mean(AEs, axis=0)
    
    errorDict = {'MAPEs':MAPEs,'MCVs':MCVs,'MRMSEs':MRMSEs,'MNRMSEs':MNRMSEs,'MAEs':MAEs}
    #errorDict = {'MAPEs':MAPEs,'MCVs':MCVs,'MNRMSEs':MNRMSEs,'MAEs':MAEs}
    
    return(errorDict)


#  general functions for MLR and ANN eg plotting, group aggregations etc

#  plot Features - plot 'n' columns of np allFeatures array starting from column 'startCol'

def plot_allFeatures(startCol, n, allFeatures, **kwargs):
    fig = plt.figure(figsize=(12, n * 2))
    axs = []
    lineColors = ['k', 'b', 'g', 'r', 'm', 'c']

    if 'windowLength' in kwargs:
        windowLength = kwargs.get('windowLength')
    else:
        windowLength = allFeatures.shape[0]
        print(windowLength)

    for i, j in enumerate(list(range(startCol, startCol + n, 1))):
        # print(i,j)
        axs.append(fig.add_subplot(n, 1, i + 1))

        if len(allFeatures.shape) <= 1:
            axs[i].plot(np.arange(windowLength), allFeatures[:windowLength], color=lineColors[j % 6])
        else:
            axs[i].plot(np.arange(windowLength), allFeatures[:windowLength, j], color=lineColors[j % 6])

        # axs[i].plot(np.arange(allFeatures.shape[0]), allFeatures[:,j], color=lineColors[j%6])
        axs[i].tick_params(axis='both', labelsize=18)
        axs[i].set_ylabel('col' + str(j), fontsize=18);
    fig.tight_layout()

# funtion to plot a time window of all passed models - load profiles stacked by calendar day

# inputs: load/model series, #weeks to plot 

def stackplot_of_forecast(models,numWeeks,*args,**kwargs):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors2
    import matplotlib.cm as cmx

    cmOcc,cNorm,scalarMap3 = setColMap('Oranges',minV=0,maxV=numWeeks*7)  # colorMap for days in Prediction Periods
    seriesLength = len(models[list(models.keys())[0]])  # assumes all model series same length
    
    nPlots = len(list(models.keys()))
    print("{0} weeks, {1} 'models'of seriesLength: {2}".format(numWeeks,nPlots,seriesLength))

    
    fig = plt.figure(figsize = (10,int((nPlots+1)/2)*4))
    axs = [fig.add_subplot(int((nPlots+1)/2),2,i+1) for i in range(nPlots)]  # add subplot for each 'model'

    plotTitles = list(models.keys())
    
    # Look at each day in the forecast period
    for i, (modName,modSeries) in enumerate(models.items()):
        if(isinstance(modSeries,pd.Series)):
            modSeries = modSeries.values
        if 'print_output' in args:
            print("Model name: {0} Length of Series: {1}, #days {2}".format(modName,len(modSeries),len(modSeries.reshape(int(seriesLength/24),24))))
        for j,row in enumerate(modSeries.reshape(int(seriesLength/24),24)):      # loop through #days in model series
            #print("j {1},colourmap denominator:{0}".format( (j/np.float(seriesLength/24))*(numWeeks*7),j ))
            cl = scalarMap3.to_rgba((j/np.float(seriesLength/24))*(numWeeks*7))
            cl = colors2.rgb2hex(cl)
            axs[i].plot(np.arange(1,25),row,color=cl)
            
        axs[i].set_title(plotTitles[i],fontsize=18)
        axs[i].set_xlim([1,24])
        axs[i].set_ylabel('load (kW)', fontsize=12)
        if i==0:
            yLim = get_limits(modSeries, decimals=-1)     
        axs[i].set_ylim([yLim[0],yLim[1]])
        axs[i].set_xticks(np.array([6,12,18]))
        axs[i].set_xticklabels(['6am','12pm','6pm'])
        axs[i].tick_params(axis='both', labelsize=11)
        axs[i].set_yticks([0,300,600])               

    #plt.subplots_adjust(wspace=0.45, hspace=0.5)
    fig.tight_layout(pad=0.2)

    cbar_ax = plt.axes([1.02, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(scalarMap3, cax=cbar_ax, ticks=np.arange(0,numWeeks*7,5), orientation='vertical')
    cbar_ax.tick_params(axis='both', which='major', labelsize=12)
    cbar.set_label('Day number', labelpad=10, rotation=90, fontsize=12)

    # Add a title to the Figure
    if 'figTitle' in kwargs:
        figTitle = kwargs.get('figTitle')
        fig.suptitle(figTitle, fontsize=18)
    fig.tight_layout()
    fig.subplots_adjust(top=0.83)

    
    if 'figName' in kwargs:
        figName = kwargs.get('figName')
        fig.savefig(figName, dpi=300, format='png',  bbox_inches='tight')


# define function to do 'out of sample' selection of random user subsets of size N,
# ...given input hourly load dataframe with uID etc

def randomUserGroups(hourly_df,N=10):
    import random

    uIDs = hourly_df.index.values   # get all userIDs
    columns=[]  # column names
    randIDgrp=[]  # each subset 
    uIDs_OutOfSample=[] # remaining users to be sampled
    df_randIDgrps = pd.DataFrame()  # df with N rows

    for i in range(int(uIDs.shape[0]/N)):   # loop over number of N sized groups
        if (i==0):
            uIDs_OutOfSample = [x for x in uIDs]  # initially just the full set uIDs
        else:
            uIDs_OutOfSample = [x for x in uIDs_OutOfSample if x not in randIDgrp]

        columns.append('grp'+str(i))
        randIDgrp = random.sample(uIDs_OutOfSample, N)
        df_randIDgrps = pd.concat([df_randIDgrps,pd.DataFrame(randIDgrp)],axis=1)

    df_randIDgrps.columns = columns
    return(df_randIDgrps)

# define function to form aggregate load for each set of random user groups

def randomUserGroupAggregate(hourly_df,df_randIDgrps):
    df_grpAggLoad = pd.DataFrame()
    
    for col in df_randIDgrps.columns:
        randIDs = df_randIDgrps[col]
        grpAggLoad = hourly_df.loc[randIDs].sum(axis=0)    # sum each column ie aggregate of all users at each time 
        df_grpAggLoad  = pd.concat([df_grpAggLoad,pd.DataFrame(grpAggLoad)],axis=1)
    df_grpAggLoad.columns = df_randIDgrps.columns
    return(df_grpAggLoad)


# In[32]:


#  for MLR ..but could be used for aggregate ANN fc errors too?
# errors = dictionary of errors
# models = forecast results
# kwargs: 1) figTitle
# args: '%only' : only show MAPE and CV % errors, don't show ax2 MAE, RMSE, NMRSE etc

def plot_errors(errors, models, myStyles, *args, **kwargs):
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(111)

    if '%only' not in args:
        ax2 = ax1.twinx()

    lineCols = myStyles['lineColors']
    lineStyles = myStyles['lineStyles']

    nModels = len(errors[list(errors.keys())[0]])
    nErrors = len(list(errors.keys()))
    ax2Yrange = [0, 0.001]

    for i, (errorType, errorVals) in enumerate(errors.items()):  # loop thro' error types
        if errorType in ['MAPEs', 'MCVs','MNRMSEs']:
            ax1.plot(np.arange(nModels), errorVals * 100, color=lineCols[i % 6], lw=2, label=errorType,
                     linestyle=lineStyles[i % 4])
            # ax1.bar(np.arange(nModels),errorVals*100)
            ax1.scatter(np.arange(nModels), errorVals * 100, color=lineCols[i], alpha=0.5)
        else:
            if '%only' not in args:
                ax2.plot(np.arange(nModels), errorVals, color=lineCols[i % 6], lw=2, label=errorType,
                         linestyle=lineStyles[i % 4])
                ax2.scatter(np.arange(nModels), errorVals, color=lineCols[i], alpha=0.5)
                l = get_limits(errorVals, decimals=0)
                if (l[1] > int(ax2Yrange[1])):
                    ax2Yrange[1] = l[1]

    ax1.set_ylabel('Error (%)', fontsize=12)
    ax1.grid(which='both')
    ax1.set_xticks(np.array(np.arange(nModels)))

    maxV = nErrors * [0]
    for j, v in enumerate(errors.values()):
        maxV[j] = max(k for k in v)

    # maxV = 100*max(j for v in errors.values() for j in v)
    print("max Value of all errors % : {0}".format(maxV))
    # ax1.set_ylim([0,maxV])
    ax1.set_ylim([0, 15])

    ax1.set_xticklabels(list(models.keys()), rotation='vertical')

    ax1.tick_params(axis='both', labelsize=12)

    ax1.legend(ncol=1, fontsize=12, loc='upper left')
    if '%only' not in args:
        ax2.set_ylim(ax2Yrange)
        ax2.set_ylabel('Error (abs units)', fontsize=12)
        ax2.legend(ncol=1, fontsize=12, loc='lower right')

    if 'figTitle' in kwargs:
        figTitle = kwargs.get('figTitle')
        ax1.set_title(figTitle, fontsize=18)


    return(fig,ax1)


# Function to generate boxplot/swarmplot of errors ... VERSION1: order suplots BY ERROR TYPES - less useful 

# inputs: a) errors dictionary (all error types) b) categorical list 'oreder' e.g. aggregation_N converted to strings

# kwargs: model (model name), figTitle (banner title) ,figSave (filename)

def sbn_plotErrors(errorsAgg_flipped,order,**kwargs):
    nKeys = len(errorsAgg_flipped.keys())   # #error types 
    
    #check if model name (i.e. key for extracting dataframe)
    if 'model' in kwargs:
        model = kwargs.get('model')
        if isinstance(model,list):
            model=model[0]
    else:
        model = 'quadratic'
   
    fig = plt.figure(figsize=(nKeys*6,6))
    axs=[]
    #axs = [fig.add_subplot(int((nPlots+1)/2),2,i+1) for i in range(nPlots)] 


    for i, (k,v) in enumerate((errorsAgg_flipped).items()):    # loop through error types
        axs.append(fig.add_subplot(1,nKeys,i+1))
        axs[i].set_ylim(0,0.6)
        axs[i].set_title(str(k),fontsize=16)

        df_ = pd.DataFrame()   

        for subK,subV in v.items():
            df = pd.DataFrame(subV.loc[model])   # in ANN version.. df = subV.T as only one model
            
            df.columns=['error']
            df['N']=subK
            df=df.reset_index()
            df_ = pd.concat([df_,df],axis=0)

        s = sns.swarmplot(x="N", y="error",data=df_,ax=axs[i],order=order,alpha=0.5)
        s = sns.boxplot(x="N", y="error", data=df_, ax=axs[i],width=0.2,order=order,boxprops=dict(alpha=0.8))
        for i,box in enumerate(s.artists):
            box.set_edgecolor('black')
            box.set_facecolor('white')            
        s.set_xlabel("aggregation level N",fontsize=16)
        s.set_ylabel('error',fontsize=16)
 
    # Add a title to the Figure
    if 'figTitle' in kwargs:
        figTitle = kwargs.get('figTitle')
        fig.suptitle(figTitle, fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.83)

    if 'figSave' in kwargs:
        figSave = kwargs.get('figSave')
        fname = str(figSave)+'.png'
        fig.savefig(fname, dpi=300, format='png',  bbox_inches='tight')


# Function to generate boxplot/swarmplot of errors ...
# Version 2 : one error type e.g. MAPE, subplot for each MODEL TYPE (just 1 model type so far for ANN) : 'Model' = df row index
# inputs:
# a) errors dictionary for specific errorType
# b) categorical list e.g. aggregation_N converted to strings
# kwargs:
#   percentiles  # dictionary with df for each test/train or just test (if want to add percentile traces)
#   figTitle (banner title)
#   subTitle (smaller txt banner sub-heading)
#   figSave (filename)
#
#   dropFromPlot (models that don't wish to plot eg seasonal naive model)
#   errorType   # for labelling
#   inputFig   # if wish to add to existing figure
#   inputAxes   # if wish to add to all subplots of existing figure
#   yLim

def sbn_plotErrors_V2(errorsAgg_flipped_error,order,percentiles=None,figTitle=None,subTitle=None,\
                      figSave=None,errorType=None,**kwargs):
    import math
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,MaxNLocator,PercentFormatter)

    if 'dropFromPlot' in kwargs:
        dropFromPlot = kwargs.get('dropFromPlot')
    else:
        dropFromPlot = []

    # get model names (and hence # subplots - assume one subplot for each model)
    for i, (k,v) in enumerate((errorsAgg_flipped_error).items()):   # loop thro AGGREG. LEVELS (to get models,nPlots)
        if i==0:
            v=v.T
            models=v.columns  # same for all its e.g. 'quadratic','ANN_test errors','ANN_train_errors'
            nPlots=v.shape[1] # same for all its
    nPlots = nPlots - len(dropFromPlot)

    #create figure object and axs of nPlots subplots
    if 'inputFig' not in kwargs:
        fig = plt.figure(figsize=(18,int(math.ceil(nPlots/2))*8))
        axs = [fig.add_subplot(int(math.ceil(nPlots/2)),2,i+1) for i in range(nPlots)]
    else:
        fig = kwargs.get('inputFig')
        if 'inputAxes' not in kwargs:
            print("Must supply axes object")
            return()
        axs = kwargs.get('inputAxes')

    for i,mod in enumerate(models):                        # loop through models (subplot for each)
        if mod not in dropFromPlot:
            if 'yLim' not in kwargs:
                yLim = [0.0, 0.6]
            else:
                yLim = kwargs.get('yLim')

            axs[i].set_ylim(0,0.6)                             # referencing subplot for ith model
            axs[i].set_title(str(mod),fontsize=14)             # title for subplot

            for j, (k,v) in enumerate((errorsAgg_flipped_error).items()):    # loop through AGGREGATION LEVELS
                v=v.T
                v['N']=k
                #print("model: {0}, agg level {1}".format(models[i],k))
                sns.swarmplot(x="N", y=mod,data=v,ax=axs[i],order=order,alpha=0.5)
                s = sns.boxplot(x="N", y=mod, data=v, whis=[5, 95], ax=axs[i],width=0.2,order=order,\
                                boxprops=dict(alpha=0.8),showmeans=False,\
                                meanprops={"marker":"o","markerfacecolor":"white","markeredgecolor":"black","markersize":"3"})
            for k,box in enumerate(s.artists):
                box.set_edgecolor('black')
                box.set_facecolor('white')
            s.set_xlabel("aggregation level N",fontsize=14)

            if errorType is not None:
                s.set_ylabel(str(errorType) + ' error', fontsize=14)
            else:
                s.set_ylabel('error', fontsize=14)

            #axs[i].set_yticklabels(fontsize=12)

            axs[i].yaxis.set_major_locator(MultipleLocator(0.1))
            axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axs[i].yaxis.set_major_formatter(PercentFormatter(1.0))


            # For the minor ticks, use no labels; default NullFormatter.
            axs[i].yaxis.set_minor_locator(MultipleLocator(0.05))

            for item in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
                item.set_fontsize(14)

            axs[i].grid(b=True, which='major', axis='y')
            axs[i].grid(b=True, which='minor', axis='y',alpha=0.3)

            # add percentile traces   - not yet implemented
            if percentiles is not None:
                print(percentiles.keys())

                #df = kwargs.get('percentile_df')
                #percentiles = [int(p) for p in df.columns.to_list()[:-1]]
                #lineCls1 = ['k']*len(percentiles)
                #lineCls2 = ['r']*len(percentiles)
                #lineSty = ['dotted','dashed','solid','dashed','dotted']  #'dashdot',
                #lineAlp = [0.1,0.6,0.2,0.6,0.1]
                #symSize = [5,10,30,10,5]

                #for i,p in enumerate(percentiles):
                #    axs[i].plot(df['N'].values,df[str(p)].values,alpha=lineAlp[i],linestyle=lineSty[i],color=lineCls1[i])
                #    axs[i].scatter(df['N'].values,df[str(p)].values,s=symSize[i],alpha=0.6,color='k')
    # Add a title to the Figure
    if figTitle is not None:
        fig.suptitle(figTitle, fontsize=18, y=1.0)
    if subTitle is not None:
        fig.text(0.5, 0.94, subTitle, fontsize=14, ha='center')

    #fig.tight_layout()
    #fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

    if figSave is not None:
        fname = str(figSave)+'.png'
        fig.savefig(fname, dpi=300, format='png',  bbox_inches='tight')

    return(fig,axs)   # return figure object in case wish to subsequently add items






# sbn_errorsAllModels  - Display ALL models on ONE plot using hue to represent each model
# inputs:
# MLR (incl naive models) and ANN error dictionaries (flipped);
# order = x axis categorical order; errorType
# args: None
# kwargs:
# figure title and file name to save
# errorType
# dropFromPlot = list of models not to show; palette = custom color palette; flierSize = show outliers & size
# whiskers = whisker limits e.g. 10,90 (%)
# subPlotSize = (x,y)

def sbn_ErrorsAllModels(MLRerrorsAgg_flipped_error, ANNerrorsAgg_flipped_error, order, errorType, figTitle=None, \
                        figSave=None, **kwargs):
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator, MaxNLocator, PercentFormatter)

    # in case wnt to drop models
    if 'dropFromPlot' in kwargs:
        dropFromPlot = kwargs.get('dropFromPlot')
    else:
        dropFromPlot = []

    # set boxplot parameters
    if 'boxplot_kwargs' in kwargs:
        boxplot_kwargs = kwargs.get('boxplot_kwargs')
    else:
        boxplot_kwargs = {'boxprops': {'edgecolor': 'k', 'linewidth': 1, 'alpha': 0.8}, 'palette': 'coolwarm',
                          'width': 0.8, 'linewidth': None}

    # set display of outlier data points
    if 'flierSize' in kwargs:
        flierSize = kwargs.get('flierSize')
    else:
        flierSize = 0

    # set whisker percentiles
    if 'whiskers' in kwargs:
        whiskers = kwargs.get('whiskers')
    else:
        whiskers = [10, 90]

    dfErrors = pd.DataFrame()
    for (k1, v1), (k2, v2) in zip(MLRerrorsAgg_flipped_error[errorType].items(),
                                  ANNerrorsAgg_flipped_error[errorType].items()):
        # print(k1,k2)
        v1 = v1.T
        v2 = v2.T
        v1['N'] = k1
        v1 = pd.concat([v2, v1], axis=1)
        dfErrors = pd.concat([dfErrors, v1], axis=0)
        dfErrors = dfErrors.drop(dropFromPlot, axis=1)

    # convert merged df to long (narrow) form:
    dfErrorsAllModels = pd.melt(dfErrors, id_vars=['N'], var_name='model', value_name='errors')

    if 'subPlotSize' in kwargs:
        subPlotSize = kwargs.get('subPlotSize')
    else:
        subPlotSize = (18, 8)

    # create figure object and axs of nPlots subplots
    fig = plt.figure(figsize=(subPlotSize[0], subPlotSize[1]))
    axs = fig.add_subplot(111)

    modelsList = dfErrors.columns[:-1]
    print("modelsList:", modelsList)

    if 'modelsCompare' in kwargs:
        modelsCompare = kwargs.get('modelsCompare')
        yVar = str(modelsCompare[0])
        hueVar = str(modelsCompare[-1])
        data_sns = dfErrors
        print("columns:", dfErrors.columns, "yVar: ", yVar, "hue_var: ", hueVar)
    else:
        yVar = 'errors'
        hueVar = 'model'
        data_sns = dfErrorsAllModels

    if 'stripplot_kwargs' in kwargs:
        stripplot_kwargs = kwargs.get('stripplot_kwargs')
        sns.stripplot(x="N", y=yVar, hue=hueVar, data=data_sns, order=order, ax=axs, **stripplot_kwargs)
    if 'swarmplot_kwargs' in kwargs:
        swarmplot_kwargs = kwargs.get('swarmplot_kwargs')
        sns.swarmplot(x="N", y=yVar, hue=hueVar, data=data_sns, order=order, ax=axs, **swarmplot_kwargs)

    s = sns.boxplot(x="N", y=yVar, hue=hueVar, data=data_sns, whis=whiskers, ax=axs,
                    order=order, showmeans=False, fliersize=flierSize, **boxplot_kwargs)  # palette=customPalette)

    if 'yLim' not in kwargs:
        yLim = [0.0, 1.0]
    else:
        yLim = kwargs.get('yLim')

    axs.set_ylim(yLim[0], yLim[1])  # referencing subplot for ith model

    s.set_xlabel("aggregation level N")  # , fontsize=14)
    s.set_ylabel(str(errorType[:-1] + ' errors'))  # , fontsize=14)

    axs.yaxis.set_major_locator(MultipleLocator(0.1))
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs.yaxis.set_major_formatter(PercentFormatter(1.0))

    # For the minor ticks, use no labels; default NullFormatter.
    axs.yaxis.set_minor_locator(MultipleLocator(0.05))

    # instead : use rcParams settings in notebook
    # for item in (axs.get_xticklabels() + axs.get_yticklabels()):
    #    item.set_fontsize(14)

    axs.grid(b=True, which='major', axis='y')
    axs.grid(b=True, which='minor', axis='y', alpha=0.3)

    # Add a title to the Figure
    if figTitle is not None:
        axs.set_title(figTitle)  # , fontsize=18)  # , y=1.0)

    # if subTitle is not None:
    #    fig.text(0.5, 0.94, subTitle, fontsize=14, ha='center')

    # axs.legend(ncol=1) #, fontsize=20)

    if figSave is not None:
        fname = str(figSave) + '.png'
        fig.savefig(fname, dpi=300, format='png', bbox_inches='tight')

    print("return types: {0},{1}".format(type(dfErrors), type(dfErrorsAllModels)))

    return (dfErrors, dfErrorsAllModels, fig, axs)

# merge errors from MLR and ANN models from dictionaries

def errorsAllModels(MLRerrorsAgg_flipped_error, ANNerrorsAgg_flipped_error,  errorType):

    dfErrors = pd.DataFrame()
    for (k1, v1), (k2, v2) in zip(MLRerrorsAgg_flipped_error[errorType].items(),
                                  ANNerrorsAgg_flipped_error[errorType].items()):
        # print(k1,k2)
        v1 = v1.T
        v2 = v2.T
        v1['N'] = k1
        v1 = pd.concat([v2, v1], axis=1)
        dfErrors = pd.concat([dfErrors, v1], axis=0)

    # convert merged df to long (narrow) form:
    dfErrorsAllModels = pd.melt(dfErrors, id_vars=['N'], var_name='model', value_name='errors')

    modelsList = dfErrors.columns[:-1]
    print("modelsList:", modelsList)

    print("return types: {0},{1}".format(type(dfErrors), type(dfErrorsAllModels)))

    return (dfErrors, dfErrorsAllModels)    # wide and narrow forms of same data



# function to plot load distribution by hour of day
def boxplotLoad(load_df, xOrder, *args, figTitle=None, **kwargs):
    from matplotlib.ticker import (MultipleLocator,FormatStrFormatter)

    # set custom palette
    if 'palette' in kwargs:
        customPalette = kwargs.get('palette')
    else:
        customPalette = 'coolwarm'

    # set display of outlier data points
    if 'flierSize' in kwargs:
        flierSize = kwargs.get('flierSize')
    else:
        flierSize = 0

    # set whisker percentiles
    if 'whiskers' in kwargs:
        whiskers = kwargs.get('whiskers')
    else:
        whiskers = [5, 95]

    # set bar width
    if 'width' in kwargs:
        width = kwargs.get('width')
    else:
        width = 0.2

    # set x var
    if 'xVar' in kwargs:
        xVar = kwargs.get('xVar')
    else:
        xVar = 'hour'

    # set y var
    if 'yVar' in kwargs:
        yVar = kwargs.get('yVar')
    else:
        yVar = 'loadValues'

    # set hue (e.g. if want multiple users or grps at each x axis position)
    if 'hue' in kwargs:
        hue = kwargs.get('hue')
    else:
        hue = None

        # set fig size
    if 'figSize' in kwargs:
        figSize = kwargs.get('figSize')
    else:
        figSize = (10, 5)

    fig = plt.figure(figsize=figSize)
    axs = fig.add_subplot(111)

    s = sns.boxplot(x=xVar, y=yVar, hue=hue, data=load_df, whis=whiskers, ax=axs, width=width,
                    order=xOrder, boxprops=dict(alpha=0.8), showmeans=False, \
                    fliersize=flierSize, linewidth=None, palette=customPalette)

    if 'stripPlot' in args:
        sns.stripplot(x=xVar, y=yVar, hue=hue, data=load_df,
                      jitter=True, split=True, linewidth=0.5)

    if 'yLim' not in kwargs:
        yLim = [0.0, 1.0]
    else:
        yLim = kwargs.get('yLim')

    axs.set_ylim(yLim[0], yLim[1])  # referencing subplot for ith model

    if 'yGridInterval' in kwargs:
        yGridInterval = kwargs.get('yGridInterval')
    else:
        yGridInterval = 0.1

    axs.yaxis.set_major_locator(MultipleLocator(yGridInterval))
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # For the minor ticks, use no labels; default NullFormatter.
    axs.yaxis.set_minor_locator(MultipleLocator(yGridInterval / 2))

    for item in (axs.get_xticklabels() + axs.get_yticklabels()):
        item.set_fontsize(14)

    axs.grid(b=True, which='major', axis='y')
    axs.grid(b=True, which='minor', axis='y', alpha=0.3)

    # Add a title to the Figure
    if figTitle is not None:
        axs.set_title(figTitle, fontsize=14)  # , y=1.0)

    axs.set_ylabel("hourly load - kWh", fontsize=14)
    axs.set_xlabel("hour of day", fontsize=14)

    # axs.legend(loc='upper left')

    return;


# shiftInputs :
# shift models/forecast series to align with hr/day indices of load ie unpack into actual time bands
# MLR version - slight differences to annPlotInputs ANN version

# load   = full continuous load series with datetime index
# windows = train/test windows (all)
# loadSeries = model series
# trainTestFlg = 'train' or 'test' (or none => default 'test')

#NB - this one is a pain - keep having to change the line with .reshape !
def shiftInputs(load,windows,loadSeries,trainTestFlg):  #load = eg aggLoad ,latest version here as keep changing : put in .py package when tody up code at end of project
    seriesPlot = np.empty_like(load)
    seriesPlot[:] = np.nan
    lengthSum = 0

    for i in range(len(windows)):
        if trainTestFlg == 'train':
            lengthInd  = windows[i,1]-(windows[i,0]+168)
            if (len(np.squeeze(loadSeries[lengthSum:lengthSum+lengthInd]).shape) < 2):
                #seriesPlot[(windows[i,0]+168):windows[i,1]]  = np.squeeze(loadSeries[lengthSum:lengthSum+lengthInd]).reshape(-1,1)  #.astype(object) #
                seriesPlot[(windows[i,0]+168):windows[i,1]]  = np.squeeze(loadSeries[lengthSum:lengthSum+lengthInd]).astype(object)  #.astype(object) #
        else:
            lengthInd  = windows[i,3]-(windows[i,2])
            if (len(np.squeeze(loadSeries[lengthSum:lengthSum+lengthInd]).shape) < 2):
                #seriesPlot[windows[i,2]:windows[i,3]]  = np.squeeze(loadSeries[lengthSum:lengthSum+lengthInd]).reshape(-1,1)       #.astype(object)
                seriesPlot[windows[i,2]:windows[i,3]]  = np.squeeze(loadSeries[lengthSum:lengthSum+lengthInd]).astype(object)       #.astype(object)

        lengthSum += lengthInd


    return(seriesPlot)


# function to plot load and forecasts

# windows = training/test windows used (assume the same for all models)
# load  = actual load data - needs to have datetime index
# start = hour of year index
# models = dictionary of forecast models - assumes all same length
# args:
# 'xLabel' : if not given, don't plot x axis label
# kwargs:
# profileLength = number of days in each profile plot (if numWeeksToPlot is longer than profileLength, multiple subplots will be made)
# myStyles = dictionary of colors, linestyles to use
# subPlotSize = (x,y) tuple , to change from default
# yLim = [y_min,y_max] list, to change from default which is computed automatically based on values
# figTitle  : figure sup title
# figName : path/filename if wish to save png of figure

def plot_forecast(numWeeksToPlot, windows, start, load, models, *args, **kwargs):
    import matplotlib.ticker as ticker  # import the full shebang
    # from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator,MaxNLocator)
    import matplotlib.lines as mlines  # 2D lines

    window = []

    if 'profileLength' in kwargs:
        profileLength = kwargs.get('profileLength')
    else:
        profileLength = 7

    dayLabels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    if 'subPlotSize' in kwargs:
        subPlotSize = kwargs.get('subPlotSize')
    else:
        subPlotSize = (20, 4)

    if 'myStyles' in kwargs:
        myStyles = kwargs.get('myStyles')
        lineCols = myStyles['lineColors']
        lineStyles = myStyles['lineStyles']
        if 'alpha' in myStyles.keys():
            alphaVals = myStyles['alpha']
        else:
            alphaVals = [0.7,1.0,1.0,1.0,1.0,1.0]
        if 'lw' in myStyles.keys():
            lws = myStyles['lw']
        else:
            lws = [1,1,1,1,1,1]

    else:
        lineCols = ['k', 'r', 'g', 'c', 'm', 'y']
        lineStyles = ['solid', 'solid', 'solid', 'solid']
        alphaVals = [0.7, 1.0, 1.0, 1.0, 1.0, 1.0]
        lws = [1,1,1,1,1,1]

    # check if train or test series in models
    if 'train' in str(models.keys()):
        trainTestflg = 'train'
    else:
        trainTestFlg = 'test'

    nColors = len(lineCols);
    nStyles = len(lineStyles)
    nAlphas = len(alphaVals)
    nLws    = len(lws)

    # define RELATIVE indices (relative to 'start' index) for sections to subplot
    for i in range(numWeeksToPlot):
        window.append([i * profileLength * 24, (
                    i + 1) * profileLength * 24])  # each 'window' ie subplot , surrently set to profileLength days for each subplot
    print("relative index : window to plot:", window)

    fig = plt.figure(figsize=(subPlotSize[0], numWeeksToPlot * subPlotSize[1]))
    axs = []

    for i in range(numWeeksToPlot):  # loop through weeks and add subplot for each

        plotStart, plotEnd = start + window[i][0], start + window[i][
            1]  # eg start = windows[5,2] ie last startPred; start = hr of year index where 0 = 00:00 1/Jan
        # NB plotStart, plotEnd are ABSOLUTE indices e.g. out of 0:8760
        print("Index range to plot : plotStart {0}, plotEnd {1}; days {2} to {3}; weeks {4} to {5}".format(plotStart,
                                                                                                           plotEnd,
                                                                                                           plotStart / 24,
                                                                                                           plotEnd / 24,
                                                                                                           plotStart / (
                                                                                                                       24 * 7),
                                                                                                           plotEnd / (
                                                                                                                       24 * 7)))
        axs.append(fig.add_subplot(numWeeksToPlot, 1, i + 1))

        indexVals = load.index  # must start at beginning of year to get valid day numbers on axis
        # find the first midnight tick in (subPlot) section to be plotted
        # j = start+window[i][0] # plotStart
        j = plotStart
        for ind in indexVals:
            if j >= plotStart:
                if ind.hour == 0:
                    startTick = 0 + j
                    print("found startTick {0} on day {1}".format(startTick, startTick / 24))
                    break
            j += 1
        j = 0

        # plot load
        axs[i].plot(np.arange(plotStart, plotEnd), load[plotStart:plotEnd], color=lineCols[0], linestyle=lineStyles[0],
                    alpha=alphaVals[0],lw=lws[0], label='Load')

        # plot 'models'
        for n, (modName) in enumerate(models.values()):
            # print(modName.shape)
            plotSeries = shiftInputs(load, windows, modName, trainTestFlg=trainTestFlg)
            # axs[i].plot(np.arange(plotStart,plotEnd), modName[plotStart:plotEnd],color=lineCols[(n+1)% nColors], linestyle=lineStyles[(n+1)% nStyles])
            axs[i].plot(np.arange(plotStart, plotEnd), plotSeries[plotStart:plotEnd], color=lineCols[(n + 1) % nColors], \
                        linestyle=lineStyles[(n + 1) % nStyles],alpha=alphaVals[(n + 1) % nAlphas], lw=lws[(n+1) % nLws])

            # set axis display limits
        axs[i].set_xlim([plotStart, plotEnd])
        if 'yLim' not in kwargs:
            yLim = get_limits(load[plotStart:plotEnd], decimals=-1)
        else:
            yLim = kwargs.get('yLim')
        axs[i].set_ylim([yLim[0], yLim[1]])

        # set tick mark positions and spacing, labels
        axs[i].set_xticks(np.arange(startTick + 12, plotEnd + 12, 24))  # major ticks (by default)
        axs[i].set_xticks(np.arange(startTick, plotEnd, 24), minor=True)

        tickLabs = []
        for ti in np.arange(startTick, plotEnd, 24):  # doesn't matter if have starttick+12 here - plots in same place
            # print("ti: {0}  startTick: {1} plotEnd {2}, day of year: {3}".format(ti,startTick,plotEnd,indexVals[ti].dayofyear))
            # tickLabs.append((dayLabels[indexVals[ti-start].dayofweek])) #,indexVals[ti-start].dayofyear)
            #tickLabs.append(str(indexVals[ti].dayofyear) + ' ' + str(dayLabels[indexVals[ti].dayofweek]))
            tickLabs.append(str(dayLabels[indexVals[ti].dayofweek]))

        # axs[i].xaxis.set_major_locator(MultipleLocator(1))  # causes compression into left margin

        # axs[i].xaxis.set_major_locator(ticker.MultipleLocator(24))  # causes day labels to shift 1 day out of place !
        axs[i].xaxis.set_minor_locator(
            ticker.MultipleLocator(12))  # minor tick location every 12 hrs (but not shown or labelled)
        # axs[i].xaxis.set_minor_locator = AutoMinorLocator(4)        # don't quite understand what this is doing as get unexpected behaviour
        # axs[i].xaxis.set_minor_formatter(FormatStrFormatter('%.2f'))
        axs[i].grid(b=True, which='major', axis='x', alpha=0.1)  # keep v faint grid at 12 noon where ticklabels are
        axs[i].grid(b=True, which='minor', axis='x', alpha=0.9)  # want prominent grid lines at midnight

        # labels
        axs[i].set_xticklabels(tickLabs);
        axs[i].tick_params(axis='both')

        if 'xLabel' in args:
            axs[i].set_xlabel('Day of Year              Day of Week', fontsize=14, labelpad=-10)

        axs[i].set_ylabel('Load (kW)');

        # line labels for legend
        lineLabels = list(models.keys())
        lines = []
        line = mlines.Line2D([], [], color=lineCols[0], label='load', linewidth=lws[(n+1) % nLws], linestyle=lineStyles[0])
        lines.append(line)

        # add line labels to legend
        for j in range(len(lineLabels)):
            line = mlines.Line2D([], [], color=lineCols[(j + 1) % nColors], label=lineLabels[j], linewidth=lws[(n+1) % nLws])
            lines.append(line)
        axs[i].legend(handles=lines, ncol=len(lineLabels) + 1, frameon=True)  # ,loc='upper right')

    fig.tight_layout()

    if 'figTitle' in kwargs:
        figTitle = kwargs.get('figTitle')
        fig.suptitle(figTitle, y=1)

    if 'figName' in kwargs:
        figName = kwargs.get('figName')
        fig.savefig(figName, dpi=300, format='png', bbox_inches='tight')

    return (fig, axs);


# MLR specific functions

# function to populate allFeatures (Total Aggregate or Subset Aggregate or Individual user cases)
# generalised for MLR 

# yData = concatenation of load series for each [startTrain+168:endPred] window\
#      need yData to identify datetimeindex for HoD,DoW,DoY (actual load values don't matter e.g. can use aggregate for individual loads etc

# load - series or dataframe : required for lagged load calc : pd.Series,starts at DAY0
#           load can be pd.Series eg AggregateLoad or pd.DataFrame e.g. hourly_df or subset aggregate df
#           if latter, should be 'nUser' rows * #Samples columns

# kwargs :
# 'lags' (default : 24,168),
# 'nDoWterms' (default: 6 NB: dow terms and lags must be consistent!)
#       nDoWterms = 6 for 'base case' ie t-24,t-168 ;
#       nDoWterms = 10 if add t-25,t-26 ie. t-24,t-25,t-26,t-168 ;)
# 'PoY' (period of year classifier)
# 'LoD' (length of day series)   : if dayLight OR sunset used in regression


def populate_AllFeatures(windows,featureLength,yData,load,interestTemp,dayClassifier,timeLists,nEnvTerms,**kwargs):

    # define time lags for load series 
    if 'lags' in kwargs:
        lags = kwargs.get('lags')
    else:
        lags = [24,168]
        
    # define number of DayOfWeek terms 
    if 'nDoWterms' in kwargs:
        nDoWterms = kwargs.get('nDoWterms')
    else:
        nDoWterms = 6
        
    if nDoWterms != 2+2*len(lags):
        print("WARNING: #lags and nDowTerms inconsistent")
        
           
    #identify if single series (eg aggregate = 1 'user') or dataframe (eg hourly_df with load series each row)
    # a small aggregation is treated as 1 'user' here ..
    if isinstance(load,pd.Series):
        #print("load is a Series of shape:",load.shape)
        nUsers=1
    elif isinstance(load,pd.DataFrame):
        #print("load is a DataFrame of shape:",load.shape,"with {0} 'users'".format(load.shape[0]))       
        nUsers=load.shape[0]
      
    #initialize 
    allFeatures = np.zeros((featureLength,nEnvTerms+nDoWterms*nUsers))    
    lengthTrainPredSum = 0
    
    #loop through all windows
    for i in range(len(windows)):
        lengthTrainPred = windows[i,3]-(windows[i,0]+168)    # exclude 1st 168 hrs to allow for t-168 load

        for j in range(lengthTrainPredSum,lengthTrainPredSum+lengthTrainPred):    
            l = j - lengthTrainPredSum         # relative index within each period

            hourTrain = yData.index[j].hour                  # for Hour of Day classifier
            dayTrain = yData.index[j].dayofweek              # for DoW classifier
            dayOfYear = yData.index[j].timetuple().tm_yday   # for Day of Year classifier

            # set TEMPERATURE HoD terms : linear f(T) term, a quadratic f(T) term and a constant term 
            # NOTE these are set irrespective of whether or not used (e.g. not used if unAware model)
            dummy = np.zeros((len(timeLists)*3))    # 12 terms
            for k in range(len(timeLists)):         # k = 0 ,1,2,3
                dummy[3*k:3*(k+1)] = (hourTrain in timeLists[k])   # fill HoD terms (1 or 0) for all 12 elements
                dummy[3*k] = dummy[3*k]*interestTemp[j]            # set HoD * T term for each HoD k val
                dummy[3*k+1] = dummy[3*k+1]*(interestTemp[j]**2)   # set HoD * T^2 term for each HoD k val
                dummy[3*k+2] = dummy[3*k+2]*1                      # set HoD * constant(1) foreach HoD k val     #(interestTemp[j]**3)*0     # no cubic T term?
            allFeatures[j,0:min(12,nEnvTerms)] = dummy                            # add all 12 HoD-Temperature terms 

            # set length of DAYLIGHT or SUNSET time PeriodOfYear (H1,H2,etc) terms : linear,quadratic and constant terms 
            # only set if variables to be used in regression
            if ((nEnvTerms % 12) >= 6):
                PoY_list = kwargs.get('PoY')
                interestLoD = kwargs.get('LoD')
                dummy = np.zeros((len(PoY_list)*3))    # eg 6 or 18 terms for h1,h2 or h1..h6
                for k in range(len(PoY_list)):         # k = 0 ,1 (for default case)
                    dummy[3*k:3*(k+1)] = (dayOfYear in PoY_list[k])  # fill PoY terms (1 or 0) for all 6 elements
                    dummy[3*k] = dummy[3*k]*interestLoD[j]           # set PoY* D term for each PoY k val
                    dummy[3*k+1] = dummy[3*k+1]*(interestLoD[j]**2)  # set PoY* D^2 term for each PoY k val
                    dummy[3*k+2] = dummy[3*k+2]*1                         # set PoY * constant(1) for each PoY k val    #(interestTemp[j]**3)*0     # no cubic T term?
                allFeatures[j,12:nEnvTerms] = dummy                             # add all PoY-Daylight  terms 

            # set DoW terms  (all cases) - just 0/1 classifiers set here 
            if (nUsers <= 1):
                # set the DoW constants (1,0) :
                dummy = np.zeros((len(dayClassifier)))  
                for k in range(len(dayClassifier)):     
                    dummy[k] = (dayTrain in dayClassifier[k])   
                allFeatures[j,nEnvTerms:nEnvTerms+2] = dummy  

                # set linear DoW lagged load terms 
                for cnt,lag in enumerate(lags):
                    dummy = np.zeros((len(dayClassifier)))        
                    for k in range(len(dayClassifier)):            
                        dummy[k] = (dayTrain in dayClassifier[k])*load[windows[i,0]+168+l-lag]      
                    allFeatures[j,nEnvTerms+2*cnt+2:nEnvTerms+2*cnt+4] = dummy  

            else:
                for nU in range(nUsers): 
                    if ((i==0) and (l==0)):       # print progress status if at start of 1st window
                        print("period {0}, hour Index {1} for user {2}".format(i,j,nU))
                    dummy = np.zeros((len(dayClassifier)))  
                    for k in range(len(dayClassifier)):     
                        dummy[k] = (dayTrain in dayClassifier[k])   
                    allFeatures[j,nEnvTerms+(nDoWterms*nU):nEnvTerms+(nDoWterms*nU)+2] = dummy   

                    # set linear DoW lagged load terms 
                    for cnt,lag in enumerate(lags):
                        dummy = np.zeros((len(dayClassifier)))        
                        for k in range(len(dayClassifier)):            
                            dummy[k] = (dayTrain in dayClassifier[k])*load.iloc[nU][windows[i,0]+168+l-lag]      
                        allFeatures[j,nEnvTerms+2*cnt+2:nEnvTerms+2*cnt+4] = dummy  

        lengthTrainPredSum += lengthTrainPred
        
    return(allFeatures)


# MLR : splice out TRAINING and PREDICTION windows from allFeatures (for aggregate or individual user cases)
# returns numpy arrays

def create_TrainPredictSeries(trainLength,predLength,allFeatures,windows,nEnvTerms,nDoWterms,*args):
    
    nUsers = int((allFeatures.shape[1]-nEnvTerms)/nDoWterms)
    allFeaturesTrain = np.zeros((trainLength,nEnvTerms+nDoWterms*nUsers)) 
    allFeaturesPredict = np.zeros((predLength,nEnvTerms+nDoWterms*nUsers)) 
    lengthTrainPredSum,lengthTrainSum,lengthPredSum = 0,0,0

    for i in range(len(windows)):
        lengthTrainPred = windows[i,3]-(windows[i,0]+168)
        lengthTrain     = windows[i,1]-(windows[i,0]+168)
        lengthPred      = windows[i,3]-(windows[i,2])
        if 'print_output' in args:
            print("lengthTrain {0} lengthPred {1} lengthTrainPred {2} ({3} weeks)".format(lengthTrain,lengthPred,lengthTrainPred,lengthTrainPred/(24*7)))

        # k = output enumerator, j = input enumerator
        for k,j in enumerate( list( range(lengthTrainPredSum,lengthTrainPredSum+lengthTrain,1) ),lengthTrainSum):                      # enumerate in parallel over abs (k)  and rel(j) index range for TRAINING windows
            #print( "i:",i,"k:",k,"j:",j,range(lengthTrainPredSum,lengthTrainPredSum+lengthTrain,1) ,lengthTrainSum)
            allFeaturesTrain[k,:] = allFeatures[j,:]
            
        for k,j in enumerate(list(range(lengthTrainPredSum+lengthTrain,lengthTrainPredSum+lengthTrainPred)),lengthPredSum):            # enumerate in parallel over abs and rel index range for PREDICTION windows
            #print("predict: i:",i,"k:",k,"j:",j,range(lengthTrainPredSum+lengthTrain,lengthTrainPredSum+lengthTrainPred),lengthPred)
            allFeaturesPredict[k,:] = allFeatures[j,:]
        
        lengthTrainPredSum += lengthTrainPred
        lengthTrainSum += lengthTrain
        lengthPredSum += lengthPred

    #print("'nUsers':",nUsers,"Shapes: allFeatures {0}, allFeaturesTrain {1}, allFeaturesPredict {2}".format\
    #      (allFeatures.shape,allFeaturesTrain.shape,allFeaturesPredict.shape))
    
    return(allFeaturesTrain,allFeaturesPredict)


# function to do MLR training, returns regression model and 'forecast'

# model = model name 
# yData : could be aggregate or individual load series 
def mlr_train(model,features,yData,*args):
    from sklearn import linear_model

    mlrModel = {'object': 'clf'+model, 'coef': 'coef'+model, 'intercept': 'intercept'+model,'r2Score': 'score'+model}
    mlrModel['object'] = linear_model.LinearRegression(fit_intercept=True)    # create sklearn linear regression object
    mlrModel['object'].fit(features,yData)                                    # fit model
    
    mlrModel['coef'] =  mlrModel['object'].coef_                              # get regression coefficients (ie slopes)
    
    mlrModel['intercept'] =  mlrModel['object'].intercept_                    # get Y axis intercept
    
    result = mlrModel['object'].predict(features)                             # run model to get forecast

    mlrModel['r2Score'] =  mlrModel['object'].score(features,yData)           # get R2 score, forecast v data
    if 'print_output' in args:
        print("Regression Model coefficients: \n",mlrModel['coef'])
        print("Regression Model Intercept:",mlrModel['intercept']) 
        print("R2 coefficient: ",mlrModel['r2Score'])
    
    return(mlrModel['object'],result)   
    


# In[40]:


# function to calc predicted load

def mlr_predict(model,features,yData):
    predicted = np.zeros((np.shape(yData)))  # initialize 
    predicted = model.predict(features)
    return(predicted)


# ANN specific functions:

# Training & Prediction Data Setup

# In[41]:


# returns dictionary of env variables for specified windows
# Set for each startTrain(i)+168 : end Pred(i))
# environmental series inputs: full annual ndarrays: e.g. temperature,LengthOfDay, etc

#kwargs: varName,VarValues pairs;
#args = 'ALL' to prevent splitting of each env series into train and test series

def set_envSeries(windows, *args, **kwargs):
    envSeriesDict = {}

    for envVarName, envVar in kwargs.items():
        if 'ALL' in args:
            envAll = []
            for i in range(len(windows)):
                envAll.append([envVar[windows[i, 0] + 168:windows[i, 3]]])
            envSeriesDict[envVarName] = np.array(envAll).reshape(1, -1)[0]
        else:
            envTrain = []
            envTest = []
            for i in range(len(windows)):
                envTrain.append([envVar[windows[i, 0] + 168:windows[i, 1]]])
                envTest.append([envVar[windows[i, 2]:windows[i, 3]]])

            envSeriesDict[str(envVarName) + 'Train'] = np.array(envTrain).reshape(1, -1)[0]
            envSeriesDict[str(envVarName) + 'Test'] = np.array(envTest).reshape(1, -1)[0]

    return (envSeriesDict)


# returns dictionary of DAY OF WEEK classifier

def set_dowSeries(windows,load,dayClassifier):
    dow = np.zeros(((load.shape[0]),2))
    df=pd.DataFrame()
    
    for j in range(load.shape[0]):
        hourTrain = load.index[j].hour                  # for Hour of Day classifier
        dayTrain = load.index[j].dayofweek              # for DoW classifier
        dayOfYear = load.index[j].timetuple().tm_yday   # for Day of Year classifier
        
        dummy = np.zeros((len(dayClassifier)))  
        for k in range(len(dayClassifier)):     
            dummy[k] = (dayTrain in dayClassifier[k]) 
        dow[j,0:2] = dummy
                       
    dowSeriesDict={}

    dowTrain=[]
    dowTest=[]

    for i in range(len(windows)):
        dowTrain.append(dow[windows[i,0]+168:windows[i,1]])
        dowTest.append(dow[windows[i,2]:windows[i,3]])
    dowTrain = np.array(dowTrain).reshape(-1,2)
    dowTest = np.array(dowTest).reshape(-1,2)
    #print("shape dow: train {0} test {1}".format(np.array(dowTrain).shape,np.array(dowTest).shape))
    dowSeriesDict['dowTrain'] = dowTrain
    dowSeriesDict['dowTest'] = dowTest

    return(dowSeriesDict)


# Function to define all load series for training / testing

# inputs:
# windows = start/end window hr index for each train/test period
# load = original complete load series
# loadSeriesList - list of training and testing window load series names (strings)
#                  must include both inputs (loads at t,t-24,t-168, etc) and target (yData) series
#                  may add more lagged loads if desired eg for t-27,t-28 etc
# loadSeriesOffsetsWinStart: time offset for start of each window e.g. 168 hrs
# loadSeriesLags: lags to use e.g. t-lag where lag = 24,168, etc
# loadSeriesWindows: window indices for each series e.g. [0,1] for train series, [2,3] for test series

def set_loadSeries(windows,load,loadSeriesList,loadSeriesOffsetsWinStart,loadSeriesLags,\
                   loadSeriesWindows,*args):

    #define list of tuples
    lTups = list(zip(loadSeriesList,loadSeriesOffsetsWinStart,loadSeriesLags,loadSeriesWindows))

    # initializing dict of pd.Series
    loadSeries = {s: pd.Series([],dtype='float64') for s in loadSeriesList}

    for i,(loadKey,loadVal) in enumerate(loadSeries.items()):
        #print(i,lTups[i],lTups[i][0],lTups[i][1],lTups[i][2],lTups[i][3])
        for j in range(len(windows)):
            loadVal = loadVal.append([load[windows[j,lTups[i][3][0]] + lTups[i][1] - lTups[i][2] :\
                                           windows[j,lTups[i][3][1]] - lTups[i][2]   ]]) # append all but 1st week of each period
            if 'print_output' in args:
                if (j==0):
                    print("series: {6}: append input load for windows({0},{1})+{2}-{3}:windows({0},{4})-{5}".format(j,lTups[i][3][0],lTups[i][1], lTups[i][2],lTups[i][3][1],lTups[i][2],loadKey))
        loadSeries[loadKey]=loadVal


    return(loadSeries)


# VERSION1: concatenate load (T-24,T-168 etc) and env series (temp, LoD etc) AND DOW CLASSIFIERS  
# V1 - set up for simple case  : load series + env + DoW classifier (1/0) channels all separate inputs
# Returns dataframe of input features

def  create_TrainPredict_df(seriesNames,loadSeries,envNames,envSeries,dowNames,dowSeries):
    
    loadSeriesUse = { key:value for key,value in loadSeries.items() if key in seriesNames}
    envSeriesUse = { key:value for key,value in envSeries.items() if key in envNames}
    dowSeriesUse = { key:value for key,value in dowSeries.items() if key in dowNames}

    df_Features = pd.DataFrame()

    for series, values in loadSeriesUse.items():
        df_Features = pd.concat([df_Features,pd.DataFrame(loadSeriesUse[series].values,columns=[series])],axis=1)
        #print("shape df_Features: {0}".format(df_Features.shape))
        
    # add envSeries
    for series, values in envSeriesUse.items():
        df_Features   = pd.concat([df_Features,  pd.DataFrame(envSeries[series],columns=[series])],axis=1)
        #print("shape df_Features after enVSeries added: {0}".format(df_Features.shape))
        
    #add DoW classifiers
    for series, values in dowSeriesUse.items():
        df_Features   = pd.concat([df_Features,  pd.DataFrame(dowSeries[series],columns=['dow1','dow2'])],axis=1)
 
    return(df_Features)


# In[45]:


# VERSION2 : CREATE 2+ VERSIONS OF EACH LOAD SERIES , SCALED BY DOW CLASSIFIERS eg weekdays/weekends 0/1
# returns DF of input features

def  create_TrainPredict_df_V2(seriesNames,loadSeries,envNames,envSeries,dowNames,dowSeries):
    
    loadSeriesUse = { key:value for key,value in loadSeries.items() if key in seriesNames}
    envSeriesUse = { key:value for key,value in envSeries.items() if key in envNames}
    dowSeriesUse = { key:value for key,value in dowSeries.items() if key in dowNames}

    #get length of first load series (all series should have same length..)
    values = loadSeries.values()
    value_iterator = iter(values)
    seriesLength = len(next(value_iterator))
    
    #number of DoW classes
    numDowClasses = dowSeries[dowNames[0]].shape[1]

    #initialize dfs
    df_Features = pd.DataFrame()
    df_dow = pd.DataFrame()
           
    #add DoW classifiers to temporary df 
    for series, values in dowSeriesUse.items():
        df_dow   = pd.concat([df_dow, pd.DataFrame(dowSeries[series],columns=['dow1','dow2'])],axis=1)

    # scale train load series by each DoW classifier to identify weekday/weekend load series
    for series, values in loadSeriesUse.items():
        if 'yData' in series:
            df_Features = pd.concat([df_Features,pd.DataFrame(loadSeriesUse[series].values,columns=[series])],axis=1)
        else:    
            for i,col in  enumerate(df_dow.columns):
                df_Features = pd.concat( [df_Features, 
                                          pd.DataFrame(loadSeriesUse[series].values*df_dow[col].values,\
                                                columns=[str(series)+col])],axis=1)
            
    # add envSeries
    for series, values in envSeriesUse.items():
        df_Features   = pd.concat([df_Features,  pd.DataFrame(envSeries[series],columns=[series])],axis=1)

    return(df_Features)


# In[46]:


# select as default training, test series list  ALL input features (load, env, dow classifier etc):

def get_TrainTestSeries(df_Train,df_Test):
    trainSeries=[]
    testSeries=[]

    for col1,col2 in zip(df_Train.columns,df_Test.columns):
        trainSeries.append(str(col1))
        testSeries.append(str(col2))

    trainSeries=trainSeries[1:]  #drop yData load 
    testSeries=testSeries[1:]    #drop yData load 
    
    return(trainSeries,testSeries)


# In[47]:


# FUNCTION to create Multilayer Perceptron model 

# kwargs : numHL : number of hidden layers (default 2), activation function (default relu)

def create_MLP(trainDataSeries,nodesPerHiddenLayer,numInputFeatures,**kwargs):
    import tensorflow as tf
    import keras
    from keras.models  import Sequential
    from keras.layers import Dense, LSTM  # Dropout, Flatten, Conv2D, MaxPPooling2D, etc



    mdl = Sequential(name="MLP model")   #https://keras.io/guides/sequential_model/
    
    if 'numHL' in kwargs:
        numHL = kwargs.get('numHL')
    else:
        numHL = 2
        
    if 'activation' in kwargs:
        activation = kwargs.get('activation')
    else:
        activation = 'relu'
        
    # first hidden layer, input tensor has dimension of y trainingData, 
    # default  Weights initialization - glorot uniform?  https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/
    mdl.add(Dense(units=nodesPerHiddenLayer, input_dim=numInputFeatures, activation=activation,name="layer1"))  

    if numHL > 1:
        for hl in range(2,numHL+1,1):
            # additional hidden layers
            mdl.add(Dense(units=nodesPerHiddenLayer, activation=activation,name="layer"+str(hl))) 

    # output layer
    mdl.add(Dense(units=1,name="layer"+str(numHL+1)))  # default linear (ie. None f(x) = x) activation function https://keras.io/api/layers/core_layers/dense/

    return(mdl) 


# In[48]:


# funtion to plot loss function


def historyPlot(history,**kwargs):
    
    fig = plt.figure(figsize = (12,4))
    ax1 = fig.add_subplot()

    # since were using MSE as loss funtion, MSE metric is redundant
    #for k,v in history.history.items():
    #    ax1.plot(v, label=str(k))
    ax1.plot(history.history['loss'], label='loss')  # if validation split
    ax1.plot(history.history['val_loss'], label='val loss')  # if validation split
    
    ax1.set_ylabel('MSE value')
    ax1.set_xlabel('No. epoch')
    ax1.grid(b=True, which='both', axis='both')

    ax1.legend(loc="upper left")
    #ax1.set_ylim(0,1000)
    plt.yscale('log')
    
    if all (k in kwargs.keys() for k in ('N','col')):
        ax1.set_title('MSE for aggregation level' +str(kwargs['N']) +', group '+str(kwargs['col']) )
   


# In[49]:


# shift predictions to align with hr/day indices ie unpack into actual time bands

def annPlotInputs(loadSeries,windows,annForecast):  #loadSeries = eg aggLoad
    train_predict_plot = np.empty_like(loadSeries)
    test_predict_plot = np.empty_like(loadSeries)
    train_predict_plot[:] = np.nan
    test_predict_plot[:] = np.nan
    
    lengthTrainSum,lengthTestSum = 0,0

    
    trainLengthInd = windows[0,1]-(windows[0,0]+168)
    predLengthInd =  windows[0,3]-windows[0,2]
    
    for i in range(len(windows)): 
        lengthTrain     = windows[i,1]-(windows[i,0]+168)
        lengthTest      = windows[i,3]-(windows[i,2])

        train_predict_plot[(windows[i,0]+168):windows[i,1]]  = np.squeeze(annForecast['trainPredict'][lengthTrainSum:lengthTrainSum+lengthTrain])
        test_predict_plot[windows[i,2]:windows[i,3]]  = np.squeeze(annForecast['testPredict'][lengthTestSum:lengthTestSum+lengthTest])

        lengthTrainSum += lengthTrain
        lengthTestSum += lengthTest

    return({'trainPredictPlot':train_predict_plot,'testPredictPlot':test_predict_plot})


# In[50]:


# plot baseline and predictions : focus on window transitions +/- 1 week

def annPlot(loadSeries,annForecastPlotSeries,windows,**kwargs):

    if 'numPlots' in kwargs:
        numPlots = kwargs.get('numPlots')
    else:
        numPlots = len(windows)
        
    if 'viewWin' in kwargs:
        viewWin = kwargs.get('viewWin')
        lhs = viewWin[0]
        rhs = viewWin[1]
    else:
        lhs=-168
        rhs=168
    
    fig = plt.figure(figsize=(12,numPlots*4))
    axs = []
        
    for i in range(numPlots):
        axs.append(fig.add_subplot(numPlots,1,i+1))
        axs[i].plot(loadSeries[windows[i,1]+lhs:windows[i,1]+rhs].index,loadSeries[windows[i,1]+lhs:windows[i,1]+rhs].values, label='Observed', color='r',alpha=0.3);
        axs[i].plot(loadSeries[windows[i,1]+lhs:windows[i,1]+rhs].index,annForecastPlotSeries['trainPredictPlot'][windows[i,1]+lhs:windows[i,1]+rhs], label='Prediction for Train Set', color='b', alpha=0.5);
        axs[i].plot(loadSeries[windows[i,1]+lhs:windows[i,1]+rhs].index,annForecastPlotSeries['testPredictPlot'][windows[i,1]+lhs:windows[i,1]+rhs], label='Prediction for Test Set', color='g');
        axs[i].legend(loc='best');
        
        if 'figTitle' in kwargs:
            figTitle = kwargs.get('figTitle')
            axs[i].set_title(figTitle,fontsize=18)
        else:
            axs[i].set_title('MLP ANN : 2 hidden layer with X nodes each : window'+str(i))
            

    fig.tight_layout()


# General for ANN and MLR outputs ..

# function to generate appropriate folder name for saving ANN run output files 
#kwargs:
# root   (path name)
# globPrefix - filename prefix to search for glob
# config
# aggregation_N = list of aggregation levels
# normalized

#args:
# 'MLR'

def get_save_folderName(paramsDict,*args,**kwargs):
    from pathlib import Path
    import re
    import os
    
    if 'root' in kwargs:
        root = kwargs.get('root')
    else:
        root = './intermediateData/'

    if 'globPrefix' in kwargs:
        globPrefix = kwargs.get('globPrefix')
    else:
        globPrefix = 'LCL_ANN*'

    if 'config' in kwargs:
        config = '_' + str(kwargs.get('config'))
    else:
        config='_'
        
    if 'aggregation_N' in kwargs:
        aggregation_N = kwargs.get('aggregation_N')
    else:
        aggregation_N = [5,10,20,30,40,50,75,100]
        
    if 'normalized' in kwargs:
        isNormalized = kwargs.get('normalized')
    else:
        isNormalized = False
        
    if not os.path.exists(root):
        os.makedirs(root)
        print("Directory " , root ,  " Created ")
        runID = 1
    else:
        # remove empty folders first - to minimize clutter
        folders = list(os.walk(root))[1:]

        for folder in folders:  
            if not folder[2]:         #check if empty
                os.rmdir(folder[0])

        #get list of folders with results and determine next run iteration ID for naming new save folder
        runList=[]
        for path in Path(root).rglob(globPrefix):
            #print(path.name)
            if 'run' in path.name:
                runList.append(int(path.name.partition("run")[-1].split('_')[0]))      
        #print("runList",runList)
        if len(runList) == 0:
            runID = 1
        else:
            runID = max(runList)+1  # get next run ID for folder naming
    
    #create folder name for saving results:
    #set indicator of which levels aggregated
    if len(aggregation_N)>= 8:
        if len(aggregation_N)>= 10:
            s_agg='ALLincl1'
        else:
            s_agg='ALL'
    else:
        s_agg = 'Nsubset'  # + str(min(aggregation_N))+'to'+ str(max(aggregation_N))    

    if 'MLR' in args:
        dirName = root + globPrefix[:-1]+ str(config)+'run'+str(runID)+'_'+s_agg + '_'+ str(paramsDict['nEnvTerms'])+'envTerms_'+str(paramsDict['nDoWterms'])+'DoWterms'
    else:
        if isNormalized==True:
            dirName = root + globPrefix[:-1]+ str(config) +'run'+str(runID)+'_'+s_agg + '_Normalised_'+'HL'+str(paramsDict['numHL'])+'_Nodes'+str(paramsDict['nNodes'])+'_Epochs'+str(paramsDict['nEpochs'])+'_BatchSize'+str(paramsDict['batchSize']) +'_nFeat'+str(len(paramsDict['trainSeries']))
        else:
            dirName = root + globPrefix[:-1]+ str(config) +'run'+str(runID)+'_'+s_agg +'_HL'+str(paramsDict['numHL'])+'_Nodes'+str(paramsDict['nNodes'])+'_Epochs'+str(paramsDict['nEpochs'])+'_BatchSize'+str(paramsDict['batchSize'])+'_nFeat'+str(len(paramsDict['trainSeries']))

    return(dirName)



# function to create folder to save output files
def create_save_folder(dirName):
    import os
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")


# function to get get MLR error dictionaries from specified folder
def get_MLRerrorDictionary(relPathName,**kwargs):
    from six.moves import cPickle as pickle

    if 'areaID' in kwargs:
        areaID = '_' + str(kwargs.get('areaID'))
    else:
        areaID=''

    inputFile = str(relPathName) + 'errorsAgg_dict' +str(areaID) + '.pickle'
    with open(inputFile, 'rb') as pickleFile:
        errorsAgg = pickle.load(pickleFile, encoding='latin-1')

    return (errorsAgg)


#function to get error dictionaries from specified folder (ANN)

def get_errorDictionary(relPathName,**kwargs):

    from six.moves import cPickle as pickle

    if 'areaID' in kwargs:
        areaID = '_' + str(kwargs.get('areaID'))
    else:
        areaID=''

    inputFile = str(relPathName) + 'errorsAggANN_dict' +str(areaID) + '.pickle'
    with open(inputFile, 'rb') as pickleFile:
        errorsAggANN_dict = pickle.load(pickleFile, encoding='latin-1')

    inputFile = str(relPathName) + 'errorsAggANNtrain_dict' +str(areaID) + '.pickle'
    with open(inputFile, 'rb') as pickleFile:
        errorsAggANNtrain_dict = pickle.load(pickleFile, encoding='latin-1')

    return (errorsAggANNtrain_dict, errorsAggANN_dict)


# function to flip nested errors dictionary inside out i.e.
# .....want KEY = ERROR Type, subKey = N level, subvalue = df for each N

def flip_dictionary(input_dict):
    from collections import defaultdict  # https://www.accelebrate.com/blog/using-defaultdict-python
    import pprint

    flipped_dict = defaultdict(dict)
    for key, val in input_dict.items():
        for subkey, subval in val.items():
            flipped_dict[subkey][key] = subval
    return (flipped_dict)

# funtion to merge ANN TEST and TRAIN error dictionaries

def merge_TrainTestErrors(errorsAggANN_flipped,errorsAggANNtrain_flipped):
    errorsAggANN_flipped_TestTrain = {}

    for errorType in errorsAggANN_flipped.keys():
        d={}
        for (k,v), (k2,v2) in zip(errorsAggANN_flipped[errorType].items(), errorsAggANNtrain_flipped[errorType].items()):
            #print ("Key1 {0}, V1 df shape {1},Key2 {2}, V2 df shape {3}".format(k,v.shape,k2,v2.shape))
            #print("V1 df head:\n {0}, v2 df head:\n {1}".format(v.head(2),v2.head(2)))
            v.index = ['ANN_MLP test errors']
            v2.index=['ANN_MLP train errors']
            df_=pd.DataFrame()
            df_ = pd.concat([v,v2],axis=0)
            #print(df_)
            d[k] = df_

        errorsAggANN_flipped_TestTrain[errorType] = d

    return(errorsAggANN_flipped_TestTrain)

#calculate percentiles
def get_error_percentiles(errorsAggANN_flipped_TestTrain,dfRowName,**kwargs):
    #dfRowName = 'ANN_MLP train errors' etc

    if 'percentiles' in kwargs:
        percentiles = kwargs.get('percentiles')
    else:
        percentiles=[5,25,50,75,95]    #defaults

    columnNames=[str(p) for p in percentiles]
    df = pd.DataFrame()
    N = []

    for errorType in errorsAggANN_flipped_TestTrain.keys():
        for (k,v) in errorsAggANN_flipped_TestTrain[errorType].items():
            #print("N:",k,"\n v values : \n",v.loc['ANN_MLP train errors'].values)   #,v.values[0])
            l=[]
            for p in percentiles:
                l.append(np.percentile(v.loc[dfRowName].values,p))
            N.append(int(k))
            #print(errorType,k,l)
            df = pd.concat([df,pd.DataFrame(np.array(l).reshape(-1,len(l)),index=[errorType]  )])

    df.columns = columnNames
    df['N'] = N

    return(df)


# extract input dictionary keys to generate list of series names for train , test

def getTrainTestSeriesNames(series):
    seriesTrain, seriesTest = [], []
    for key in series.keys():
        # print(key)
        if ('train' in key) or ('Train' in key):
            # print("train: ",key)
            seriesTrain.append(key)
        elif ('test' in key) or ('Test' in key):
            # print("test: ",key)
            seriesTest.append(key)
    return (seriesTrain, seriesTest)



# plot error percentiles from boxplot on separate figure
# inputs:
# percentiles dictionary (set of dictionaries - one for each config, each with train , test dfs )
# args:
# 'sns' - use seaborn for plots - best if plotting just median; else uses matplotlin plot
# kwargs:
# errorType (default : MAPE)
# optional style changes - dictionary
# input figure and axes objects if wish to append to existing figure or subplot - testing
# percSupressList if want to hide some percentiles (if 'sns', only plits 50% (median) anyways))
# labels
# figTitle
# subTitle
# keep **kwargs in function def line so can add further arguments

def plot_percentileTraces(percentilesDict, order, *args, errorType=None, styles=None, figure=None, axs=None,
                          percSupressList=[5, 25, 75, 95], labels=None, figTitle=None, subTitle=None, **kwargs):
    import math
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator, MaxNLocator)
    import seaborn as sns

    # get number of subplots/config eg test and train; set mid percentile index for mirroring of line styles etc
    for i, (k, v) in enumerate(percentilesDict.items()):
        if i == 0:
            nPlots = len(v.keys())
            if nPlots < 2:  # but ensure minimum of 2 subplots even if one empty for consistency between MLR and ANN plots i.e. ANN : plot train and test; MLR plot test only
                nPlots = 2
            for j, (series, df) in enumerate(v.items()):
                if j == 0:
                    numPerc = df.shape[1] - 1 - len(percSupressList)
                    midPercIdx = int(round_down(numPerc / 2))
                    # print("midPerc Idx {0}".format(midPercIdx))

    if styles is None:
        styles = {'lSty': ['dashdot', 'dotted', 'dashed', 'solid'],
                  'lAlpha': [0.3, 0.3, 0.6, 0.4],
                  'symS': [5, 5, 10, 30]}

    if 'sns' not in args:  # use matplotlib to plot percentiles using these styles
        lineStyles = styles['lSty'][midPercIdx - 1:] + styles['lSty'][midPercIdx - 1:][::-1][1:]
        lineAlphas = styles['lAlpha'][midPercIdx - 1:] + styles['lAlpha'][midPercIdx - 1:][::-1][1:]
        symSizes = styles['symS'][midPercIdx - 1:] + styles['symS'][midPercIdx - 1:][::-1][1:]
        midIdx = int(round_down(len(lineStyles) / 2))

        lineStyles = lineStyles[midIdx - midPercIdx:midIdx + midPercIdx + 1]
        lineAlphas = lineAlphas[midIdx - midPercIdx:midIdx + midPercIdx + 1]
        symSizes = symSizes[midIdx - midPercIdx:midIdx + midPercIdx + 1]
        # print("lineStyles: {0}\n Line Tranparency: {1}\n Symbol sizes {2}\n".format(lineStyles,lineAlphas,symSizes))
        if 'colset' not in kwargs:
            colSet = ['k', 'r', 'b', 'g', 'c', 'y', 'm', 'w']
        else:
            colSet = kwargs.get('colset')

    if errorType is None:
        errorType = 'MAPEs'

    if figure is None:  # OTHERWISE, append to existing figure
        fig = plt.figure(figsize=(18, int(math.ceil(nPlots / 2)) * 8))
    else:
        fig = figure

    if 'yLim' not in kwargs:
        yLim = [0.0, 0.6]
    else:
        yLim = kwargs.get('yLim')

    if axs is None:  # OTHERWISE, append to axes of input (existing) figure object
        axs = [fig.add_subplot(int(math.ceil(nPlots / 2)), 2, i + 1) for i in range(nPlots)]

    for c, (configDict, configDictVals) in enumerate(
            percentilesDict.items()):  # loop through configs (adding each config to all subplots)
        counter = 0  # reset (to avoid duplicate labelling)
        for i, (series, df) in enumerate(
                configDictVals.items()):  # loop through  dfs (subplot) for each config (e.g. train,test)
            # first, change series (key) name if for MLR (test rror dictionaries were created with 'percentiles' key name)
            if series == 'percentiles':
                old_key = series
                new_key = old_key + 'Test'
                configDictVals[new_key] = configDictVals.pop(old_key)
                series = new_key

            # only saved 'test' window errors for MLR - want to plot on right for comparison with ANN equivalent
            if (i == 0) and (('MLR' in str(configDict)) or ('mlr' in str(configDict))):
                # print("MLR config:",series)
                i = i + 1

            # get N list as tuple for x-axis - assumes first config has full N range - will be missing x-axis labels if NOT the case
            if c == 0:
                xLabels = tuple(df['N'].values)

            percentilesPlot = [str(p) for p in df.columns.to_list()[:-1] if p not in \
                               [str(el) for el in percSupressList]]  # get percentiles as list of df colnames

            if 'sns' not in args:
                df1 = df  # keep original df for scatter plot

            df = df[percentilesPlot + ['N']]
            df = pd.melt(df.loc[errorType], id_vars=['N'], var_name='percentile',
                         value_name='error')  # convert to narrow format df

            if 'sns' in args:  # use seaborn for plotting - will only display median (50% percentile) regardless of surpressList
                sns.lineplot(x='N', y='error', data=df, ax=axs[i], ci=None, markers=True, dashes=False, \
                             palette=sns.color_palette('RdBu_r', n_colors=len(percentilesDict)), \
                             marker="o", markersize=6, label=labels[
                        c])  # color=configCol,hue='percentile',style='percentile'  - these add too many items to legend and mess up colours

            else:  # if want to plot more percentiles with different line styles etc
                configCol = colSet[c]

                for j, p in enumerate(percentilesPlot):  # loop through percentile bands for subplot
                    if j == midPercIdx:
                        axs[i].plot(df1.loc[errorType]['N'].values, df1.loc[errorType][str(p)].values,
                                    alpha=lineAlphas[j], \
                                    linestyle=lineStyles[j], color=configCol, label=labels[c])
                    else:
                        axs[i].plot(df1.loc[errorType]['N'].values, df1.loc[errorType][str(p)].values,
                                    alpha=lineAlphas[j], \
                                    linestyle=lineStyles[j], color=configCol)  # label='Percentile '+str(p))
                    axs[i].scatter(df1.loc[errorType]['N'].values, df1.loc[errorType][str(p)].values, s=symSizes[j],
                                   alpha=0.6,
                                   color=configCol)
                axs[i].legend()

            axs[i].set_xlabel('aggregation level, N')
            axs[i].set_ylabel(str(errorType[:-1]) + ' error %')

            # set subplot artist properties
            axs[i].set(xscale='log')

            axs[i].set_ylim(yLim[0], yLim[1])
            axs[i].set_xlim(1, 100)
            axs[i].set_title(str(series))  #, fontsize=14)  # title for subplot

            # set y axis major grid spacing
            axs[i].yaxis.set_major_locator(MultipleLocator(0.1))
            axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            # For the minor ticks, use no labels; default NullFormatter.
            axs[i].yaxis.set_minor_locator(MultipleLocator(0.05))

            axs[i].grid(b=True, which='major', axis='both', alpha=0.5)
            axs[i].grid(b=True, which='minor', axis='y', alpha=0.3)

            axs[i].set_xticks(xLabels)
            axs[i].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axs[i].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            axs[i].get_yaxis().set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

            #instead use rcparams settings
            #for item in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
            #    item.set_fontsize(14)

    if figTitle is not None:
        fig.suptitle(figTitle,  y=1.0)   #fontsize=18,
    if subTitle is not None:
        fig.text(0.5, 0.94, subTitle,  ha='center')    #fontsize=14,

    return (fig, axs);


# function to do all prep work for box plotting: errors by aggregation etc

# 1. Extract errors for requested N level                            e.g. errorsAggLCLallModels
# 2. Extract user grp id and attach to corresponding error data      e.g. grpAggLoad_N_dictLCL
# 3. Categorize user grps into bins by ANN forecast error
# 4. Select random user group from specified bin (e.g. low error, high error)
# 5. Get specific user IDs in said group (e.g. for labelling)
# 5. Extract load data for said user grp                             e.g. grpAggLoad_N_dictLCL

# sample input files :
# mlrTest_dictLCL           : mlr & SN forecast profiles
# annTrainPredict_dictLCL  :  ANN forecast profiles
# errorsAggLCLallModels  :  MAPE (or errorType selected when created) errors for all models
# grpAggLoad_N_dictLCL   :  aggregated grp loads and userIDs for each grp

def getForecastByGrp(N, nBins, nGrpsToPlot, category, MLRfc, ANNfc, errors_df, loadDicts):
    df_1 = errors_df.loc[errors_df['N'] == N].T
    df_2 = loadDicts[N]['randIDgrps_N']
    df_2 = df_2.set_index([pd.Index(int(N) * ['userID'])])
    df_3 = pd.concat([df_1, df_2]).T.sort_values(
        by='ANN_MLP')  # concat errors values and userID info and sort by error values
    # print(df_3['ANN_MLP'].median())    # check median stats and compare with boxplot (sense check)

    # categorize errors into bins
    bin_labels = ['Cat{}'.format(x) for x in range(nBins)]
    binned, retBins = pd.qcut(df_3['ANN_MLP'], q=nBins, retbins=True, precision=2, labels=bin_labels)
    print("Bins: {0}".format(retBins))
    df_3['category'] = binned

    catCounts = df_3['category'].value_counts()  # df with category as index and number of grps in each category

    # get random grp from category
    randGrpIDs = []
    for j in range(nGrpsToPlot):
        grpID = df_3.loc[df_3['category'] == category].index[
            np.random.randint(catCounts.loc[catCounts.index == category].values)]
        randGrpIDs.append(grpID)

    randGrpIDs = [val for sublist in randGrpIDs for val in sublist]  # flatten list
    # print(randGrpIDs)
    # get userID infor for this group from temporary df df_2 above
    uIDsList = df_2[randGrpIDs].values.tolist()
    uIDsList = [val for sublist in uIDsList for val in sublist]

    # to get agg load in aggregation grp for selected grp
    df_4 = loadDicts[N]['grpAggLoad_N']
    load_df = df_4[randGrpIDs]

    models = {'ANN': np.squeeze(ANNfc[str(N)]['testPredictSeries'][randGrpIDs].values),
              'quadratic': np.squeeze(MLRfc[str(N)]['quadraticTest'][randGrpIDs].values),
              'unaware': np.squeeze(MLRfc[str(N)]['unawareTest'][randGrpIDs].values),
              'SN24': np.squeeze(MLRfc[str(N)]['SN24test'][randGrpIDs].values),
              'SN168': np.squeeze(MLRfc[str(N)]['SN168test'][randGrpIDs].values)}

    return (load_df, uIDsList, randGrpIDs, models);


# simplified version of getForecastByGrp - this one gets a specific forecast for a supplied grp name instead of random set of grps
# 1. Extract errors for requested N level                            e.g. errorsAggLCLallModels
# 2. Extract user grp id and attach to corresponding error data      e.g. grpAggLoad_N_dictLCL
# 3. Get specific user IDs in said group (e.g. for labelling)
# 4. Extract load data for said user grp                             e.g. grpAggLoad_N_dictLCL

# sample input files :
# mlrTest_dictLCL           : mlr & SN forecast profiles
# annTrainPredict_dictLCL  :  ANN forecast profiles
# errorsAggLCLallModels  :  MAPE (or errorType selected when created) errors for all models
# grpAggLoad_N_dictLCL   :  aggregated grp loads and userIDs for each grp

def getForecastByGrpName(N, MLRfc, ANNfc, errors_df, loadDicts, grpID):
    df_1 = errors_df.loc[errors_df['N'] == N].T
    df_2 = loadDicts[N]['randIDgrps_N']
    df_2 = df_2.set_index([pd.Index(int(N) * ['userID'])])
    df_3 = pd.concat([df_1, df_2]).T  # concat errors values and userID info

    # get userID infor for this group from temporary df df_2 above
    # print(df_2[grpID])
    # uIDsList = df_2[grpID].values.tolist()
    # uIDsList = [val for sublist in uIDsList for val in sublist]

    # to get agg load in aggregation grp for selected grp
    df_4 = loadDicts[str(N)]['grpAggLoad_N']
    load_df = df_4[str(grpID)]

    models = {'ANN': np.squeeze(ANNfc[str(N)]['testPredictSeries'][grpID].values),
              'quadratic': np.squeeze(MLRfc[str(N)]['quadraticTest'][grpID].values),
              'unaware': np.squeeze(MLRfc[str(N)]['unawareTest'][grpID].values),
              'SN24': np.squeeze(MLRfc[str(N)]['SN24test'][grpID].values),
              'SN168': np.squeeze(MLRfc[str(N)]['SN168test'][grpID].values)}

    return (load_df, models);  # uIDsList,


# define function to save all MLR result to disk
# ParamsDict          # mlr model params used (for naming)
# root                # path root
# globPrefix          # prefix for file search
# config              : config identifier (string)
# aggregation_N       : array of N values
# resultsDictionaries # dictionaries of forecasts, loads, errors , userid grps etc
def saveMLRconfigXresults(paramsDict,root,globPrefix,config,aggregation_N,resultsDictionaries):

    dirName = get_save_folderName(paramsDict,'MLR',root=root,globPrefix=globPrefix,\
                              config=config,aggregation_N=aggregation_N)

    create_save_folder(dirName)

    #save dictionaries to pickle file: TEST errors, TRAIN errors,Forecast series,N level aggregate load series
    pklsToSave=[]
    for dictName in resultsDictionaries.keys():
        pickleFileName = '/' + str(dictName) +'.pickle'
        pklsToSave.append(pickleFileName)

    for pklfile,dictName,dict in zip(pklsToSave,resultsDictionaries.keys(),resultsDictionaries.values()):
        with open(dirName+pklfile, 'wb') as f:
            pickle.dump(dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("dictionary {0} saved to pickle file {1}".format(dictName,pklfile))

    #save parameters used to json file
    with open(dirName+'/mlrParamsDict.json', 'w') as f:
        json.dump(paramsDict, f)
        f.close()

# define function to save all ANN results (dictionaries) to disk
# ParamsDict          # ANN model params used (for naming)
# root                # path root
# globPrefix          # prefix for file search
# config              : config identifier (string)
# aggregation_N       : array of N values
# normalizedBool      # string yes/No to indicate if minmax normalisation was used
# resultsDictionaries # dictionaries of forecasts, loads, errors , userid grps etc

def saveAnnConfigXresults(paramsDict,root,globPrefix,config,aggregation_N,normalizedBool,resultsDictionaries):

    dirName = get_save_folderName(paramsDict,root=root,globPrefix=globPrefix,\
                              config=config,aggregation_N=aggregation_N,normalized=normalizedBool)

    create_save_folder(dirName)

    #save dictionaries to pickle file: TEST errors, TRAIN errors,Forecast series,N level aggregate load series
    pklsToSave=[]
    for dictName in resultsDictionaries.keys():
        pickleFileName = '/' + str(dictName) +'.pickle'
        pklsToSave.append(pickleFileName)

    for pklfile,dictName,dict in zip(pklsToSave,resultsDictionaries.keys(),resultsDictionaries.values()):
        with open(dirName+pklfile, 'wb') as f:
            pickle.dump(dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("dictionary {0} saved to pickle file {1}".format(dictName,pklfile))

    #save parameters used to json file
    with open(dirName+'/paramsDict.json', 'w') as f:
        json.dump(paramsDict, f)
        f.close()


# function to get agg load and errors by aggregation group
# 1. Extract errors for requested N level                            e.g. errorsAggLCLallModels
# 2. Extract user grp id and attach to corresponding error data      e.g. grpAggLoad_N_dictLCL

# 3. Extract load data for said user grp                             e.g. grpAggLoad_N_dictLCL

# sample input files :
# LCL:  errorsAggLCLallModels,grpAggLoad_N_dictLCL

def getLoadByGrp(N, errors_df, loadDicts):
    df_1 = errors_df.loc[errors_df['N'] == N].T
    df_2 = loadDicts[N]['randIDgrps_N']
    df_2 = df_2.set_index([pd.Index(int(N) * ['userID'])])
    df_3 = pd.concat([df_1, df_2]).T  # concat errors values and userID info

    # get userID infor for this group from temporary df df_2 above
    uIDsList = df_2.values.tolist()
    uIDsList = [val for sublist in uIDsList for val in sublist]

    # to get agg load in aggregation grp for selected grp
    load_df = loadDicts[N]['grpAggLoad_N']

    return (load_df, uIDsList);

# other misc functions

# Function to calculate time difference between two datetime objects.
def diff_datetime(start,end):
    import datetime
    import time
    from datetime import timedelta
    datetimeFormat = '%Y-%m-%d %H:%M:%S'
    diff = datetime.datetime.strptime(end, datetimeFormat)\
    - datetime.datetime.strptime(start, datetimeFormat)
    #print("Days:", diff.days)
    #print("Hours:", diff.seconds/3600)
    hours = (diff.days*24)+(diff.seconds/3600)
    return (int(hours))

