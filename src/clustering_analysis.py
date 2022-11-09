# imports packages

from collections import Counter, defaultdict
import copy
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import unravel_index
import pandas as pd
from pandas import *
from scipy import linalg
from scipy.stats.stats import pearsonr
import seaborn as sns
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path
import seaborn as sns

theGreen = sns.color_palette("Paired")[3]
theBlue = sns.color_palette("Paired")[1]

def normalise_to_normal(aNormalisedData,aOriginalData):
    myNormalisedDataPointsInOriginalBasis = copy.deepcopy(aNormalisedData)
    for k in aNormalisedData.keys():
        myMean = np.mean(aOriginalData[k])
        myStd  = np.std(aOriginalData[k])
        myNormalisedDataPointsInOriginalBasis[k] *= myStd
        myNormalisedDataPointsInOriginalBasis[k] += myMean
    return myNormalisedDataPointsInOriginalBasis

def plot_quantiles(aPlot,aY):
    
    aPlot.plot(range(24),np.median(aY,axis=1),color='k',linewidth=1)
    myQuantilesToPlot = [25,50,74]
    myColours = sns.color_palette("tab20c")
    for i in range(len(myQuantilesToPlot)):
        q = myQuantilesToPlot[i]
        myUpperQuant = np.array(np.percentile(aY, 100.0 - q/2.0, axis=1), dtype=float)
        myBottomQuant = np.array(np.percentile(aY, q/2.0, axis=1), dtype=float)
        aPlot.fill_between(range(24),myUpperQuant,myBottomQuant,color=myColours[i])

def get_data(aFile):
  
    # imports the data
    myData = pd.read_csv(aFile)
    myData.drop('WindOff_WM', inplace=True, axis=1)

    # normalises the data: each column is then zero mean and 1 std 
    myNormalisedData = (myData - myData.mean()) / myData.std()

    return myData, myNormalisedData

def get_data_with_peak_info(aFile):
  
    # imports the data
    myData = pd.read_csv(aFile)
    myData.drop('WindOff_WM', inplace=True, axis=1)

    # normalises the data: each column is then zero mean and 1 std 
    myNormalisedData = (myData - myData.mean()) / myData.std()

    #############
    # Peak Elec #
    #############

    # gets day of peak electricity
    myDayIndexPeakElec = np.argmax(np.max(myData['Elec'].values.reshape(365,24), axis=1)) # equivalent to: np.unravel_index(np.argmax(myData['Elec']),(365,24))[0]

    # gets the data corresponding to the day with national peak elec demand
    # myDataDayNationalPeakElecDemand = [myData[i].values.reshape(365,24)[myDayIndexPeakElec] for i in myData]
    i1 = np.ravel_multi_index((myDayIndexPeakElec,0), (365,24))
    i2 = np.ravel_multi_index((myDayIndexPeakElec+1,0), (365,24))
    myDataDayNationalPeakElecDemand = myData[:][i1:i2]

    ############
    # Peak Gas #
    ############

    # gets the national gas demand based on the gas demand of each zone
    myColGasDemand = ['EA', 'EM', 'NE', 'NO', 'NT', 'NW', 'SC', 'SE', 'SO', 'SW','WM', 'WN', 'WS']
    myColGasDemand = ['GasDem_'+s for s in myColGasDemand]
    myNationalGasDemand = sum(myData[i] for i in myData if i in myColGasDemand).values.reshape(365,24)

    # gets the day index with national peak gas demand
    myDayIndexPeakNationalGasDemand = np.argmax(np.max(myNationalGasDemand, axis=1))

    # gets the data corresponding to the day with national peak gas demand
    myDataDayNationalPeakGasDemand = [myData[i].values.reshape(365,24)[myDayIndexPeakNationalGasDemand] for i in myData]
    j1 = np.ravel_multi_index((myDayIndexPeakNationalGasDemand,0), (365,24))
    j2 = np.ravel_multi_index((myDayIndexPeakNationalGasDemand+1,0), (365,24))
    myDataDayNationalPeakGasDemand = myData[:][j1:j2]

    ################
    # Data no peak #
    ################

    # gets data without day with peak electricity or gas demand
    myDataNoPeak = myData.drop(np.concatenate((range(i1,i2),range(j1,j2))),axis=0,inplace=False)
    # Old: myDataNoPeak = np.delete(data_days,[myDayIndexPeakElec,myDayIndexPeakNationalGasDemand],axis=1)

    # gets the normalised data without day with peak electricity or gas demand
    myNormalisedDataNoPeak = myNormalisedData.drop(np.concatenate((range(i1,i2),range(j1,j2))),axis=0,inplace=False)

    return myData, myNormalisedData, myDataNoPeak, myNormalisedDataNoPeak, myDayIndexPeakElec, myDataDayNationalPeakElecDemand, myDayIndexPeakNationalGasDemand, myDataDayNationalPeakGasDemand

######################
# Clustering methods #
######################

def get_array_data_points(aData):
    """returns a vector where each element is a data point, i.e. a vector with all the data.
    The `aData` input is a vector where each index corresponds to an attribute

    :aData: TODO
    :returns: TODO

    """
    return np.concatenate(aData, axis=1)

def my_quantile(x):
    return np.quantile(x, 0.4)

# sorts the clusters attribution
def get_sorted_attribution(aClusterAttribution):
    
    l = list(aClusterAttribution)
    l.sort(reverse=False, key=my_quantile)
    return np.array(l)

# ------- #
# K-means #
# ------- #

def get_kmeans_clusters(aData,aNClusters,aSeed):

    # fixes the seed
    np.random.seed(aSeed)

    # returns an array where each index is a day and contains a vector which corresponds to all the attributes are concatenated, returns data (365,24*nAttributes)
    myDataPoints = aData
    if type(myDataPoints) is list:
        myDataPoints = get_array_data_points(aData)

    # runs the K-means clustering
    kmeans = KMeans(aNClusters,init='k-means++', max_iter=10**3).fit(myDataPoints)
    
    # gets the centroids of each cluster
    centroids = kmeans.cluster_centers_ 
    
    # gets the inertia, i.e. sum of squared distances of samples to their closest cluster centre, weighted by the sample weights if provided.
    inertia = kmeans.inertia_
    
    # gets an array where for each index i, its return the days index associated to the cluster i
    myClusterAttribution = [np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)]

    return centroids, myClusterAttribution

def get_summary(aClusterAttribution):
    
    # gets the number of clusters
    myNClusters = len(aClusterAttribution)

    # gets the number of days in each cluster
    myClusterNDays = [len(aClusterAttribution[i]) for i in range(myNClusters)]

    # scales the weight of each cluster from 0 to 1
    myClusterWeights = [myClusterNDays[i]/365.0 for i in range(len(myClusterNDays))]
    
    # creates a dataframe with the representative day of each cluster
    myClusters= pd.DataFrame({"Cluster_iD":range(myNClusters),"Weights": pd.Series(myClusterWeights),"Number_days": pd.Series(myClusterNDays),"Days_In_Cluster": pd.Series(aClusterAttribution)})
    
    # sort Clusters chronologically
    myClusters = myClusters.sort_values(by=['Weights'], ascending=[False])

    # reindex the cluster such that the one with index 1 corresponds to the first representative day
    myClusters["Cluster_iD"]=range(myNClusters)

    return myClusters 

def representative_day_kmeans(aCentroids, aClusterAttribution, aData, aDayIndexPeakElec, aDayIndexPeakNationalGasDemand):

    # gets the number of clusters
    myNClusters = len(aClusterAttribution)

    # gets the number of attributes
    myNAttributes = int(len(aCentroids[1])/24.0)

    ############
    # Clusters #
    ############

    # get details of centroids for each column
    myDaysCentroids = [aCentroids[:,i*24:24*(1+i)] for i in range(myNAttributes)]

    # reduces the data of each attribute by computing its mean over the 24 hours. This is done for each cluster. np.vstack allows to put everything back in a 2D array instead of an array with sub-array.
    myCentroidsMeans = np.vstack([myDaysCentroids[i].mean(axis=1) for i in range(myNAttributes)])

    ########
    # Days #
    ########

    # reduces the data of each attribute by computing its mean over the 24 hours. This is done for each day. np.vstack allows to put everything back in a 2D array instead of an array with sub-array.
    myDayMeans = np.vstack([aData[i].mean(axis=1) for i in range(myNAttributes)])

    ############################################
    # Cluster's representative day attribution #
    ############################################

    # computes the index of the representative day when there is no the peak elec and gas day
    myClusterRepDays = []
    for clusterIndex in range(myNClusters):
        # X = np.abs(aCentroidMeans.transpose()[clusterIndex]-aItemMeans.transpose()[aClusterAttribution[clusterIndex]])
        X = np.square(myCentroidsMeans.transpose()[clusterIndex]-myDayMeans.transpose()[aClusterAttribution[clusterIndex]])
        myClusterRepDays.append(aClusterAttribution[clusterIndex][X.sum(axis=1).argmin()])

    # updates the index of the representative day when adding the peak elec and gas day
    for i in range(len(myClusterRepDays)):
        shift = 0
        day = myClusterRepDays[i]
        if day >= aDayIndexPeakElec:
            shift += 1
        if day >= aDayIndexPeakNationalGasDemand and aDayIndexPeakElec != aDayIndexPeakNationalGasDemand:
            shift += 1
        myClusterRepDays[i] = day + shift  
    
    # gets the number of data points in each cluster
    myClusterAttribution = aClusterAttribution

    # gets the number of days in each cluster
    myClusterNDays = [len(aClusterAttribution[i]) for i in range(myNClusters)]

    # adds the peak elec and gas days
    myClusterRepDays.append(aDayIndexPeakElec)
    myClusterAttribution.append([aDayIndexPeakElec])
    myClusterNDays.append(1)
    myClusterRepDays.append(aDayIndexPeakNationalGasDemand)
    myClusterAttribution.append([aDayIndexPeakNationalGasDemand])
    myClusterNDays.append(1)

    # scales the weight of each cluster from 0 to 1
    myClusterWeights = [myClusterNDays[i]/365.0 for i in range(len(myClusterNDays))]
    
    # creates a dataframe with the representative day of each cluster
    myClusters= pd.DataFrame({"Cluster_iD":range(myNClusters+2),"Repres_Days":myClusterRepDays,"Weights": pd.Series(myClusterWeights),"Number_days": pd.Series(myClusterNDays),"Days_In_Cluster": pd.Series(myClusterAttribution)})

    # sort Clusters chronologically
    myClusters = myClusters.sort_values(by=['Repres_Days'])

    # reindex the cluster such that the one with index 1 corresponds to the first representative day
    myClusters["Cluster_iD"]=range(myNClusters+2)

    return myClusters 

# -----------------#
# Gaussian Mixture #
# -----------------#

def get_gaussian_mixture_clusters_attribution(aData,aNClusters,aSeed,aMethod):
    
    # fixes the seed
    np.random.seed(aSeed)

    myDataPoints = aData

    myFittedMixtureModel = 1
    if aMethod == 'gmm':
        print("Fit: Gaussian Mixture")
        # Gaussian Mixture
        myFittedMixtureModel = mixture.GaussianMixture(n_components=aNClusters, covariance_type="full", max_iter=10**6).fit(myDataPoints)
    elif aMethod == 'dpgmm':
        print("Fit: Bayesian Gaussian Mixture with a Dirichlet process prior")
        # Bayesian Gaussian Mixture with a Dirichlet process prior
        myFittedMixtureModel = mixture.BayesianGaussianMixture(n_components=aNClusters, covariance_type="full", max_iter=10**6).fit(myDataPoints)
    else:
        myErrorMessage = "Method '{}' not found".format(aMethod)
        raise NameError(myErrorMessage)

    # the number of clusters after the fitting
    myNClusters = myFittedMixtureModel.n_components
    
    # gets the cluster attribution of each data point
    myLabels = myFittedMixtureModel.predict(myDataPoints)
    
    # gets an array where for each index i, its return the days index associated to the cluster i
    myClusterAttribution = [np.where(myLabels == i)[0] for i in range(myNClusters)]

    return get_sorted_attribution(myClusterAttribution)

def get_gaussian_mixture_clusters(aData,aNClusters,aSeed,aMethod):
    
    # fixes the seed
    np.random.seed(aSeed)

    # returns an array where each index is a day and contains a vector which corresponds to all the attributes are concatenated, returns data (365,24*nAttributes)
    myDataPoints = get_array_data_points(aData)

    # gets the number of attributes
    nAttributes = int(len(myDataPoints[1])/24.0)

    myFittedMixtureModel = 1
    if aMethod == 'gmm':
        print("Fit: Gaussian Mixture")
        # Gaussian Mixture
        myFittedMixtureModel = mixture.GaussianMixture(n_components=aNClusters, covariance_type="full", max_iter=10**6).fit(myDataPoints)
    elif aMethod == 'dpgmm':
        print("Fit: Bayesian Gaussian Mixture with a Dirichlet process prior")
        # Bayesian Gaussian Mixture with a Dirichlet process prior
        myFittedMixtureModel = mixture.BayesianGaussianMixture(n_components=aNClusters, covariance_type="full", max_iter=10**6).fit(myDataPoints)
    else:
        myErrorMessage = "Method '{}' not found".format(aMethod)
        raise NameError(myErrorMessage)

    # the number of clusters after the fitting
    myNClusters = myFittedMixtureModel.n_components
    
    # gets the cluster attribution of each data point
    myLabels = myFittedMixtureModel.predict(myDataPoints)
    
    # gets an array where for each index i, its return the days index associated to the cluster i
    myClusterAttribution = [np.where(myLabels == i)[0] for i in range(myNClusters)]

    # gets the weights of each cluster
    myClusterWeights = myFittedMixtureModel.weights_

    # gets the number of days associated to each cluster
    myClusterNDays = [len(myClusterAttribution[i]) for i in range(myNClusters)]

    # gets the mean of each cluster
    myMeans = [myFittedMixtureModel.means_[i] for i in range(len(myFittedMixtureModel.means_))]

    # gets the covariance matrix of each cluster
    myCovariances = [myFittedMixtureModel.covariances_[i] for i in range(len(myFittedMixtureModel.covariances_))]

    #############
    # Dataframe #
    #############

    # creates a dataframe with the representative day of each cluster
    myClusters= pd.DataFrame({"Cluster_iD":range(myNClusters),"Weights": pd.Series(myClusterWeights),"Number_days": pd.Series(myClusterNDays),"Days_In_Cluster": pd.Series(myClusterAttribution),"Means":pd.Series(myMeans),"Covariances":pd.Series(myCovariances)})

    return myClusters

def get_correlation_figure(aColumns,aCorrArray,aPower):

    x, y = np.mgrid[range(len(aColumns)), range(len(aColumns))]
    z = np.multiply(np.sign(aCorrArray),np.abs(np.power(aCorrArray,aPower)))

    # Generates the plot
    fig, ax = plt.subplots(figsize=(30, 30), dpi=50)

    # ticks
    ticks = [i for i in range(len(aColumns))]

    # Set the ticks and ticklabels for all axes
    plt.setp(ax, xticks=ticks, xticklabels=aColumns, yticks=ticks, yticklabels=aColumns)
    plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees

    c = ax.pcolor(x, y, z, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_title('pcolor')
    # ax.grid(linestyle='-', color='k', linewidth=1)
    fig.colorbar(c, ax=ax)

    return fig

def get_correlation_figure_given_data(aData):

    columns = range(aData.shape[0])

    myCorrelationArray = []
    for i in range(len(columns)):
        myCorrelationArrayElement = []
        for j in range(len(columns)):
            r, pvalue = pearsonr(aData[i], aData[j])
            myCorrelationArrayElement.append(r)
        myCorrelationArray.append(myCorrelationArrayElement)

    return get_correlation_figure(columns,myCorrelationArray,1)

def get_data_in_pca_basis(aX,aPCA):
    return aPCA.transform(aX)

def get_data_in_pca_basis_second_def(aX,aPCA):
    return np.dot(aX - aPCA.mean_, aPCA.components_.T)

def get_data_in_original_basis(aZ,aPCA):
    return np.dot(aZ, aPCA.components_) + aPCA.mean_

def get_plot_variance_ratio(aDictLayer,aAttributeNameToIllustrate):

    # gets the pca
    pca = aDictLayer[aAttributeNameToIllustrate]['PCA']

    fig = plt.figure(figsize=(30, 30), dpi=50)
    ax = fig.add_subplot(2, 1, 1)
    line, = ax.plot(pca.explained_variance_ratio_, color='blue', lw=2)
    plt.xlabel('Principal component', size=30)
    plt.ylabel('Variance Ratio', size=30)
    plt.xticks(np.arange(0, pca.explained_variance_ratio_.shape[0], 10))
    plt.grid(color='k', linestyle='-', linewidth=1)
    ax.set_yscale('log')
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    return fig

def get_plot_comparison_days(aDaysVec,aAttributeNames,aOriginalDataDict,aRecoveredDataDict):

    # gets the original data related to the attribute to plot
    myOriginalData = [aOriginalDataDict[attr] for attr in aAttributeNames]
    myOriginalData = np.concatenate(myOriginalData, axis=1)
    
    # gets the recovered data after the recovering of the data
    myRecoveredData = [aRecoveredDataDict[attr] for attr in aAttributeNames]
    myRecoveredData = np.concatenate(myRecoveredData, axis=1)

    fig, axs = plt.subplots(len(aDaysVec),2,figsize=(12,int(len(aDaysVec)*3)))
    fig.suptitle('Graphical comparison of the original and recovered data',size=20)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
    
    for i in range(len(aDaysVec)):
        axs[i,0].plot(myOriginalData[aDaysVec[i]])
        axs[i,0].set_title('Original data: Day {}'.format(aDaysVec[i]))
        axs[i,1].plot(myRecoveredData[aDaysVec[i]])
        axs[i,1].set_title('Recovered data: Day {}'.format(aDaysVec[i]))
    
    return fig

def get_attribute_correlation_plot(aDataDict,aNameToSave):
    myData = None
    for key in aDataDict.keys():
        if (myData is None):
            myData = aDataDict[key]
        else:
            myData = np.hstack((myData,aDataDict[key]))
    
    myFigCorr = get_correlation_figure_given_data(myData.T)
    plt.savefig(aNameToSave,bbox_inches='tight')
    return myFigCorr

def generates_plot_days_cluster_attribution(aClusterAttribution,aClusterLabels,aData,aAttributesToPlot,aOutputFolder,aImageFormat,aFileNameExtension,aPlotPerAttribute,aPlotType, aPlotTitle='', aDPI=20, aFigSize=(20, 10), aNBins=50):
    
    # gets the number of clusters
    myNClusters = len(aClusterAttribution)
        
    # gets the cluster attribution sorted in decreasing number of days in each cluster
    mySortedClusterAttribution = get_sorted_attribution(aClusterAttribution)

    # Creating histogram
    fig, ax = plt.subplots()
    ax.hist(mySortedClusterAttribution, bins=aNBins, stacked=True, histtype='bar', range=(0,364), density=False)
    # ax.hist(mySortedClusterAttribution, label=aClusterLabels, bins=aNBins, stacked=True, histtype='bar', range=(0,364), density=False)

    # # loop on the cluster
    # for i in range(myNClusters):
    #     # plots histogram with days 
    #     ax.hist(mySortedClusterAttribution[i], label = aClusterLabels[i], alpha = 0.5, edgecolor='black', density=False, bins=aNBins, range=(0,364))

    plt.title(aPlotTitle)
    plt.xlabel(r'Year')
    plt.ylabel(r'# of days')
    plt.xticks([15,104,195,287], [r'Jan 15',r'Apr 15',r'Jul 15',r'Oct 15'])
    plt.xlim((0, 364))
    plt.ylim((0, 17.5))

    # Put a legend below current axis
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=len(aClusterLabels))
    # plt.legend(loc='upper right')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig('outputs/images/cluster/cluster_' + aFileNameExtension + '.eps', format='eps', bbox_inches='tight')
    plt.savefig('outputs/images/cluster/cluster_' + aFileNameExtension + '.png', dpi=aDPI, bbox_inches='tight')

def generates_plot_clustering(aClusterAttribution,aData,aAttributesToPlot,aOutputFolder,aImageFormat,aFileNameExtension,aPlotPerAttribute,aPlotType):

    # put the output folder as lower case
    myOutputFolder = aOutputFolder.lower()

    # creates the output folder if it does not exist
    Path(myOutputFolder).mkdir(parents=True, exist_ok=True)
    
    # defines the cmap
    min_val, max_val = 0.2,1.0
    n = 100
    orig_cmap = plt.cm.Blues
    colors = orig_cmap(np.linspace(min_val, max_val, n))
    mycmap = mpl.colors.LinearSegmentedColormap.from_list("mycmap", colors)
    
    if aPlotPerAttribute:

        ####################################
        # Generates one plot per attribute #
        ####################################
        
        # gets the number of clusters
        myNClusters = len(aClusterAttribution)
        
        # gets the number of lines for the plot
        myNCols  = 2
        myNLines = int(np.ceil(myNClusters / float(myNCols)))  
        
        # gets the cluster attribution sorted in decreasing number of days in each cluster
        mySortedClusterAttribution = get_sorted_attribution(aClusterAttribution)
        
        for att_loc in aAttributesToPlot: # loop on the attributes
            
            # gets the columns index of the corresponding attribute
            myColIndex = aData.columns.get_loc(att_loc)
            
            # initialises the plot which is a grid with the representation of the attributes for each cluster
            fig, axs = plt.subplots(myNLines,2*myNCols,figsize=(20,4*myNLines))
            fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)

            # get the minimum and maximum of the attributes
            myAttrMin  = np.amin(aData[att_loc])
            myAttrMax  = np.amax(aData[att_loc])
            myAttrMean = np.mean(aData[att_loc])

            for i in range(myNClusters): # loop on the cluster
                
                myNDaysInCluster = len(mySortedClusterAttribution[i])
                myTitle = r'Cluster '+format(i)+': # days = '+format(myNDaysInCluster)
                
                myLine = i // myNCols 
                myCol = i % myNCols

                myX = np.empty(24*myNDaysInCluster,dtype=object)
                myY = np.empty(24*myNDaysInCluster,dtype=object)

                myYHour = np.empty((myNDaysInCluster,24),dtype=object)
                
                for j in range(len(mySortedClusterAttribution[i])): # loop on each day in the cluster
                    myDay = mySortedClusterAttribution[i][j]
                    myX[j*24:(j+1)*24] = range(24)
                    myY[j*24:(j+1)*24] = aData[att_loc][myDay*24:(myDay+1)*24]
                    myYHour[j,:] = aData[att_loc][myDay*24:(myDay+1)*24]
                    # myValues[j][:] = aData[att_loc][myDay*24:(myDay+1)*24]
                
                # values for the heatmap
                myValues = np.vstack((myX,myY)).T

                # values for countourf
                myNBins = 50
                myYHour = myYHour.T
                myYRange = np.linspace(myAttrMin, myAttrMax, num=myNBins+1)
                myXRangePlot = range(0,24)
                myYRangePlot = 0.5 * ( myYRange[:-1] + myYRange[1:] )
                myXGrid, myYGrid = np.meshgrid(myXRangePlot, myYRangePlot)
                myVGrid = np.empty((len(myXRangePlot),len(myYRangePlot)),dtype=object)
                for h in myXRangePlot:
                    hist, bin_edges = np.histogram(myYHour[h], bins=myNBins,range=(myAttrMin, myAttrMax), density=True)
                    myVGrid[h] = hist

                myVGrid = myVGrid.T

                # opacity of the lines
                myAlpha = 1.0
                if aPlotType == 'heatmap':
                    myAlpha=0.2
                if myNLines == 1: 
                    axs[2*i].set_title(myTitle)
                    axs[2*i].set_ylim([myAttrMin,myAttrMax])
                    axs[2*i].set_xlim([0,23])
                    if aPlotType == 'line':
                        for j in mySortedClusterAttribution[i]: # loop on each day in the cluster
                            axs[2*i].plot(range(24),aData[att_loc][j*24:(j+1)*24],alpha=myAlpha)
                    if aPlotType == 'heatmap':
                        sns.kdeplot(myValues, shade=True, ax=axs[2*i])
                    if aPlotType == 'countour':
                        axs[2*i].contourf(myXGrid, myYGrid, myVGrid, 100, cmap=mycmap)
                    if aPlotType == 'box':
                        sns.boxplot(data=myYHour.T, ax=axs[2*i])
                    if aPlotType == 'quantiles':
                        myQuantilesToPlot = [25,50,75,100]
                        for q in myQuantilesToPlot:
                            myUpperQuant = np.percentile(myYHour, 100 - q/2.0,axis=1)
                            myBottomQuant = np.percentile(myYHour, q/2.0,axis=1)
                            axs[2*i].fill_between(range(24),myBottomQuant,myUpperQuant,color=theBlue,alpha = q/100.0)

                    # plots histogram with days 
                    axs[2*i+1].hist(mySortedClusterAttribution[i], density=False, bins=25, range=(0,364))
                    axs[2*i+1].set_title(myTitle)
                    axs[2*i+1].set_xticks([15,104,195,287])
                    axs[2*i+1].set_xticklabels(['Jan 15','Apr 15','Jul 15','Oct 15'])
                else:
                    axs[myLine,2*myCol].set_title(myTitle)
                    axs[myLine,2*myCol].set_ylim([myAttrMin,myAttrMax])
                    axs[myLine,2*myCol].set_xlim([0,23])
                    if aPlotType == 'line':
                        for j in mySortedClusterAttribution[i]: # loop on each day in the cluster
                            axs[myLine,2*myCol].plot(range(24),aData[att_loc][j*24:(j+1)*24],alpha=myAlpha)
                    if aPlotType == 'heatmap':
                        sns.kdeplot(myValues, shade=True, ax=axs[2*i])
                    if aPlotType == 'countour':
                        axs[myLine,2*myCol].contourf(myXGrid, myYGrid, myVGrid, 100, cmap=mycmap)
                    if aPlotType == 'box':
                        sns.boxplot(data=myYHour.T, ax=axs[myLine,2*myCol])
                    if aPlotType == 'quantiles':
                        plot_quantiles(axs[myLine,2*myCol],myYHour)
                    
                    # plots histogram with days 
                    axs[myLine,2*myCol+1].hist(mySortedClusterAttribution[i], density=False, bins=25, range=(0,364))
                    axs[myLine,2*myCol+1].set_title(myTitle)
                    axs[myLine,2*myCol+1].set_xticks([15,104,195,287])
                    axs[myLine,2*myCol+1].set_xticklabels(['Jan 15','Apr 15','Jul 15','Oct 15'])

            myFigPath = myOutputFolder+'att_'+format(att_loc)
            if aFileNameExtension != '':
                myFigPath += '_'+aFileNameExtension
            if aPlotType == 'line':
                myFigPath += '_line'
            if aPlotType == 'heatmap':
                myFigPath += '_heatmap'
            if aPlotType == 'countour':
                myFigPath += '_countour'
            if aPlotType == 'box':
                myFigPath += '_box'
            if aPlotType == 'quantiles':
                myFigPath += '_quantiles'
            myFigPath += '_nCluster_'+format(len(aClusterAttribution))+'.'+aImageFormat
            plt.savefig(myFigPath,bbox_inches='tight')
            plt.close()
    else:

        ##################################
        # Generates one plot per cluster #
        ##################################
        
        # gets the number of clusters
        myNAttributes = len(aAttributesToPlot)

        # gets the number of lines for the plot
        myNCols  = 3
        myNLines = int(np.ceil((myNAttributes+1) / float(myNCols)))  # the +1 is for the plot of the histogram with the days

        # gets the cluster attribution sorted in decreasing number of days in each cluster
        mySortedClusterAttribution = get_sorted_attribution(aClusterAttribution)
        
        for c in range(len(mySortedClusterAttribution)) : # loop on cluster
            
            # initialises the plot which is a grid with the representation of each attributes for the cluster
            fig, axs = plt.subplots(myNLines,myNCols,figsize=(20,4*myNLines))
            fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)

            if aPlotHeatMap:
                for i in range(myNAttributes): # loop on the attributes
           
                    myLine = i // myNCols
                    myCol = i % myNCols

                    myAttrMin = np.amin(aData[aAttributesToPlot[i]])
                    myAttrMax = np.amax(aData[aAttributesToPlot[i]])

                    myNDaysInCluster = len(mySortedClusterAttribution[c])
                    myX = np.empty(24*myNDaysInCluster,dtype=object)
                    myY = np.empty(24*myNDaysInCluster,dtype=object)

                    for j in range(myNDaysInCluster): # loop on each day in the cluster
                        myDay = mySortedClusterAttribution[c][j]
                        myX[j*24:(j+1)*24] = range(24)
                        myY[j*24:(j+1)*24] = aData[aAttributesToPlot[i]][myDay*24:(myDay+1)*24]
                    
                    myValues = np.vstack((myX,myY)).T
                    
                    if myNLines == 1: 
                        # axs[i].hist2d(myX, myY, bins=(24,100), cmap=mycmap, range=[(0,23),(myAttrMin,myAttrMax)])
                        sns.kdeplot(myValues, shade=True, ax=axs[i])
                        axs[i].set_xlim([0,23])
                        axs[i].set_ylim([myAttrMin, myAttrMax])
                        axs[i].set_title(aAttributesToPlot[i])
                    else:
                        # axs[myLine,myCol].hist2d(myX, myY, bins=(24,100), cmap=mycmap, range=[(0,23),(myAttrMin,myAttrMax)])
                        sns.kdeplot(myValues, shade=True, ax=axs[myLine,myCol])
                        axs[myLine,myCol].set_ylim([myAttrMin, myAttrMax])
                        axs[myLine,myCol].set_xlim([0, 23])
                        axs[myLine,myCol].set_title(aAttributesToPlot[i])
                        
                # plots histogram with days 
                i = myNAttributes + 1
                myLine = i // myNCols
                myCol = i % myNCols
                if myNLines == 1: 
                    axs[i].hist(mySortedClusterAttribution[c], density=False, bins=25, range=(0,364))
                else:
                    axs[myLine,myCol].hist(mySortedClusterAttribution[c], density=False, bins=25, range=(0,364))

                myFigPath = myOutputFolder+'cluster_'+format(c+1)
                if aFileNameExtension != '':
                    myFigPath += '_'+aFileNameExtension
                myFigPath += '.'+aImageFormat
                plt.savefig(myFigPath,bbox_inches='tight')
                plt.close()
            else:
                for i in range(myNAttributes): # loop on the attributes
           
                    myLine = i // myNCols
                    myCol = i % myNCols

                    myAttrMin = np.amin(aData[aAttributesToPlot[i]])
                    myAttrMax = np.amax(aData[aAttributesToPlot[i]])

                    if myNLines == 1: 
                        for j in mySortedClusterAttribution[c]: # loop on each day in the cluster
                            axs[i].plot(range(24),aData[aAttributesToPlot[i]][j*24:(j+1)*24])
                        axs[i].set_title(aAttributesToPlot[i])
                        axs[i].set_ylim([myAttrMin, myAttrMax])

                    else:
                        for j in mySortedClusterAttribution[c]: # loop on each day in the cluster
                            axs[myLine,myCol].plot(range(24),aData[aAttributesToPlot[i]][j*24:(j+1)*24])
                        axs[myLine,myCol].set_title(aAttributesToPlot[i])
                        axs[myLine,myCol].set_ylim([myAttrMin, myAttrMax])

                # plots histogram with days 
                i = myNAttributes + 1
                myLine = i // myNCols
                myCol = i % myNCols
                if myNLines == 1: 
                    axs[i].hist(mySortedClusterAttribution[c], density=False, bins=25, range=(0,364))
                else:
                    axs[myLine,myCol].hist(mySortedClusterAttribution[c], density=False, bins=25, range=(0,364))

                myFigPath = myOutputFolder+'cluster_'+format(c+1)
                if aFileNameExtension != '':
                    myFigPath += '_'+aFileNameExtension
                myFigPath += '.'+aImageFormat
                plt.savefig(myFigPath,bbox_inches='tight')
                plt.close()
