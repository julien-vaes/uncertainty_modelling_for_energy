# imports packages

from collections import Counter, defaultdict
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

def get_data_in_PCA_basis(aX,aPCA):
    return aPCA.transform(aX)

def get_truncated_data_in_PCA_basis(aX,aPCA,aNPrincipDir):
    return np.dot(aX - aPCA.mean_, aPCA.components_.T)[:,:aNPrincipDir]

def get_data_in_PCA_basis_second_def(aX,aPCA):
    return np.dot(aX - aPCA.mean_, aPCA.components_.T)

def get_data_in_original_basis(aZ,aPCA):
    return np.dot(aZ, aPCA.components_) + aPCA.mean_

def reverse_PCA_map(aZ,aPCA):
    myZTruncFull = np.zeros((aZ.shape[0], aPCA.components_.shape[1]))
    myZTruncFull[:,:aZ.shape[1]] = aZ
    return np.dot(myZTruncFull, aPCA.components_) + aPCA.mean_

def get_data_previous_layer(aDetailsStagedSizeReductionViaPCA,aDictData):
    
    # initialises the dictionary to return
    myDictDataPreviousLayer = {}
    
    # gets the dictionary with the details of the columns that has been reduced together
    d = aDetailsStagedSizeReductionViaPCA['AttributesMerged']
    for key in np.sort(list(d.keys())):

        # recovers the data if some data is given for the specified key
        if key in aDictData:

            # gets the recovered data
            myRecoveredData = reverse_PCA_map(aDictData[key],aDetailsStagedSizeReductionViaPCA['PCA'][key])
            
            # myDictDetailsSizeOriginalData = aDetailsStagedSizeReductionViaPCA[key]['DimX']
            myRecoveredDataNames = np.sort(list(aDetailsStagedSizeReductionViaPCA['AttributesMerged'][key]))
            myDimensionBeforePCA = aDetailsStagedSizeReductionViaPCA['DimensionBeforePCA'][key]
            
            iloc = 0
            for key_loc in myRecoveredDataNames:
                myNColsLoc = myDimensionBeforePCA[key_loc]
                myDictDataPreviousLayer[key_loc] = myRecoveredData[:,iloc:iloc+myNColsLoc]
                iloc += myNColsLoc

    return myDictDataPreviousLayer

def perform_PCA_reduction(
        aDetailsSizeReductionViaPCA,  # an array with the details on the successive size reduction performed via the truncated PCA
        aStage,                       # the index of the stage
        aDictAttributesToMergeViaPCA, # dictionary where the keys or the names of resulting attributes and the values are vectors where the elements are the attributes for which we perform a PCA together
        aVarianceRatioThreshold=0.0,  # variance ratio threshold
        aNDirectionsThreshold=0       # Number of PCA directions to keep
        ):

    # gets the data to merge
    myInputData = aDetailsSizeReductionViaPCA[aStage]['Data']
    
    # initialises the dictionary to return
    # adds the details of the attributes that are reduced together using the PCA method 
    myDictDetails                            = {}
    myDictDetails['AttributesMerged']        = aDictAttributesToMergeViaPCA
    myDictDetails['Data']                    = {}
    myDictDetails['PCA']                     = {}
    myDictDetails['DimensionBeforePCA']      = {}
    myDictDetails['DimensionBeforePCASum']   = {}
    myDictDetails['DimensionAfterPCA']       = {}
    # myDictDetails['get_data_previous_stage'] = {}
    # myDictDetails['get_data_initial_stage']  = {}
    
    # computes the PCA reduction by alphabetical order of the output names
    for key in np.sort(list(aDictAttributesToMergeViaPCA.keys())):
        
        # sorts the columns names that have to be aggregated
        myAttributesToMergeSorted = np.sort(aDictAttributesToMergeViaPCA[key])
        
        # gets the data of all the attributes to merge
        myX = [myInputData[key_loc] for key_loc in myAttributesToMergeSorted]
        myX = np.concatenate(myX, axis=1)

        # computes the PCA
        myPCA = PCA(myX.shape[1])
        myPCA.fit(myX)

        # gets the data formulated in the PCA basis
        myZ = get_data_in_PCA_basis(myX,myPCA)

        ##################
        # Size reduction #
        ##################

        # initialises the number of PCA directions to keep
        myLastIndex = myX.shape[1]

        # if a number of directions are given in the arguments
        if aNDirectionsThreshold != 0:
            myLastIndex = aNDirectionsThreshold

        # checks the condition on the explained variance ratio
        if aVarianceRatioThreshold != 0.0:

            # gets the number of principal direction to keep to reach the variance threshold
            myVarianceRatio = myPCA.explained_variance_ratio_

            # boolean to tell whether there exists an index such that explained variance ratio is smaller than than the threshold
            myIndicesSatisfyingVarianceRatio = np.argwhere(myVarianceRatio < aVarianceRatioThreshold)

            if len(myIndicesSatisfyingVarianceRatio) > 0:
                myLastIndex = min(myLastIndex,myIndicesSatisfyingVarianceRatio[0][0])

        # gets the data truncated formulated in the PCA basis
        myZTrunc = myZ[:,:myLastIndex]

        # creates the dictionary with the details of the dictionary reduction
        myDictDetails['PCA'][key]                   = myPCA
        myDictDetails['Data'][key]                  = myZTrunc.copy()
        myDictDetails['DimensionBeforePCA'][key]    = {keyloc:(myInputData[keyloc].shape[1]) for keyloc in myAttributesToMergeSorted}
        myDictDetails['DimensionBeforePCASum'][key] = sum(myDictDetails['DimensionBeforePCA'][key].values())
        myDictDetails['DimensionAfterPCA'][key]     = myZTrunc.shape[1]

    # defines the functions to map from the (truncated) PCA basis to the original basis
    def my_function_get_data_previous_stage(aDictData):
        return get_data_previous_layer(myDictDetails,aDictData)
    myDictDetails['get_data_previous_stage'] = my_function_get_data_previous_stage
    
    # defines the functions to map from the (truncated) PCA basis to the initial basis
    def my_function_get_data_initial_stage(aDictData):
        return aDetailsSizeReductionViaPCA[aStage]['get_data_initial_stage'](my_function_get_data_previous_stage(aDictData))
    myDictDetails['get_data_initial_stage']  = my_function_get_data_initial_stage

    # updates the value of the PCA reduction
    aDetailsSizeReductionViaPCA = aDetailsSizeReductionViaPCA[:aStage+1]
    aDetailsSizeReductionViaPCA.append(myDictDetails)

    return aDetailsSizeReductionViaPCA
    
def get_data_next_layer(aDict,aDictData,aToTruncate,aVarianceRatioThreshold):
    
    # initialises the dictionary to return
    myDictDataNextLayer = {}
    
    d = aDict['AttributesMerged']
    for key in np.sort(list(d.keys())):

        # gets the data in terms of a matrix
        myDataToReduceAttributesDays = [aDictData[d[key][i]] for i in range(len(d[key]))]
        myX = np.concatenate(myDataToReduceAttributesDays, axis=1)
        
        myZ = -1
        myPCA = aDict[key]['PCA']
        
        if aToTruncate:
            # gets the number of principal direction to keep to reach the variance threshold
            myVarianceRatio = myPCA.explained_variance_ratio_
            myLatIndexSatisfyingRatio = next(i for i, v in enumerate(myVarianceRatio) if v < aVarianceRatioThreshold)
            myZ = get_truncated_data_in_PCA_basis(myX,myPCA,myLatIndexSatisfyingRatio)
        else:
            myZ = get_data_in_PCA_basis(myX,aDict[key]['PCA'])

        myDictDataNextLayer[key] = myZ
        
    return myDictDataNextLayer

def get_recovered_data(aListDictReduction,aDictData):
    myCurrentData = aDictData.copy()
    for myLocReduction in reversed(aListDictReduction):
        myCurrentData = get_data_previous_layer(myLocReduction,myCurrentData)
    return myCurrentData

def get_recovered_data_initial_stage(aListDictReduction,aDictData,aStage):
    myCurrentData = aDictData.copy()
    for myLocReduction in reversed(aListDictReduction[0:aStage]):
        myCurrentData = get_data_previous_layer(myLocReduction,myCurrentData)
    return myCurrentData

def get_recovered_data_initial_stage_successive_days(aNSuccessiveProfiles,aListDictReduction,aDictData,aStage):

    myInputData = aDictData.copy()

    # gets the number of data points per day in the PCA Basis
    myNDataPointsPerDay = {}
    for k in aDictData:
        myNDataPointsPerDay[k] = int(float(myInputData[k].shape[1]) / aNSuccessiveProfiles)
        print(myNDataPointsPerDay[k])

    # Initialises the output data dictionary
    myOutputData = {}

    # generates the dictionary with the data in the truncated PCA basis for day `i`
    myDailyInputData = {}
    for d in range(aNSuccessiveProfiles):
        for k in aDictData:
            print(k)
            myDailyInputData[k] = aDictData[k][:,d*myNDataPointsPerDay[k]:(d+1)*myNDataPointsPerDay[k]]
            print(myDailyInputData[k].shape)
        myDailyRecoveredData = get_recovered_data_initial_stage(aListDictReduction,myDailyInputData,aStage)

        # save the output 
        if myOutputData == {}:
            myOutputData = myDailyRecoveredData.copy() 
        else:
            for k in myOutputData:
                myNewData = np.concatenate((myOutputData[k].copy(),myDailyRecoveredData[k].copy()),axis=1)
                myOutputData[k] = myNewData

    return myOutputData

def get_recovered_data_previous_stage(aListDictReduction,aDictData,aStage):
    return get_data_previous_layer(aListDictReduction[aStage-1],aDictData)

def get_weigthed_reduced_data_dict(aColsToReduce,aDictDataAttributesDays):
    
    # initialises the dictionary with all the truncated data weighted by PCA variance ratio
    myWeigthedReducedDataDict = {}

    for key in aColsToReduce:

        # gets the reduced data
        myWeigthedMat = aDictDataAttributesDays[key]['Z_Trunc'].copy().T

        # gets the vector with the explained variance ratio of the PCA
        myExplainedVarRatio = aDictDataAttributesDays[key]['PCA'].explained_variance_ratio_

        for j in range(myWeigthedMat.shape[0]):
            myWeigthedMat[j] = myWeigthedMat[j] * np.sqrt(myExplainedVarRatio[j])

        myWeigthedReducedDataDict[key] = myWeigthedMat.T
        
    return myWeigthedReducedDataDict
