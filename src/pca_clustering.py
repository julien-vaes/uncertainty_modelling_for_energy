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

def get_data_in_pca_basis(aX,aPCA):
    return aPCA.transform(aX)

def get_truncated_data_in_pca_basis(aX,aPCA,aNPrincipDir):
    return np.dot(aX - aPCA.mean_, aPCA.components_.T)[:,:aNPrincipDir]

def get_data_in_pca_basis_second_def(aX,aPCA):
    return np.dot(aX - aPCA.mean_, aPCA.components_.T)

def get_data_in_original_basis(aZ,aPCA):
    return np.dot(aZ, aPCA.components_) + aPCA.mean_

def get_recovered_data_in_original_basis(aZ,aPCA):
    myZTruncFull = np.zeros((aZ.shape[0], aPCA.components_.shape[1]))
    myZTruncFull[:,:aZ.shape[1]] = aZ
    return np.dot(myZTruncFull, aPCA.components_) + aPCA.mean_

def perform_pca_reduction(aDictColumnsToReduce,aDictData,aVarianceRatioThreshold):
    
    # initialises the dictionary to return
    myDictDetails = {}
    
    # adds the details on the attributes that are reduced together using the PCA method 
    myDictDetails['ColumnsToReduce'] = aDictColumnsToReduce
    
    # computes the pca reduction by alphabetical order of the output names
    for key in np.sort(list(aDictColumnsToReduce.keys())):
        
        # sorts the columns names that have to be aggregated
        myColsToReduceSorted = np.sort(aDictColumnsToReduce[key])
        
        myX = [aDictData[key_loc] for key_loc in myColsToReduceSorted]
        myX = np.concatenate(myX, axis=1)

        mySizeOriginalDataToReduce = {key_loc : aDictData[key_loc].shape[1] for key_loc in myColsToReduceSorted}

        # computes the PCA
        myPCA = PCA(myX.shape[1])
        myPCA.fit(myX)

        # gets the number of principal direction to keep to reach the variance threshold
        myVarianceRatio = myPCA.explained_variance_ratio_
        myShouldTruncate = True in (ele < aVarianceRatioThreshold for ele in myVarianceRatio)
        myLatIndexSatisfyingRatio = len(myVarianceRatio)
        if myShouldTruncate:
            myLatIndexSatisfyingRatio =next(i for i, v in enumerate(myVarianceRatio) if v < aVarianceRatioThreshold)

        # gets the data formulated in the PCA basis
        myZ = get_data_in_pca_basis(myX,myPCA)
        
        # gets the data truncated formulated in the PCA basis
        myZTrunc = myZ[:,:myLatIndexSatisfyingRatio]

        # creates the dictionary with the details of the dictionary reduction
        myDict = {}
        myDict['PCA'] = myPCA
        myDict['NAttributes'] = myX.shape[1]
        myDict['NPrincipalDir'] = myLatIndexSatisfyingRatio
        myDict['X'] = myX
        myDict['Z'] = myZ
        myDict['DimX'] = mySizeOriginalDataToReduce
        myDict['Z_Trunc'] = myZTrunc
        
        # adds the dict to the dict that contains all the pca reduction
        myDictDetails[key] = myDict
        
    return myDictDetails
    
def get_data_next_layer(aDict,aDictData,aToTruncate,aVarianceRatioThreshold):
    
    # initialises the dictionary to return
    myDictDataNextLayer = {}
    
    d = aDict['ColumnsToReduce']
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
            myZ = get_truncated_data_in_pca_basis(myX,myPCA,myLatIndexSatisfyingRatio)
        else:
            myZ = get_data_in_pca_basis(myX,aDict[key]['PCA'])

        myDictDataNextLayer[key] = myZ
        
    return myDictDataNextLayer
    
def get_data_next_layer_trunc(aDict,aDictData,aVarianceRatioThreshold):
    return get_data_next_layer(aDict,aDictData,True,aVarianceRatioThreshold)
    
def get_data_next_layer_full(aDict,aDictData):
    return get_data_next_layer(aDict,aDictData,False,0.0)

def get_data_previous_layer(aDict,aDictData):
    
    # initialises the dictionary to return
    myDictDataPreviousLayer = {}
    
    # gets the dictionary with the details of the columns that has been reduced together
    d = aDict['ColumnsToReduce']
    for key in np.sort(list(d.keys())):
        
        # recovers the data if some data is given for the specified key
        if key in aDictData.keys():

            # gets the recovered data
            myXRecovered = get_recovered_data_in_original_basis(aDictData[key],aDict[key]['PCA'])
            
            # gets the number of recovered attributes
            myNRecoveredAttributes = myXRecovered.shape[1]
            
            # gets the theoretical number of recovered attributes
            myTheoNRecoveredAttributes = sum(aDict[key]['DimX'].values())
            
            # checks that the number of recovered attributes is valid
            if myNRecoveredAttributes != myTheoNRecoveredAttributes:
                print('Error in the number of recovered attributes to get')
                return None
            
            myDictDetailsSizeOriginalData = aDict[key]['DimX']
            myRecoveredDataNames = np.sort(list(myDictDetailsSizeOriginalData.keys()))
            
            iloc = 0
            for key_loc in myRecoveredDataNames:
                myNColsLoc = myDictDetailsSizeOriginalData[key_loc]
                myDictDataPreviousLayer[key_loc] = myXRecovered[:,iloc:iloc+myNColsLoc]
                iloc += myNColsLoc

    return myDictDataPreviousLayer

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

def get_recovered_data_previous_stage(aListDictReduction,aDictData,aStage):
    return get_data_previous_layer(aListDictReduction[aStage-1],aDictData)

def get_weigthed_reduced_data_dict(aColsToReduce,aDictDataAttributesDays):
    
    # initialises the dictionary with all the truncated data weighted by PCA variance ratio
    myWeigthedReducedDataDict = {}

    for key in aColsToReduce.keys():

        # gets the reduced data
        myWeigthedMat = aDictDataAttributesDays[key]['Z_Trunc'].copy().T

        # gets the vector with the explained variance ratio of the PCA
        myExplainedVarRatio = aDictDataAttributesDays[key]['PCA'].explained_variance_ratio_

        for j in range(myWeigthedMat.shape[0]):
            myWeigthedMat[j] = myWeigthedMat[j] * np.sqrt(myExplainedVarRatio[j])

        myWeigthedReducedDataDict[key] = myWeigthedMat.T
        
    return myWeigthedReducedDataDict

def get_data_per_day(aDictData):
    return {key : aDictData[key].reshape(365,24) for key in aDictData.keys()}
