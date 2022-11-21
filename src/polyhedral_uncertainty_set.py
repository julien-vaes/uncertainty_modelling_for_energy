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

def get_lower_upper_quantiles(aDict,aAttr,aNDir,aα):
    myLowerQuantiles = np.zeros(aNDir)
    myUpperQuantiles = np.zeros(aNDir)
    for i in range(aNDir):
        myValues = aDict[aAttr][:,i]
        myLowerQuantiles[i] = np.quantile(myValues, aα)
        myUpperQuantiles[i] = np.quantile(myValues, 1.0-aα)
    return myLowerQuantiles, myUpperQuantiles

def get_vertices_uncertainty_set(aDict,aAttr,aNDir,aα):

    # gets the quantiles along each principal directions
    myLB, myUB = get_lower_upper_quantiles(aDict,aAttr,aNDir,aα)

    # gets all the vertices
    myVertices = np.zeros((2**aNDir,aNDir))
    for i in range(2**aNDir):
        myVertex = np.zeros(aNDir)
        myBinaryRepresentation = format(i, '#0'+format(aNDir+2)+'b').split('b')[1]
        for j in range(aNDir):
            if myBinaryRepresentation[j] == '0':
                myVertex[j] = myLB[j]
            else:
                myVertex[j] = myUB[j]
        myVertices[i] = myVertex
    return myVertices

def sort_realisations(aRealisations,aIndex=0):
    myRealisations = aRealisations.copy()
    myFirstKeyData = next(iter(myRealisations))
    mySortedIndices = (myRealisations[myFirstKeyData][:,aIndex]).argsort()
    for k in myRealisations:
        myRealisations[k] = myRealisations[k][mySortedIndices]
    return myRealisations

def get_pus_random_realisations_in_uncertainty_set(aNRealisations,aDict,aAttr,aNDir,aα):

    # gets the quantiles along each principal directions
    myLB, myUB = get_lower_upper_quantiles(aDict,aAttr,aNDir,aα)

    # initialises the array that will contain the random realisations
    myRealisations = np.zeros((aNRealisations,aNDir))

    # generates the random realisations
    for j in range(aNRealisations):
        myR = np.random.rand(aNDir)
        myξ = np.multiply(myLB,myR) + np.multiply(myUB,1-myR)

        myRealisations[j] = myξ

    return myRealisations

def uniform_boundary_point_in_simplex(n):
    myR = np.random.rand(n)
    myE = - np.log10(myR)
    mySumE = np.sum(myE)
    return myE / mySumE

def uniform_point_in_simplex(n):
    myUniformBoundaryPointInSimplex = uniform_boundary_point_in_simplex(n)
    return np.random.rand() * myUniformBoundaryPointInSimplex

def get_pus_random_realisations_PCA_basis_boundary(
        aNRealisations,
        aDetailsSizeReductionViaPCA,
        aStage,
        aAttr,
        aNDir,
        aα,
        aBudget=10**3,
        aPairwiseBudget=10**3,
        aNDirPairwiseCons=0,
        aMaxNTrials=10**3,
        aCluster=-1,
        aClusterAttribution=[]
        ):

    # gets the data
    myData = aDetailsSizeReductionViaPCA[aStage]['Data']

    # if we have to plot for a given cluster then copies the data and remove the unneeded one
    if aCluster != -1:
        # gets the data
        myData = aDetailsSizeReductionViaPCA[aStage]['Data'].copy()

        # gets the days associated to the cluster
        myClusterDays = aClusterAttribution[aAttr][aCluster]

        # gets the data related to the desired cluster
        for key in myData:
            myData[key] = myData[key][myClusterDays]


    # gets the quantiles along each principal directions
    myLB, myUB = get_lower_upper_quantiles(myData,aAttr,aNDir,aα)

    # initialises the array that will contain the random realisations
    myRealisations = np.zeros((aNRealisations,aNDir))

    # generates the random realisations
    for j in range(aNRealisations):
        myR = 0
        myTrials = 0
        myNotFoundR = True
        while myNotFoundR & (myTrials < aMaxNTrials):
            myTrials += 1
            myR = np.random.choice([0.0,1.0],aNDir) * np.random.choice([-1.0,1.0],aNDir)
            myNotFoundR = False
            if sum(abs(myR)) > aBudget:
                myR = myR * (aBudget/sum(abs(myR)))

            for k in range(aNDirPairwiseCons):
                for l in range(aNDirPairwiseCons):
                    mySum = abs(myR[k]) + abs(myR[l])
                    if (k != l) & (mySum > aPairwiseBudget):
                        myR[k] = myR[k] * (aPairwiseBudget / mySum)
                        myR[l] = myR[l] * (aPairwiseBudget / mySum)

        if myTrials >= aMaxNTrials:
            print('The maximum number of iterations to find a realisation in the budget limits has been reached')

        myλ = (myR + 1.0) / 2.0
        myξ = np.multiply(myLB,myλ) + np.multiply(myUB,(1-myλ))
        myRealisations[j] = myξ

    return {aAttr: myRealisations}

def get_pus_random_realisations_PCA_basis(
        aNRealisations,
        aDetailsSizeReductionViaPCA,
        aStage,
        aAttr,
        aNDir,
        aα,
        aBudget=10**3,
        aPairwiseBudget=10**3,
        aNDirPairwiseCons=0,
        aMaxNTrials=10**3,
        aBoundary=False,
        aCluster=-1,
        aClusterAttribution=[]
        ):

    if aBoundary:
        return get_pus_random_realisations_PCA_basis_boundary(
                aNRealisations,
                aDetailsSizeReductionViaPCA,
                aStage,
                aAttr,
                aNDir,
                aα,
                aBudget=aBudget,
                aPairwiseBudget=aPairwiseBudget,
                aNDirPairwiseCons=aNDirPairwiseCons,
                aMaxNTrials=aMaxNTrials,
                aCluster=aCluster,
                aClusterAttribution=aClusterAttribution
                )

    # gets the data
    myData = aDetailsSizeReductionViaPCA[aStage]['Data']

    # if we have to plot for a given cluster then copies the data and remove the unneeded one
    if aCluster != -1:
        # gets the data
        myData = aDetailsSizeReductionViaPCA[aStage]['Data'].copy()

        # gets the days associated to the cluster
        myClusterDays = aClusterAttribution[aAttr][aCluster]

        # gets the data related to the desired cluster
        for key in myData:
            myData[key] = myData[key][myClusterDays]

    # gets the quantiles along each principal directions
    myLB, myUB = get_lower_upper_quantiles(myData,aAttr,aNDir,aα)
    myMean = 0.5 * (myLB + myUB)

    # generates boundary random realisation
    myRealisations = get_pus_random_realisations_PCA_basis_boundary(
            aNRealisations,
            aDetailsSizeReductionViaPCA,
            aStage,
            aAttr,
            aNDir,
            aα,
            aBudget=aBudget,
            aPairwiseBudget=aPairwiseBudget,
            aNDirPairwiseCons=aNDirPairwiseCons,
            aMaxNTrials=aMaxNTrials,
            aCluster=aCluster,
            aClusterAttribution=aClusterAttribution
            )

    # then take a random realisations between the boundary ones and the mean one.
    for i in range(myRealisations[aAttr].shape[0]):
        myRealisations[aAttr][i] = myMean + np.random.rand() * ( myRealisations[aAttr][i] - myMean )

    return myRealisations

def get_pus_random_realisations_original_basis(
        aNRealisations,
        aDetailsSizeReductionViaPCA,
        aStage,
        aAttr,
        aNDir,
        aα,
        aBudget=10**3,
        aPairwiseBudget=10**3,
        aNDirPairwiseCons=0,
        aMaxNTrials=10**3,
        aBoundary=False,
        aCluster=-1,
        aClusterAttribution=[]
        ):

    myRealisationsPCABasis = get_pus_random_realisations_PCA_basis(
            aNRealisations,
            aDetailsSizeReductionViaPCA,
            aStage,
            aAttr,
            aNDir,
            aα,
            aBudget=aBudget,
            aPairwiseBudget=aPairwiseBudget,
            aNDirPairwiseCons=aNDirPairwiseCons,
            aMaxNTrials=aMaxNTrials,
            aBoundary=aBoundary,
            aCluster=aCluster,
            aClusterAttribution=aClusterAttribution
            )

    myRealisationsOriginalBasis = aDetailsSizeReductionViaPCA[aStage]['get_data_initial_stage'](myRealisationsPCABasis)

    return myRealisationsOriginalBasis

def get_indices_verify_continuity_constraints(
        aNRealisations,
        aRealisations,
        aNextDailyProfileRealisations,
        aDetailsSizeReductionViaPCA,
        aStdFactorTolContinuity=5.0
        ):

    myVerifyContinuity = np.array([True for i in range(aNRealisations)])

    # checks the continuity for all attributes
    for k in aRealisations:
        myLastIndexData = aRealisations[k][:,-1]
        myFirstIndexData = aNextDailyProfileRealisations[k][:,0]
        myDiff = np.abs(myLastIndexData - myFirstIndexData)
        myMean = aDetailsSizeReductionViaPCA[0]['StatsDailyContinuity'][k]['Mean']
        myStd  = aDetailsSizeReductionViaPCA[0]['StatsDailyContinuity'][k]['Std']
        myAttributeContinuityCheck = ( ( (myMean - aStdFactorTolContinuity * myStd) <= myDiff ) & ( (myMean + aStdFactorTolContinuity * myStd) >= myDiff ) )
        myVerifyContinuity = myVerifyContinuity * myAttributeContinuityCheck

    return myVerifyContinuity

def sort_new_realisation_most_similar_to_last_component(
        aRealisations,
        aNextDailyProfileRealisations,
        aAttributeHelpContinuity = ''
        ):


    # gets the key to help with the continuity condition (ideally the most restrictive to be verified)
    myAttributeHelpContinuity = aAttributeHelpContinuity
    if myAttributeHelpContinuity == '':
        myAttributeHelpContinuity = next(iter(aRealisations))

    if aRealisations[myAttributeHelpContinuity].shape[0] == 0:
        return aNextDailyProfileRealisations 
    
    # last component of the realisations until now
    myLastComponent = aRealisations[myAttributeHelpContinuity][:,-1]
    
    # first component of the realisations for the new daily profile
    myFistComponent = aNextDailyProfileRealisations[myAttributeHelpContinuity][:,0]

    # difference between the components
    myDifferences = (myLastComponent.reshape(1,-1) - myFistComponent.reshape(-1,1))

    # gets the indices with the lowest difference
    myIndices = np.abs(myDifferences).argmin(axis=0)

    # gets the next daily profiles reordered
    myNextDailyProfileRealisations = {}
    for k in aNextDailyProfileRealisations:
        myNextDailyProfileRealisations[k] = aNextDailyProfileRealisations[k][myIndices]

    return myNextDailyProfileRealisations

def get_random_index_where_true(aLine):
    myIndicesWhereContinuityConditionVerified = [i for i, x in enumerate(aLine) if x]
    if len(myIndicesWhereContinuityConditionVerified) > 0:
        return np.random.choice(myIndicesWhereContinuityConditionVerified)
    else:
        return -1

def find_next_random_daily_realisations(
        aRealisations,
        aNextDailyProfileRealisations,
        aDetailsSizeReductionViaPCA,
        aStdFactorTolContinuity=5.0
        ):

    myAttributes = [k for k in aRealisations]
    myMatrixLastComponentRealisations    = np.array([aRealisations[k][:,-1] for k in myAttributes])
    myMatrixFistComponentNewRealisations = np.array([aNextDailyProfileRealisations[k][:,0] for k in myAttributes])
    myMeanStep = np.array([aDetailsSizeReductionViaPCA[0]['StatsDailyContinuity'][k]['Mean'] for k in myAttributes])
    myStdStep  = np.array([aDetailsSizeReductionViaPCA[0]['StatsDailyContinuity'][k]['Std'] for k in myAttributes])

    myNReal    = myMatrixLastComponentRealisations.shape[1]
    myNNewReal = myMatrixFistComponentNewRealisations.shape[1]

    # initialises the matrix that will contain the analysis of continuity
    myMatrixContinuity = np.empty((myNReal,myNNewReal), dtype=float)
    for i in range(myNReal):
        for j in range(myNNewReal):
            myStep = myMatrixFistComponentNewRealisations[:,j] - myMatrixLastComponentRealisations[:,i] 
            myStepNormalise = np.abs(myStep - myMeanStep) / myStdStep
            myMatrixContinuity[i,j] = np.max(myStepNormalise)

    # gets the matrix with booleans where the continuity condition is verified
    myMatrixVerifiesContinuity = (myMatrixContinuity < aStdFactorTolContinuity)

    # gets a vector where each index is a random index of the next daily profiles such that the continuity condition is verified 
    myFeasibleIndices = np.apply_along_axis(get_random_index_where_true, 1, myMatrixVerifiesContinuity)

    # gets the indices with a feasible next realisation verifying the continuity condition
    myIndicesWithFeasibleNextRealisation = np.argwhere(myFeasibleIndices >= 0)[:,0]

    return myFeasibleIndices, myIndicesWithFeasibleNextRealisation 

def get_successive_feasible_pus_random_realisations_original_basis(
        aNRealisations,
        aDetailsSizeReductionViaPCA,
        aStage,
        aAttr,
        aNDir,
        aα,
        aBudget=10**3,
        aPairwiseBudget=10**3,
        aNDirPairwiseCons=0,
        aMaxNTrials=10**3,
        aBoundary=False,
        aCluster=-1,
        aClusterAttribution=[],
        aRealisations={},
        aEnforceContinuity=False,
        aStdFactorTolContinuity=5.0,
        aNRealisationTrialsForNextDailyProfile=10**3
        ):

    myCandidateNextDailyProfileRealisations = get_pus_random_realisations_original_basis(
            aNRealisations,
            aDetailsSizeReductionViaPCA,
            aStage,
            aAttr,
            aNDir,
            aα,
            aBudget=aBudget,
            aPairwiseBudget=aPairwiseBudget,
            aNDirPairwiseCons=aNDirPairwiseCons,
            aMaxNTrials=aMaxNTrials,
            aBoundary=aBoundary,
            aCluster=aCluster,
            aClusterAttribution=aClusterAttribution
            )

    myNRealisations = aRealisations[next(iter(aRealisations))].shape[0]
    myFeasibleIndices                    = range(myNRealisations)
    myIndicesWithFeasibleNextRealisation = np.array([True for i in range(myNRealisations)])

    print(myNRealisations)

    # verifies that the realisations satisfy the continuity constraint
    if aEnforceContinuity:

        # gets the indices for the next daily profile 
        myFeasibleIndices, myIndicesWithFeasibleNextRealisation = find_next_random_daily_realisations(
                aRealisations,
                myCandidateNextDailyProfileRealisations,
                aDetailsSizeReductionViaPCA,
                aStdFactorTolContinuity=aStdFactorTolContinuity
                )

    # get the matrix with the next daily profiles
    myNextDailyProfileRealisations = {}
    for k in myCandidateNextDailyProfileRealisations:
        myNextDailyProfileRealisations[k] = myCandidateNextDailyProfileRealisations[k][myFeasibleIndices]

    # keeps only the realisations that verifies the continuity constraint
    myRealisationsContinuity = aRealisations.copy()

    # keeps only the indices that verifies the continuity constraint
    for k in myRealisationsContinuity:
        myRealisationsContinuity[k] = np.concatenate((myRealisationsContinuity[k][myIndicesWithFeasibleNextRealisation],myNextDailyProfileRealisations[k][myIndicesWithFeasibleNextRealisation]),axis=1)

    return myRealisationsContinuity

def get_pus_random_realisations_succession_days_original_basis(
        aNRealisations,
        aDetailsSizeReductionViaPCA,
        aStage,
        aAttr,
        aNDir,
        aα,
        aBudget=10**3,
        aPairwiseBudget=10**3,
        aNDirPairwiseCons=0,
        aMaxNTrials=10**3,
        aBoundary=False,
        aSuccessiveClusters=[],
        aClusterAttribution=[],
        aEnforceContinuity=False,
        aStdFactorTolContinuity=5.0,
        aNRealisationTrialsForNextDailyProfile = 10**3
        ):

    # gets the number of successive clusters to return
    myNSuccessiveClusters = len(aSuccessiveClusters)

    # checks whether there is at least one cluster to generate
    if myNSuccessiveClusters == 0:
        raise NameError('The argument *aSuccessiveClusters* should contain at least one element.')

    # initialises the realisations by generation realisations for the first cluster 
    myRealisations = get_pus_random_realisations_original_basis(
        aNRealisations,
        aDetailsSizeReductionViaPCA,
        aStage,
        aAttr,
        aNDir,
        aα,
        aBudget=aBudget,
        aPairwiseBudget=aPairwiseBudget,
        aNDirPairwiseCons=aNDirPairwiseCons,
        aMaxNTrials=aMaxNTrials,
        aBoundary=aBoundary,
        aCluster=aSuccessiveClusters[0],
        aClusterAttribution=aClusterAttribution
        )

    # adds the successive days iteratively
    for d in range(1,myNSuccessiveClusters):

        # concatenates the next day to the current realisations
        myRealisations = get_successive_feasible_pus_random_realisations_original_basis(
                aNRealisations,
                aDetailsSizeReductionViaPCA,
                aStage,
                aAttr,
                aNDir,
                aα,
                aBudget=aBudget,
                aPairwiseBudget=aPairwiseBudget,
                aNDirPairwiseCons=aNDirPairwiseCons,
                aMaxNTrials=aMaxNTrials,
                aBoundary=aBoundary,
                aCluster=aSuccessiveClusters[d],
                aClusterAttribution=aClusterAttribution,
                aRealisations = myRealisations,
                aEnforceContinuity=aEnforceContinuity,
                aStdFactorTolContinuity=aStdFactorTolContinuity,
                aNRealisationTrialsForNextDailyProfile = aNRealisationTrialsForNextDailyProfile
                )

    print(myRealisations[next(iter(myRealisations))].shape[0])

    return myRealisations
