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
        myValues = aDict[aAttr]['Z'][:,i]
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

def get_random_realisations_in_uncertainty_set(aNRealisations,aDict,aAttr,aNDir,aα):

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

def get_random_realisations_boundary_uncertainty_set_budget(aNRealisations,aDict,aAttr,aNDir,aα,aBudget=10**3,aPairwiseBudget=10**3,aNDirPairwiseCons=0,aMaxNTrials=10**3):

    # gets the quantiles along each principal directions
    myLB, myUB = get_lower_upper_quantiles(aDict,aAttr,aNDir,aα)

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

    return myRealisations

def get_random_realisations_in_uncertainty_set_budget(aNRealisations,aDict,aAttr,aNDir,aα,aBudget=10**3,aPairwiseBudget=10**3,aNDirPairwiseCons=0,aMaxNTrials=10**3):

    # gets the quantiles along each principal directions
    myLB, myUB = get_lower_upper_quantiles(aDict,aAttr,aNDir,aα)
    myMean = 0.5 * (myLB + myUB)

    myRealisations = get_random_realisations_boundary_uncertainty_set_budget(aNRealisations,aDict,aAttr,aNDir,aα,aBudget=aBudget,aPairwiseBudget=aPairwiseBudget,aNDirPairwiseCons=aNDirPairwiseCons,aMaxNTrials=aMaxNTrials)

    for i in range(myRealisations.shape[0]):
        myRealisations[i] = myMean + np.random.rand() * ( myRealisations[i] - myMean )

    return myRealisations
