# imports packages

from collections import Counter, defaultdict
import copy
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from numpy import unravel_index
import os
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

import pca_size_reduction as psr
import polyhedral_uncertainty_set as pus

# -------------------- #
# Scenarios definition #
# -------------------- #

class Scenario:
    def __init__(self, a_cluster_index_for_each_attribute, a_cluster_pus_for_each_attribute, a_probability):

        # Details on the scenario: for each final attribute contains the cluster index 
        self.cluster_index_for_each_attribute = a_cluster_index_for_each_attribute

        # Details on the scenario: for each final attribute contains the PolyhedralUncertaintySet related to the corresponding cluster  
        self.cluster_pus_for_each_attribute = a_cluster_pus_for_each_attribute

        # Probability of the scenario
        self.probability = a_probability

    def get_probability(self):
        return self.probability

    def get_linear_constraints_for_optimisation(self):
        return {k:self.cluster_pus_for_each_attribute[k].get_linear_constraints_for_optimisation() for k in self.cluster_pus_for_each_attribute} 

    def get_points_in_scenario(self,
                               a_n_points,
                               a_recompute = False,
                               a_epsilon = None,
                               a_minimum_epsilon = 1.0e-4,
                               a_max_n_trials = 10**1,
                               a_tol = 1.0e-5):

        my_scenario_data = {}
        for attr in list(self.cluster_pus_for_each_attribute.keys()):

            print(f"Compute point in the PUS for the attribute = {attr}")

            # Get the polyhedral uncertainty set associated to the attribute 'attr'
            my_attr_pus = self.cluster_pus_for_each_attribute[attr]

            # Generate random realisation inside the pus
            my_attr_data_in_pus = my_attr_pus.get_points_in_pus(a_n_points = a_n_points,
                                                                a_recompute = a_recompute,
                                                                a_epsilon = a_epsilon,
                                                                a_minimum_epsilon = a_minimum_epsilon,
                                                                a_max_n_trials = a_max_n_trials,
                                                                a_tol = a_tol)

            my_scenario_data[attr] = my_attr_data_in_pus.copy()

        return my_scenario_data

def get_scenarios(a_size_reduction_via_pca,
                  a_cluster_attribution,
                  a_cluster_pus,
                  a_attributes,
                  a_prob_threshold):

    for att in a_attributes:
        if att not in list(a_cluster_attribution.keys()):
            raise ValueError(f"No clustering has been computed related to the attribute '{att}'. The clusters have been computed for the following attributes:\n{list(a_cluster_attribution.keys())}")

    # Initialise the matrix counting the number of days attributed to each scenario of the Cartesian product
    my_n_days_cartesian_prod_clusters = np.zeros(tuple((len(a_cluster_attribution[att]) for att in a_attributes)))

    for d in range(a_size_reduction_via_pca.n_data_points):

        clust_attribution = []

        for k in a_attributes:
            for i, _ in enumerate(a_cluster_attribution[k]):
                if d in a_cluster_attribution[k][i]:
                    clust_attribution.append(i)

        my_n_days_cartesian_prod_clusters[tuple(clust_attribution)] += 1

    if np.sum(my_n_days_cartesian_prod_clusters) != a_size_reduction_via_pca.n_data_points:
        raise ValueError(f"The number of days in the clustering does not corresponds to the number of data points.")

    # Create a boolean mask to find the elements that are below the threshold and should be fixed to 0
    my_mask = ( my_n_days_cartesian_prod_clusters / a_size_reduction_via_pca.n_data_points ) < a_prob_threshold

    # Remove scenarios with prob < a_prob_threshold
    my_n_days_cartesian_prod_clusters[my_mask] = 0

    # Derive the probability of each retained scenario when removing those not satisfying the threshold
    my_prob_cartesian_prod_clusters = my_n_days_cartesian_prod_clusters / np.sum(my_n_days_cartesian_prod_clusters)  

    # Round the probabilities to 3 digits
    my_prob_cartesian_prod_clusters = np.around(my_prob_cartesian_prod_clusters, decimals=3) 

    # Initialise the vector with the Scenario objects
    my_scenarios_indices = np.where(my_prob_cartesian_prod_clusters > 0.0)
    my_n_retained_scenarios = len(my_scenarios_indices[0])
    my_scenarios = []
    for i in range(my_n_retained_scenarios):
        my_cluster_attribution = {}
        my_cluster_pus = {}
        my_indices = []
        for j, att in enumerate(a_attributes):
            my_index = my_scenarios_indices[j][i]
            my_indices.append(my_index)
            my_cluster_attribution[att] = my_index
            my_cluster_pus[att] = a_cluster_pus[att][my_index]

        # Define the scenario
        my_scenario = Scenario(my_cluster_attribution,my_cluster_pus,my_prob_cartesian_prod_clusters[tuple(my_indices)])

        # Add the scenario to the vector of scenarios
        my_scenarios.append(my_scenario)

    return my_scenarios, my_n_retained_scenarios



