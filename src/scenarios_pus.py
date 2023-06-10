from collections import Counter, defaultdict
import copy
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import unravel_index
from numpy import random
import os
import pandas as pd
from pandas import *
from scipy import linalg
from scipy.stats import norm
from scipy.stats.stats import pearsonr
import seaborn as sns
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
import sys
import shutil
from pathlib import Path

import pca_size_reduction as psr
import polyhedral_uncertainty_set as pus
import clustering_analysis as cl
import scenarios as sc

######################
# Database functions #
######################

def transform_data_daily_profiles(a_data, a_n_values_per_attribute_per_data_point):
    """
    Reshape a pandas dataframe of time series data to daily profiles.

    Args:
        a_data (pandas.DataFrame): The dataframe containing the time series data to reshape.
        a_n_values_per_attribute_per_data_point (int): The number of values describing a data point for each attribute.

    Returns:
        dict: A dictionary where each key corresponds to an attribute in the original dataframe, and each value is
        a 2D numpy array of daily profiles (each row is a daily profile).

    """

    # Check that the number of data points is a multiple of the number of points per daily profile
    if a_data.shape[0] % a_n_values_per_attribute_per_data_point != 0:
        raise ValueError("The number of data points must be a multiple of the number of points per daily profile")

    # Initialize an empty dictionary to hold the reshaped data
    my_attributes_data_points = {}

    # Iterate over the columns of the original dataframe
    for col in a_data.columns:
        # Extract the data for this column as a numpy array
        my_data_array = a_data[col].values
        # Reshape the data into daily profiles
        my_attribute_data_points = my_data_array.reshape(-1, a_n_values_per_attribute_per_data_point)
        # Add the daily profiles to the dictionary
        my_attributes_data_points[col] = my_attribute_data_points

    return my_attributes_data_points

def get_stats_daily_continuity(a_data):
    """
    Computes the mean and standard deviation of the step between the last entry of a daily profile and the first of the next daily profile for each attribute in the input dictionary.

    Args:
        a_data (dict): A dictionary where each key corresponds to an attribute and the value is a 2D numpy array representing the daily profiles of that attribute.

    Returns:
        dict: A dictionary where each key corresponds to an attribute and the value is another dictionary containing the mean and standard deviation of the step for that attribute.
    """
    # Initializes the dictionary to store the computed statistics
    my_dict_stats_daily_continuity = {}

    # Computes the mean and standard deviation of the step between the last entry of a daily profile and the first of the next one for each attribute
    for k in a_data:
        my_diff_first_with_last_component = a_data[k][1:, 0] - a_data[k][:-1, -1]
        my_dict_stats_daily_continuity[k] = {'mean': np.mean(my_diff_first_with_last_component), 'std': np.std(my_diff_first_with_last_component)}

    return my_dict_stats_daily_continuity


###################################################
# Class: polyhedral uncertainty set for scenarios #
###################################################

class ScenariosPUS:

    def __init__(self,a_file,a_n_values_per_attribute_per_data_point):

        # data
        self.dataframe_original_data = pd.read_csv(a_file) # loads the data as a dataframe where each column corresponds to an attribute
        self.original_data = transform_data_daily_profiles(self.dataframe_original_data, a_n_values_per_attribute_per_data_point = a_n_values_per_attribute_per_data_point) # transforms the data in data points profiles
        self.n_data_points = int(self.dataframe_original_data.shape[0] / a_n_values_per_attribute_per_data_point) # the number of data points
        self.final_attributes = None

        self.size_reduction_via_pca = None
        self.clusters = None

    def get_scenarios(self):
        return self.scenarios

    def perform_size_reduction(self,a_attributes_to_merge,a_explained_variance_threshold,a_n_directions_threshold):

        # Get the final attributes
        self.final_attributes = np.sort(list(a_attributes_to_merge[-1].keys()))

        # Initialise and perform the size reduction object via clustering
        self.size_reduction_via_pca = psr.SizeReductionViaPCA(self.original_data,self.n_data_points,a_attributes_to_merge,a_explained_variance_threshold,a_n_directions_threshold)

    def perform_clustering(self,a_n_clusters,a_method='kmeans',a_seed=None):

        if self.size_reduction_via_pca is None:
            raise ValueError(f"A size reduction via PCA must be performed via the function 'perform_size_reduction' before clustering.")
        
        # Get the final attributes resulting from the size reduction via PCA
        my_attributes_to_cluster = self.size_reduction_via_pca.attributes[-1]

        # Get the reduced data resulting from the size reduction via PCA
        my_data = (self.size_reduction_via_pca.get_stages_data()[-1]).get_data()

        my_cluster_attribution = {}
        for att in my_attributes_to_cluster:

            # Check whether we have a number of clusters specified
            if att not in list(a_n_clusters.keys()):
                raise ValueError(f"The clustering of {att} cannot be computed as no number of clusters is specified in the argument 'a_n_clusters'.")

            # Perform the clustering and save the cluster attributions for the new attribute 'att'
            my_cluster_attribution[att] = cl.get_clusters_attribution(my_data[att],a_n_clusters[att],a_method=a_method,a_seed=a_seed)

        self.cluster_attribution = my_cluster_attribution

    def compute_polyhedral_uncertainty_set_for_each_cluster_of_each_attribute(self,a_details_polyhedral_uncertainty_set_generation):
        my_pus_for_each_cluster_of_each_attribute = {}
        for attr in self.cluster_attribution:

            # Initialise the parameters for the generation of PUS for the attribute attr 
            my_details_polyhedral_uncertainty_set_generation = {'n_directions_pus':self.size_reduction_via_pca.get_attr_data_final_stage(attr).shape[1],
                                                                'α':0.05,
                                                                'cumulated_budget':5.0,
                                                                'pairwise_budget':1.5,
                                                                'n_dir_pairwise_budget':24}

            # Get the details on for the generation of PUS for the attribute attr 
            my_details_polyhedral_uncertainty_set_generation.update(a_details_polyhedral_uncertainty_set_generation[attr])

            # Extract the values of the parameters
            my_n_directions_pus = my_details_polyhedral_uncertainty_set_generation['n_directions_pus']
            my_α = my_details_polyhedral_uncertainty_set_generation['α']
            my_cumulated_budget=my_details_polyhedral_uncertainty_set_generation['cumulated_budget']
            my_pairwise_budget=my_details_polyhedral_uncertainty_set_generation['pairwise_budget']
            my_n_dir_pairwise_budget=my_details_polyhedral_uncertainty_set_generation['n_dir_pairwise_budget']

            # Get the linear transformation from the truncated basis to the original basis
            my_A_to_original_basis, my_b_to_original_basis, my_original_data_description = self.size_reduction_via_pca.linear_transformation_trunc_pca_to_original_basis[-1][attr]

            # Initialise the array with the PUS for each cluster of the attribute 'attr'
            my_clusters_pus = []
            for cluster_data_indices in self.cluster_attribution[attr]:
                # Get the data of the cluster related to 'attr'
                my_data_attr = self.size_reduction_via_pca.get_attr_cluster_data_final_stage(attr,cluster_data_indices)

                # Generate the polyhedral uncertainty set
                my_polyhedral_uncertainty_set = pus.PolyhedralUncertaintySet(my_data_attr,
                                                                             a_n_directions_pus      = my_n_directions_pus,
                                                                             a_α                     = my_α,
                                                                             a_cumulated_budget      = my_cumulated_budget,
                                                                             a_pairwise_budget       = my_pairwise_budget,
                                                                             a_n_dir_pairwise_budget = my_n_dir_pairwise_budget,
                                                                             a_A_to_original_basis   = my_A_to_original_basis,
                                                                             a_b_to_original_basis   = my_b_to_original_basis,
                                                                             a_description_var_original_basis = my_original_data_description
                                                                             )

                # Add the pus for the cluster to the array of pus for the attribute
                my_clusters_pus.append(my_polyhedral_uncertainty_set)

            # Add the array of pus for the attribute to the dictionary with the arrays of pus for each final attribute
            my_pus_for_each_cluster_of_each_attribute[attr] = my_clusters_pus

        # Add the pus of each cluster of each attribute to the ScenariosPUS object
        self.cluster_pus = my_pus_for_each_cluster_of_each_attribute

    def perform_scenario_definition(self,a_prob_threshold,a_attributes=None):

        my_attributes = a_attributes
        if a_attributes is None:
            my_attributes = self.final_attributes

        my_scenarios, my_n_retained_scenarios = sc.get_scenarios(a_size_reduction_via_pca = self.size_reduction_via_pca,
                                                                 a_cluster_attribution = self.cluster_attribution,
                                                                 a_cluster_pus = self.cluster_pus,
                                                                 a_attributes = my_attributes,
                                                                 a_prob_threshold = a_prob_threshold)

        self.prob_threshold = a_prob_threshold 
        self.n_scenarios = my_n_retained_scenarios
        self.scenarios = my_scenarios

    #################
    # Get functions #
    #################

    def get_dataframe_original_data(self):
        return self.dataframe_original_data
    def get_original_data(self):
        return self.original_data
    def get_n_data_points(self):
        return self.n_data_points
    def get_size_reduction_via_pca(self):
        return self.size_reduction_via_pca

    ##################
    # Plot functions #
    ##################

    def plot_data_projection_principal_components(self,
                                                  a_stage,
                                                  a_attribute,
                                                  a_n_directions_to_plot,
                                                  a_n_cols_in_plot = 4,
                                                  a_title = 'Projection of the truncated data on the first principal directions',
                                                  a_file = None):

        return self.get_size_reduction_via_pca().plot_data_projection_principal_components(a_stage = a_stage,
                                                                                           a_attribute = a_attribute,
                                                                                           a_n_directions_to_plot = a_n_directions_to_plot,
                                                                                           a_n_cols_in_plot = a_n_cols_in_plot,
                                                                                           a_title = a_title,
                                                                                           a_file = a_file)

    def plot_explained_variance(
            self,
            a_stage,
            a_attribute,
            a_include_variance_threshold = True,
            a_title = None,
            a_x_label = 'Principal components',
            a_y_label = 'Explained variance ratio',
            a_file = None):
        
        return self.get_size_reduction_via_pca().plot_explained_variance(a_stage = a_stage,
                                                                         a_attribute = a_attribute,
                                                                         a_include_variance_threshold = a_include_variance_threshold,
                                                                         a_title = a_title,
                                                                         a_x_label = a_x_label,
                                                                         a_y_label = a_y_label,
                                                                         a_file = a_file)

    def plot_stage_explained_variance(self,
                                      a_stage,
                                      a_include_variance_threshold = True,
                                      a_title = None,
                                      a_x_label = 'Principal components',
                                      a_y_label = 'Explained variance ratio',
                                      a_file = None,
                                      a_n_cols_in_plot = 4,
                                      ):

        return self.get_size_reduction_via_pca().plot_stage_explained_variance(a_stage = a_stage,
                                                                               a_include_variance_threshold = a_include_variance_threshold,
                                                                               a_title = a_title,
                                                                               a_x_label = a_x_label,
                                                                               a_y_label = a_y_label,
                                                                               a_file = a_file,
                                                                               a_n_cols_in_plot = a_n_cols_in_plot)
    
    def plot_correlation_matrix(self,a_stage,a_attributes=None,a_attributes_rename=None,a_corr_all_values_att=False,a_ticks_att_unified=False,a_fig_size=None,a_dpi=None,a_n_digits=2,a_annotations=True,a_file=None):

        # Get the data
        my_data = None
        if a_stage == 0:
            # Get the original data
            my_data = self.original_data.copy()
        else:
            # Get the reduced data obtained via the size reduction via PCA
            my_data = (self.size_reduction_via_pca.get_stages_data()[a_stage]).get_data().copy()

        my_attributes = a_attributes
        if my_attributes is None:
            my_attributes = np.sort(list(my_data.keys()))

        my_data_to_plot = {}
        my_col_names = []
        for attr in my_attributes:
            if (attr in list(my_data.keys())):
                my_data_loc = my_data[attr]
                # Plot the correlation between each value of each attribute
                if a_corr_all_values_att:
                    for i in range(my_data_loc.shape[1]):
                        my_name = attr+f"[{i}]"
                        my_data_to_plot[my_name] = my_data_loc[:,i]

                        # Add the column name
                        my_col_names.append(my_name)
                # Plot the correlation between each attribute
                else:
                    # Flatten the 2D array to a 1D array
                    my_data_to_plot[attr] = my_data_loc.flatten() 

                    # Add the column name
                    my_col_names.append(attr)

        # Get a dataframe of the data for which we desire the correlation matrix
        my_dataframe_data = pd.DataFrame.from_dict(my_data_to_plot)

        # Order the columns as desired
        my_dataframe_data = my_dataframe_data[my_col_names]

        # Change the name of the columns if needed
        for key in a_attributes_rename:
            my_dataframe_data.columns = my_dataframe_data.columns.str.replace(key,a_attributes_rename[key])

        # Get the correlation matrix
        my_correlation_matrix = my_dataframe_data.corr()

        # Round all the values to two digits
        my_correlation_matrix.round(a_n_digits)

        # Create the figure
        fig, ax = None, None
        if a_fig_size is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(figsize=a_fig_size)

        if a_corr_all_values_att:

            x, y = np.mgrid[range(len(my_col_names)), range(len(my_col_names))]

            # ticks
            my_ticks = np.array([i for i in range(len(my_col_names))],float) + 0.5
            my_ticks_label = my_col_names

            if a_ticks_att_unified:

                my_n_values_per_attribute = []
                my_ticks_label = []
                for attr in my_attributes:
                    if (attr in list(my_data.keys())):
                        my_n_values_per_attribute.append(my_data[attr].shape[1])
                        my_ticks_label.append(attr)

                my_n_values_per_attribute = np.cumsum(my_n_values_per_attribute)
                my_ticks = (my_n_values_per_attribute + np.hstack(([0],my_n_values_per_attribute[:-1]))) / 2.0

            # Set the ticks and ticklabels for all axes
            plt.setp(ax, xticks=my_ticks, xticklabels=my_ticks_label, yticks=my_ticks, yticklabels=my_ticks_label)

            # Rotates X-Axis Ticks by 90-degrees
            plt.xticks(rotation = 90)

            c = ax.pcolor(x, y, my_correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
            ax.set_title('pcolor')
            fig.colorbar(c, ax=ax)
        else:
            sns.heatmap(my_correlation_matrix, annot=a_annotations, cmap='RdBu', vmin=-1, vmax=1)

        if a_file is not None:
            # Create the output folder if it does not exist
            Path(os.path.dirname(a_file)).mkdir(parents=True, exist_ok=True)

            # Save the figure
            if a_dpi is not None:
                plt.savefig(a_file, dpi=a_dpi, bbox_inches='tight')
            else:
                plt.savefig(a_file, bbox_inches='tight')

        return fig, ax
