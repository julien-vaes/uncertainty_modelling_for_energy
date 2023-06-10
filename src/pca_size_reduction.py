# Import packages
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
from scipy.sparse import csr_matrix
from scipy.stats.stats import pearsonr
import seaborn as sns
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

theGreen = sns.color_palette("Paired")[3]
theBlue = sns.color_palette("Paired")[1]

# Function to merge the linear transformation matrices

def merge_two_matrices(a_A,a_B):
    s_A = a_A.shape
    s_B = a_B.shape
    my_M = np.zeros((s_A[0]+s_B[0],s_A[1]+s_B[1]))
    my_M[:s_A[0],:s_A[1]] = a_A.copy()
    my_M[s_A[0]:,s_A[1]:] = a_B.copy()
    return my_M

def merge_list_matrices(a_list_matrices):
    if len(a_list_matrices) == 1:
        return a_list_matrices[0]
    my_M = merge_two_matrices(a_list_matrices[0],a_list_matrices[1])
    for i in range(len(a_list_matrices)):
        if i > 1:
             my_M = merge_two_matrices(my_M,a_list_matrices[i])
    return my_M

######################
# Database functions #
######################

def get_linear_transformations_between_original_and_standardized_data(a_data):
    """
    For each attribute return A_std, b_std, A_std_inv, b_std_inv such that
    X_standardized = X * A_std + b_std
    and
    X = X_standardized * A_std_inv + b_std_inv
    
    Args:
    a_data (dict): A_std dictionary containing the attributes to be standardized.
                   Each attribute is represented by a key and the corresponding value
                   is a numpy array containing the attribute data.
                   
    Returns:
    tuple: A tuple of four elements containing the following functions and dictionaries:
        - get_standardized_data_from_original_data: A function that returns standardized data given the original data
        - get_original_data_from_standardized_data: A function that returns original data given the standardized data
        - my_linear_transformation_original_to_standardized_data: A dictionary containing the linear transformation
                                                                  details to move from the original data to the
                                                                  standardized data
        - my_linear_transformation_standardized_to_original_data: A dictionary containing the linear transformation
                                                                  details to move from the standardized data to
                                                                  the original data
    """
    
    # Create two empty dictionaries to store the linear transformation details
    my_linear_transformation_original_to_standardized_data = {} 
    my_linear_transformation_standardized_to_original_data = {} 

    # Loop on each attribute
    for att_loc in a_data:
        my_x = a_data[att_loc]

        # Create a StandardScaler instance and fit-transform the data
        my_scaler = StandardScaler()
        my_scaler.fit_transform(my_x)

        # Extract the scaling matrix A_std and the mean vector b
        A_std = np.diag(1 / my_scaler.scale_)
        b_std = -my_scaler.mean_ / my_scaler.scale_

        # Add the details on the linear transformation to move from the original data to the standardized data: x_standardized = x * A_std + b_std 
        my_linear_transformation_original_to_standardized_data[att_loc] = (A_std, b_std, [(att_loc,len(b_std))])

        # Compute the inverse scaling matrix A_std_inv and the inverse mean vector b_std_inv
        A_std_inv = np.diag(my_scaler.scale_)
        b_std_inv = my_scaler.mean_

        # Add the details on the linear transformation to move from the standardized data to the original data: x = x_standardized * A_std_inv + b_std_inv
        my_linear_transformation_standardized_to_original_data[att_loc] = (A_std_inv, b_std_inv, [(att_loc,len(b_std_inv))])

    # Define the function to get standardized data from the original data 
    def get_standardized_data_from_original_data(a_original_data):
        """
        Returns standardized data given the original data.
        
        Args:
        a_original_data (dict): A dictionary containing the original data to be standardized.
                                Each attribute is represented by a key and the corresponding value
                                is a numpy array containing the attribute data.
                                
        Returns:
        dict: A dictionary containing the standardized data. Each attribute is represented by a key and the
              corresponding value is a numpy array containing the standardized data for that attribute.
        """
        my_standardized_data = {}
        for att_loc in a_original_data:
            A_std, b_std, _ = my_linear_transformation_original_to_standardized_data[att_loc]
            my_standardized_data[att_loc] = np.dot(a_original_data[att_loc], A_std) + b_std
        return my_standardized_data

    # Define the function to get original data from the standardized data 
    def get_original_data_from_standardized_data(a_standardized_data):
        """
        Returns original data given the standardized data.
        
        Args:
        a_standardized_data (dict): A dictionary containing the standardized data to be un-standardized.
                                    Each attribute is represented by a key and the corresponding value
                                    is a numpy array containing the attribute data.
                                
        Returns:
        dict: A dictionary containing the un-standardized data. Each attribute is represented by a key and the
              corresponding value is a numpy array containing the un-standardized data for that attribute.
        """
        my_original_data = {}
        for att_loc in a_standardized_data:
            A_std_inv, b_std_inv, _ = my_linear_transformation_standardized_to_original_data[att_loc]
            my_original_data[att_loc] = np.dot(a_standardized_data[att_loc], A_std_inv) + b_std_inv
        return my_original_data

    return get_standardized_data_from_original_data, get_original_data_from_standardized_data, my_linear_transformation_original_to_standardized_data, my_linear_transformation_standardized_to_original_data


#####################
# Class: stage data #
#####################


class StageData:
    """
    A class representing a stage of data in the size reduction via PCA procedure.
    
    Attributes:
    -----------
    stage : str
        The name of the stage.
    data : dict
        The data for this stage, represented as a dictionary where keys are attribute names and values are arrays of data.
    get_data_in_original_basis : function
        A function that can be used to project the data back into its original basis.
    """
    
    def __init__(self, a_stage, a_data, a_get_data_in_original_basis):
        """
        Initializes a new instance of the StageData class.
        
        Args:
        -----
        a_stage : int
            The stage.
        a_data : dict
            The data for this stage, represented as a dictionary where keys are attribute names and values are arrays of data.
        a_get_data_in_original_basis : function
            A function that can be used to project the data back into its original basis.
        """
        self.stage = a_stage
        self.data = a_data
        self.get_data_in_original_basis = a_get_data_in_original_basis
        self.dimensions = {k:a_data[k].shape[1] for k in a_data}
    
    def get_data(self):
        return self.data
    
    def get_stage(self):
        return self.stage

###########################################
# Class: step of a size reduction via PCA #
###########################################

class StepSizeReductionViaPCA:

    def __init__(self, a_step_index, a_attributes_to_merge_together, a_explained_variance_threshold, a_n_directions_threshold):

        ####################
        # Check for errors #
        ####################

        for att in list(a_attributes_to_merge_together.keys()):
            if att not in list(a_explained_variance_threshold.keys()):
                raise ValueError(f"No 'explained variance threshold' is provided for the attribute '{att}' for the step '{a_step_index}' of the size reduction via PCA.")
            if att not in list(a_n_directions_threshold.keys()):
                raise ValueError(f"No 'number of directions threshold' is provided for the attribute '{att}'.")

        ##################################################################
        # Initialise the details that defines the size reduction via PCA #
        ##################################################################

        # the step index
        self.step_index = a_step_index

        # the attributes to merge together for the step reduction, it corresponds to a dictionary where the keys correspond to the new attributes and the values the attributes to merge together
        self.attributes_to_merge_together = {key:a_attributes_to_merge_together[key] for key in a_attributes_to_merge_together}

        # the explained variance threshold for each new attribute, i.e. its influence the number of directions retained 
        self.explained_variance_threshold = a_explained_variance_threshold

        # the maximum of number of directions to retain in the PCA basis for each new attribute
        self.n_directions_threshold = a_n_directions_threshold

        # the number of directions retained in the PCA basis for each new attribute
        self.n_directions_retained = None

        ####################################################
        # Initialise details on the size reduction via PCA #
        ####################################################

        # Initialisation of the dictionary that contains the PCA for each attribute
        self.pca = None

        # Initialisation of the dictionary that contains the linear map to go from the original basis to the truncated PCA basis and conversely:
        #   y_trunc_pca = y_original_pca * A_trunc_pca + b_trunc_pca
        # and 
        #   y_original_pca = y_trunc_pca * A_trunc_pca_inv + b_trunc_pca_inv
        self.get_trunc_pca_basis_from_original_data = None
        self.get_original_data_from_trunc_pca_basis = None
        self.linear_transformation_input_to_trunc_pca_basis = None
        self.linear_transformation_trunc_pca_to_input_basis = None

        ## data before the size reduction via PCA 
        self.stage_data_before_size_reduction    = None
        self.dimension_before_size_reduction     = None
        self.dimension_before_size_reduction_sum = None

        ## data after the size reduction via PCA 
        self.stage_data_after_size_reduction = None
        self.dimension_after_size_reduction  = None

    #################
    # Get functions #
    #################

    def get_pca(self):
        return self.pca

    def get_attributes_to_merge_together(self):
        return self.attributes_to_merge_together

    def get_n_directions_retained(self):
        return self.n_directions_retained

    ###################
    # Class functions #
    ###################

    # Define the function to map from the input to the truncated PCA basis
    def get_trunc_pca_from_input_data(self,a_input_data):

        my_trunc_pca_data = {}
    
        # Get the details on which attributes must be merged together
        my_attributes_to_merge_together = self.attributes_to_merge_together
    
        # Project the input data into the truncated PCA basis by alphabetical order of the new attributes
        for att_new in np.sort(list(my_attributes_to_merge_together.keys())):
    
            # Get the data as a matrix
            my_x = [a_input_data[att_old] for att_old in my_attributes_to_merge_together[att_new]]
            my_x = np.concatenate(my_x, axis=1)
    
            # Get the data for the linear transformation
            A_trunc_pca, b_trunc_pca = self.linear_transformation_input_to_trunc_pca_basis[att_new]
    
            # Project the input data related to the new attribute 'attr_new' in the truncated PCA basis
            my_trunc_pca_data[att_new] = np.dot(my_x,A_trunc_pca) + b_trunc_pca
    
        return my_trunc_pca_data
    
    # Define the function to map from the truncated PCA basis to the input basis 
    def get_input_from_trunc_pca_data(self,a_trunc_pca_data):
        
        my_input_data = {}
    
        # Get the details on which attributes must be merged together
        my_attributes_to_merge_together = self.attributes_to_merge_together
    
        # Project the input data into the truncated PCA basis by alphabetical order of the new attributes
        for att_new in np.sort(list(a_trunc_pca_data.keys())):
    
            # Get the data as a matrix
            my_y = a_trunc_pca_data[att_new]
    
            # Get the data for the linear transformation
            A_trunc_pca_inv, b_trunc_pca_inv = self.linear_transformation_trunc_pca_to_input_basis[att_new]
    
            # Project the truncated PCA data related to the new attribute 'attr_new' in the input basis
            my_recovered_data = np.dot(my_y,A_trunc_pca_inv) + b_trunc_pca_inv
    
            my_index_att_old = 0
            for att_old in my_attributes_to_merge_together[att_new]:
                my_dimension_att_old = self.dimension_before_size_reduction[att_new][att_old]
                my_input_data[att_old] = my_recovered_data[:,my_index_att_old:my_index_att_old+my_dimension_att_old] 
                my_index_att_old += my_dimension_att_old
    
        return my_input_data

    # Define the function to map from to the truncated PCA basis to the original basis
    def get_data_in_original_basis(self,a_trunc_pca_data):
        return (self.stage_data_before_size_reduction).get_data_in_original_basis(self.get_input_from_trunc_pca_data(a_trunc_pca_data))
    
    def fill_stage_data_before_size_reduction(self,a_stage_data):

        self.stage_data_before_size_reduction    = a_stage_data
        self.dimension_before_size_reduction     = {k:{key_loc:((a_stage_data.get_data())[key_loc].shape[1]) for key_loc in self.attributes_to_merge_together[k]} for k in self.attributes_to_merge_together}
        self.dimension_before_size_reduction_sum = {k:sum(list(self.dimension_before_size_reduction[k].values())) for k in self.attributes_to_merge_together}

    def compute_pca_for_each_group_of_attributes(self,a_stage_data_before_size_reduction):

        # Get the data and stage
        my_data = a_stage_data_before_size_reduction.get_data()

        # Add the stage data, which serves as the input data to the step size reduction via PCA object
        self.fill_stage_data_before_size_reduction(a_stage_data_before_size_reduction)

        # Initialise the variables that contain the details on the size reduction via PCA
        self.pca = {}
        self.dimension_after_size_reduction = {}
        my_data_next_stage = {}
        self.n_directions_retained = {}

        # Initialise the linear transformation variables
        self.get_trunc_pca_basis_from_original_data = {}
        self.get_original_data_from_trunc_pca_basis = {}
        self.linear_transformation_input_to_trunc_pca_basis = {}
        self.linear_transformation_trunc_pca_to_input_basis = {}

        # Compute the PCA reduction by alphabetical order of the new attributes
        for att_new in np.sort(list(self.attributes_to_merge_together.keys())):

            # Get the data of all the attributes to merge
            my_x = [my_data[att_old] for att_old in self.attributes_to_merge_together[att_new]]
            my_x = np.concatenate(my_x, axis=1)

            # Compute the PCA
            my_pca = PCA(my_x.shape[1])
            my_pca.fit(my_x)
            self.pca[att_new] = my_pca

            # Get the data formulated in the PCA basis
            my_z = my_pca.transform(my_x)

            ##################
            # Size reduction #
            ##################

            # Initialise the number of PCA directions to keep
            my_n_directions_retained = my_x.shape[1]

            # Check if a maximum number of directions is given
            if self.n_directions_threshold[att_new] is not None:
                my_n_directions_retained = self.n_directions_threshold[att_new]

            # Check whether there is the condition on the explained variance ratio
            if self.explained_variance_threshold[att_new] is not None:

                # gets the number of principal direction to keep to reach the variance threshold
                my_explained_variance_ratio = my_pca.explained_variance_ratio_

                # boolean to tell whether there exists an index such that explained variance ratio is smaller than than the threshold
                my_indices_satisfying_variance_ratio = np.argwhere(my_explained_variance_ratio < self.explained_variance_threshold[att_new])

                if len(my_indices_satisfying_variance_ratio) > 0:
                    my_n_directions_retained = min(my_n_directions_retained,my_indices_satisfying_variance_ratio[0][0])

            # Add the data of the new attribute to the data dictionary of the next stage    
            self.n_directions_retained[att_new] = my_n_directions_retained
            my_data_next_stage[att_new] = my_z[:,:my_n_directions_retained]
            self.dimension_after_size_reduction[att_new] = my_data_next_stage[att_new].shape[1]

            ####################################
            # Linear transformation definition #
            ####################################

            # ----------------------------------- #
            # Input basis --> Truncated PCA basis #
            # ----------------------------------- #

            # Matrix 'M_trunc' to keep only the first 'my_n_directions_retained' components
            M_trunc = np.zeros((my_x.shape[1], my_n_directions_retained), float)
            np.fill_diagonal(M_trunc, 1.0)
            A_trunc_pca = np.dot(my_pca.components_.T,M_trunc)
            b_trunc_pca = np.dot(-my_pca.mean_,A_trunc_pca)
            self.linear_transformation_input_to_trunc_pca_basis[att_new] = (A_trunc_pca,b_trunc_pca)

            # ----------------------------------- #
            # Truncated PCA basis --> Input basis #
            # ----------------------------------- #

            # Matrix 'M_trunc_inv' to add the missing components to have a full matrix
            M_trunc_inv = M_trunc.T
            A_trunc_pca_inv = np.dot(M_trunc_inv,my_pca.components_)
            b_trunc_pca_inv = my_pca.mean_
            self.linear_transformation_trunc_pca_to_input_basis[att_new] = (A_trunc_pca_inv,b_trunc_pca_inv)

        # Add the stage data after the size reduction via PCA
        self.stage_data_after_size_reduction = StageData(a_stage_data_before_size_reduction.get_stage()+1,self.get_trunc_pca_from_input_data(a_stage_data_before_size_reduction.get_data()),self.get_data_in_original_basis)

        return self.stage_data_after_size_reduction

#################################
# Class: Size reduction via PCA #
#################################

class SizeReductionViaPCA:

    def __init__(self, a_original_data, a_n_data_points, a_attributes_to_merge_together, a_explained_variance_threshold, a_n_directions_threshold):

        # the original data
        self.original_data = a_original_data

        # the number of data points
        self.n_data_points = a_n_data_points

        # number of steps to execute for the data size reduction
        self.n_steps = len(a_attributes_to_merge_together)

        # Attributes to merge together for each step, i.e. for each step we specify a dictionary where the keys correspond to the new aggregate attribute and where the values are the attributes to merge together sorted alphabetically
        self.attributes_to_merge_together = []
        for i, _ in enumerate(a_attributes_to_merge_together):
            my_step_dict = {}
            for key in a_attributes_to_merge_together[i]:
                my_attr_previous_step = a_attributes_to_merge_together[i][key]
                my_attr_previous_step = sorted(my_attr_previous_step, reverse=False)
                my_step_dict[key] = my_attr_previous_step
            self.attributes_to_merge_together.append(my_step_dict)

        # list of the attributes at each stage
        my_attributes = []
        my_attributes.append(list(self.original_data.keys()))
        for attributes_to_merge_together_loc in a_attributes_to_merge_together:
            my_attributes.append(list(attributes_to_merge_together_loc.keys()))
        self.attributes = my_attributes

        # Get dictionary with as keys the new attributes and values a list of all initial attributes used to generate the new attribute
        my_mapping_attr_and_attr_initial = [{k:[k] for k in self.attributes[0]}]
        for i, _ in enumerate(self.attributes_to_merge_together):
                my_step_dict = {}
                for key in self.attributes_to_merge_together[i]:
                    my_attributes_init = []
                    for attr_prev in self.attributes_to_merge_together[i][key]:
                        my_attributes_init += my_mapping_attr_and_attr_initial[i][attr_prev]
                    my_step_dict[key] = my_attributes_init
                my_mapping_attr_and_attr_initial.append(my_step_dict)
        self.mapping_attr_and_attr_initial = my_mapping_attr_and_attr_initial.copy()

        # explained variance threshold for each step, i.e. for each step we specify a dictionary where the keys correspond to the new aggregate attribute and where the values are is the explained variance threshold
        self.explained_variance_threshold = []
        for i in range(self.n_steps):
            my_stage_explained_variance_threshold = {k:None for k in my_attributes[i]}
            my_stage_explained_variance_threshold.update(a_explained_variance_threshold[i])
            self.explained_variance_threshold.append(my_stage_explained_variance_threshold)

        # maximum number of PCA directions to keep in the PCA basis for each step, i.e. for each step we specify a dictionary where the keys correspond to the new aggregate attribute and where the values are is the maximum number of directions
        self.n_directions_threshold = []
        for i in range(self.n_steps):
            my_stage_n_directions_threshold = {k:None for k in my_attributes[i]}
            my_stage_n_directions_threshold.update(a_n_directions_threshold[i])
            self.n_directions_threshold.append(my_stage_n_directions_threshold)

        ####################
        # Check for errors #
        ####################

        my_list_attributes = list(self.original_data.keys())
        for i in range(self.n_steps):
            my_list_attributes_to_reduce_size = my_list_attributes.copy()
            for key in self.attributes_to_merge_together[i]:
                for att in self.attributes_to_merge_together[i][key]:
                    if att not in my_list_attributes:
                        raise ValueError(f"The attribute '{att}', which will be part of the size reduction to create the new attribute '{key}' is not an attribute at stage {i-1}, which contains\n {my_list_attributes}.")
                    else:
                        if att not in my_list_attributes_to_reduce_size:
                            raise ValueError(f"The attribute '{att}' is used to defined 2 or more new attributes, one of them being '{key}'.")
                        else:
                            my_list_attributes_to_reduce_size.remove(att)               
            # Update the list of the attributes that are ignored and considered in the PCA size reduction
            if i == 0:
                self.attributes_ignored = my_list_attributes_to_reduce_size
                if len(my_list_attributes_to_reduce_size) > 0:
                    print(f"WARNING: Some initial attributes are ignored in the dimension reduction: '{my_list_attributes_to_reduce_size}'.")
                self.attributes_considered = list(set(my_list_attributes) - set(my_list_attributes_to_reduce_size))
            # Check whether any attribute is lost while performing the size reduction
            else:
                if len(my_list_attributes_to_reduce_size) > 0:
                    raise ValueError(f"Some attributes are lost in the size reduction: e.g. '{my_list_attributes_to_reduce_size}' at step {i+1}.")
            my_list_attributes = list(self.attributes_to_merge_together[i].keys())

        ###############################################
        # Initialise the value for the size reduction #
        ###############################################

        # Get the functions to move between the original and standardized data
        self.get_standardized_data_from_original_data, self.get_original_data_from_standardized_data, self.linear_transformation_original_to_standardized_data, self.linear_transformation_standardized_to_original_data = get_linear_transformations_between_original_and_standardized_data(self.original_data)

        #---------------#
        # Initial stage #
        #---------------#

        # Standardize the data and get the function that map the standardized data to the original data 
        my_data_stage_0 = self.get_standardized_data_from_original_data(self.original_data)

        # Initialise the vector that will contain the data for each stage
        self.stages_data = [StageData(0,my_data_stage_0,self.get_original_data_from_standardized_data)]

        #-----------------------------------------------#
        # Details on the steps to do for size reduction #
        #-----------------------------------------------#

        # Initialise the list that contains the details on each step of the size reduction
        self.steps_size_reduction_via_pca = []
        for i in range(self.n_steps):
            self.steps_size_reduction_via_pca.append(StepSizeReductionViaPCA(i,a_attributes_to_merge_together[i],a_explained_variance_threshold[i], a_n_directions_threshold[i]))

        ##############################
        # Perform the size reduction #
        ##############################

        self.perform_size_reduction()

        #####################################################################
        # Linear mapping:   x_original_basis =  x_trunc_pca * A_inv + b_inv #
        #####################################################################

        self.linear_transformation_trunc_pca_to_original_basis = None
        self.get_original_data_from_stage_data = None
        self.compute_linear_mappings_from_trunc_pca_to_original_basis()

    #################
    # Get functions #
    #################

    def get_original_data(self):
        return self.original_data
    def get_n_steps(self):
        return self.n_steps
    def get_attributes(self):
        return self.attributes
    def get_attributes_to_merge_together(self):
        return self.attributes_to_merge_together
    def get_explained_variance_threshold(self):
        return self.explained_variance_threshold
    def get_n_directions_threshold(self):
        return self.stages_data
    def get_stages_data(self):
        return self.stages_data
    def get_data(self,a_stage):
        return (self.get_stages_data()[a_stage]).get_data()
    def get_steps_size_reduction_via_pca(self):
        return self.steps_size_reduction_via_pca
    def get_attr_data_final_stage(self,a_attr):
        return self.get_data(-1)[a_attr]
    def get_attr_cluster_data_final_stage(self,a_attr,a_data_points_indices):
        return (self.get_attr_data_final_stage(a_attr))[a_data_points_indices,:]

    ####################
    # Class: functions #
    ####################

    def perform_size_reduction(self):

        # Re-initialise the stages data (by deleting those that would have been computed)
        self.stages_data = [self.get_stages_data()[0]]

        # Compute iteratively every step of the size reduction
        for i in range(self.get_n_steps()):
            my_current_stage_data = self.get_stages_data()[-1]
            my_next_stage_data = self.steps_size_reduction_via_pca[i].compute_pca_for_each_group_of_attributes(my_current_stage_data)
            self.stages_data.append(my_next_stage_data)

    # Define the function that executes the mapping from the truncated PCA basis to the original basis
    def get_function_get_original_data_from_stage_data(self,a_stage):

        my_linear_transformation_trunc_pca_to_original_basis =  self.linear_transformation_trunc_pca_to_original_basis[a_stage]

        def get_original_data_from_stage_data_loc(a_data):

            my_data_in_original_basis = {}
            for attr in a_data:
    
                my_A_inv_loc, my_b_inv_loc, my_original_data_description = my_linear_transformation_trunc_pca_to_original_basis[attr]
    
                # Compute the reverse linear map from the truncated PCA basis to the original basis
                my_x = np.dot(a_data[attr],my_A_inv_loc) + my_b_inv_loc
    
                my_idx = 0
                for (original_attr,my_n_values_original_attr) in my_original_data_description:
                    my_data_in_original_basis[original_attr] = my_x[:,my_idx:my_idx+my_n_values_original_attr]
                    my_idx += my_n_values_original_attr

            return my_data_in_original_basis 
        return get_original_data_from_stage_data_loc

    def compute_linear_mappings_from_trunc_pca_to_original_basis(self):

        # Initialise the array with the linear transformation for each new attribute of each stage
        self.linear_transformation_trunc_pca_to_original_basis = []
        self.get_original_data_from_stage_data = []

        for stage in range(self.get_n_steps()+1):

            # Get the step in the size reduction
            step = stage-1

            if stage == 0:
                # Initial stage
                self.linear_transformation_trunc_pca_to_original_basis.append(self.linear_transformation_standardized_to_original_data)
                self.get_original_data_from_stage_data.append(self.get_original_data_from_standardized_data)
            else:

                my_stage_linear_transformation_trunc_pca_to_input_basis = self.steps_size_reduction_via_pca[step].linear_transformation_trunc_pca_to_input_basis

                my_stage_linear_transformation_trunc_pca_to_original_basis = {}
                for attr_stage in self.attributes_to_merge_together[step]:

                    # Get the linear transformation that maps from the pca truncated basis to the input basis (previous stage basis)
                    A_trunc_pca_inv, b_trunc_pca_inv = my_stage_linear_transformation_trunc_pca_to_input_basis[attr_stage]

                    # Get the linear transformation to go from the input basis to the original basis
                    my_input_attributes = self.attributes_to_merge_together[step][attr_stage]

                    # Get the A_inv, b_inv of the previous step
                    my_A_inv_previous_step = []
                    my_b_inv_previous_step = []
                    for input_attr in my_input_attributes:
                        my_A_inv, my_b_inv, _ = self.linear_transformation_trunc_pca_to_original_basis[stage-1][input_attr]
                        my_A_inv_previous_step.append(my_A_inv)
                        my_b_inv_previous_step.append(my_b_inv)

                    # Create the inverse matrix for all the input attributes:
                    # input_attributes_origin_basis = input_attributes_input_basis * [ A_inv_1    0      ...   0     , + [ b_inv_1,
                    #                                                                    0     A_inv_2   ...   0     ,     b_inv_2,
                    #                                                                    .        .            .              .
                    #                                                                    .        .            .              . 
                    #                                                                    .        .            .              . 
                    #                                                                    0        0          A_inv_n ]     b_inv_n]
                    my_A_inv_input_basis = merge_list_matrices(my_A_inv_previous_step)
                    my_b_inv_input_basis = np.concatenate(my_b_inv_previous_step)

                    # Get the linear transformation of the new attribute
                    my_A_inv = np.dot(A_trunc_pca_inv,my_A_inv_input_basis)
                    my_b_inv = np.dot(b_trunc_pca_inv,my_A_inv_input_basis) + my_b_inv_input_basis

                    # Add the linear transformation to the dictionary
                    my_stage_linear_transformation_trunc_pca_to_original_basis[attr_stage] = my_A_inv.copy(), my_b_inv.copy(), [(attr_loc,self.get_stages_data()[0].dimensions[attr_loc]) for attr_loc in self.mapping_attr_and_attr_initial[stage][attr_stage]]

                # Add the linear transformation to the dictionary
                self.linear_transformation_trunc_pca_to_original_basis.append(my_stage_linear_transformation_trunc_pca_to_original_basis.copy())

                # Define the function that executes the mapping from the truncated PCA basis to the original basis
                my_function_get_original_data_from_stage_data =  self.get_function_get_original_data_from_stage_data(stage)
                self.get_original_data_from_stage_data.append(my_function_get_original_data_from_stage_data)

    def get_data_in_original_basis(self,a_stage,a_data):

        # Get the data of the corresponding stage
        my_stage_data = self.get_stages_data()[a_stage]

        # Project the truncated stage data back into the original basis
        my_recovered_data = my_stage_data.get_data_in_original_basis(a_data)

        return my_recovered_data

    def recover_all_data(self,a_stage):

        # Get the data of the corresponding stage
        my_stage_data = self.get_stages_data()[a_stage]

        # Project the truncated stage data back into the original basis
        my_recovered_data = self.get_data_in_original_basis(a_stage,my_stage_data.get_data())

        return my_recovered_data

    def recover_data(self,a_attributes,a_stage):

        # Get the recover data for all attributes
        my_recovered_data = self.recover_all_data(a_stage)

        return {att:my_recovered_data[att] for att in a_attributes}

    #########################
    # Class: plot functions #
    #########################

    def plot_data_projection_principal_components(self,
                                                  a_stage,
                                                  a_attribute,
                                                  a_n_directions_to_plot,
                                                  a_n_cols_in_plot = 4,
                                                  a_title = 'Projection of the truncated data on the first principal directions',
                                                  a_file = None):

        # Gets the (reduced) data of the given stage
        my_data = (self.get_stages_data()[a_stage]).get_data()

        # computes the number of rows in the plot
        my_n_rows_in_plot = a_n_directions_to_plot // a_n_cols_in_plot + (a_n_directions_to_plot % a_n_cols_in_plot + a_n_cols_in_plot - 1) // a_n_cols_in_plot

        # the subplot
        fig, axs = plt.subplots(my_n_rows_in_plot,a_n_cols_in_plot,figsize=(15 * a_n_cols_in_plot, 10 * my_n_rows_in_plot))
        fig.suptitle(a_title,size=50)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)

        for i in range(my_n_rows_in_plot):
            for j in range(a_n_cols_in_plot):
                myDirectionIndex = i*a_n_cols_in_plot + j
                if myDirectionIndex < a_n_directions_to_plot:
                    axs[i,j].hist(my_data[a_attribute][:,i*a_n_cols_in_plot+j], 50, density=True, facecolor=theGreen)
                    axs[i,j].set_title(f"Direction {i*a_n_cols_in_plot+j+1}",size=50)

        if a_file is not None:
            plt.savefig('outputs/images/projection_princ_dir.eps',bbox_inches='tight')
        return fig, axs

    def plot_explained_variance_help(
            self,
            fig,
            ax,
            a_stage,
            a_attribute,
            a_include_variance_threshold = True,
            a_title = None,
            a_x_label = 'Principal components',
            a_y_label = 'Explained variance ratio',
            ):

        # Get the stage size reduction via PCA 
        my_stage_size_reduction_via_pca = self.get_steps_size_reduction_via_pca()[a_stage-1]

        # Retrieve the explained variance ratio for the given stage and attribute to analyse
        my_variance_ratio = (my_stage_size_reduction_via_pca.get_pca())[a_attribute].explained_variance_ratio_

        # Sort the values of the explained variance ratio and calculate the corresponding index positions
        x_interpol = np.sort(np.array(my_variance_ratio))
        y_interpol = np.arange(len(my_variance_ratio)-1, -1, -1, dtype=float)

        # Create a plot with the number of principal components and explained variance ratio
        ax.plot(y_interpol,x_interpol)
        ax.set_yscale('log')

        if a_include_variance_threshold:
            # Calculate the number of components to retain to reach the desired threshold value
            my_variance_threshold = self.explained_variance_threshold[a_stage-1][a_attribute]
            if my_variance_threshold is not None:
                x_interp = [my_stage_size_reduction_via_pca.explained_variance_threshold[a_attribute]]
                y_interp = np.interp(x_interp, x_interpol, y_interpol)
                y_interp = np.vectorize(lambda x: math.floor(x))(y_interp)
                n_dir_variance_threshold = y_interp[0]
                ax.scatter(n_dir_variance_threshold,my_variance_threshold,label=f'Threshold = {my_variance_threshold:.1e}',color='r') 
                ax.legend()

        # Add labels to the plot and save it as an EPS file
        ax.set_xlabel(a_x_label,size=14)
        ax.set_ylabel(a_y_label,size=14)

        if a_title is not None:
            ax.set_title(a_title,size=20)

    def plot_explained_variance(
            self,
            a_stage,
            a_attribute,
            a_include_variance_threshold = True,
            a_title = None,
            a_x_label = 'Principal components',
            a_y_label = 'Explained variance ratio',
            a_file = None):

        # Initialise the plot
        fig, axs = plt.subplots(1,1)

        # Fill the plot
        self.plot_explained_variance_help(
                fig,
                axs,
                a_stage,
                a_attribute,
                a_include_variance_threshold,
                a_title,
                a_x_label,
                a_y_label,
                )

        if a_file is not None:
            plt.savefig(a_file)

        return fig, axs

    def plot_stage_explained_variance(
            self,
            a_stage,
            a_include_variance_threshold = True,
            a_title = None,
            a_x_label = 'Principal components',
            a_y_label = 'Explained variance ratio',
            a_file = None,
            a_n_cols_in_plot = 4,
            ):

        # Compute the number of rows in the plot
        my_attributes = self.attributes[a_stage]
        my_n_attributes_to_plot = len(my_attributes)
        my_n_rows_in_plot = my_n_attributes_to_plot // a_n_cols_in_plot + (my_n_attributes_to_plot % a_n_cols_in_plot + a_n_cols_in_plot - 1) // a_n_cols_in_plot

        # the subplot
        fig, axs = plt.subplots(my_n_rows_in_plot,a_n_cols_in_plot,figsize=(10 * a_n_cols_in_plot, 6 * my_n_rows_in_plot))
        if a_title is not None:
            fig.suptitle(a_title)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)

        for i in range(my_n_rows_in_plot):
            for j in range(a_n_cols_in_plot):
                my_attribute_index = i*a_n_cols_in_plot + j
                if my_attribute_index < my_n_attributes_to_plot:
                    my_attribute = my_attributes[my_attribute_index]
                    my_ax = None
                    if my_n_rows_in_plot > 1:
                        my_ax = axs[i,j]
                    else:
                        my_ax = axs[my_attribute_index]

                    # fill the plot
                    self.plot_explained_variance_help(
                            fig,
                            my_ax,
                            a_stage,
                            a_attribute = my_attribute,
                            a_include_variance_threshold = a_include_variance_threshold,
                            a_x_label = a_x_label,
                            a_y_label = a_y_label,
                            a_title = my_attribute
                            )

        if a_file is not None:
            plt.savefig('outputs/images/projection_princ_dir.eps',bbox_inches='tight')

        return fig, axs

    def plot_data_help(self,a_data,a_attribute,a_n_data_points_to_plot,a_y_label,a_x_label,a_title,a_dpi,a_file):

        my_n_data_points_to_plot = a_n_data_points_to_plot
        if my_n_data_points_to_plot is None:
            my_n_data_points_to_plot = a_data.shape[0]

        # Get the data to plot
        my_data_to_plot = a_data[:my_n_data_points_to_plot]

        if a_attribute not in self.attributes[0]:
            raise ValueError(f"The attribute '{a_attribute}' is not part of the original attributes, i.e. choose one of the following attributes \n{self.attributes[0]}")

        # finding scaling parameters based on the original data
        my_attribute_min = np.min(self.original_data[a_attribute])
        my_attribute_max = np.max(self.original_data[a_attribute])
        s = 0.1 *(my_attribute_max - my_attribute_min)

        # create the plot
        fig, axs = plt.subplots()
        if a_dpi is not None:
            fig, axs = plt.subplots(dpi=a_dpi)

        axs.plot(my_data_to_plot.T,marker='.')
        axs.set_title(a_title)
        axs.set_xlabel(a_x_label)
        axs.set_ylabel(a_y_label)
        axs.set_ylim([my_attribute_min - s, my_attribute_max + s])

        # Save the plot
        if a_file is not None:
            plt.savefig(a_file, bbox_inches='tight')

        return fig, axs

    def plot_original_data(self,a_attribute,a_n_data_points_to_plot=None,a_data_points_indices=None,a_y_label='',a_x_label=r'Time (h)',a_title=r'Original data',a_dpi=None,a_file=None):
        my_data_to_plot = None
        if a_data_points_indices is None:
            my_data_to_plot = self.original_data[a_attribute]
        else:
            my_data_to_plot = self.original_data[a_attribute][a_data_points_indices]
        return self.plot_data_help(my_data_to_plot,a_attribute,a_n_data_points_to_plot,a_y_label,a_x_label,a_title,a_dpi,a_file)

    def plot_data_in_original_basis(self,a_data,a_attribute,a_n_data_points_to_plot=None,a_y_label='',a_x_label=r'Time (h)',a_title=r'Original data',a_dpi=None,a_file=None):
        my_data_to_plot = a_data[a_attribute]
        return self.plot_data_help(my_data_to_plot,a_attribute,a_n_data_points_to_plot,a_y_label,a_x_label,a_title,a_dpi,a_file)

    def plot_recovered_data(self,a_attribute,a_stage,a_n_data_points_to_plot=None,a_y_label='',a_x_label=r'Time (h)',a_title=r'Recovered data',a_dpi=None,a_file=None):
        my_data_to_plot = self.recover_data([a_attribute],a_stage)[a_attribute]
        return self.plot_data_help(my_data_to_plot,a_attribute,a_n_data_points_to_plot,a_y_label,a_x_label,a_title,a_dpi,a_file)
