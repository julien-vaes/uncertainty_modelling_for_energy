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

######################
# Clustering methods #
######################

def sort_clustering_attribution_with_regards_to_number_data_points(a_cluster_attribution):

    def sort_key(val):
        return len(val)
    
    # Sort the clusters in decreasing order with regards to the number of days in each cluster
    my_sorted_cluster_attribution = a_cluster_attribution.copy()
    my_sorted_cluster_attribution.sort(key=sort_key,reverse=True)

    return my_sorted_cluster_attribution

# ------- #
# K-means #
# ------- #

def get_kmeans_clusters(a_data,a_n_clusters):

    # Return an array where each index is a day and contains a vector which corresponds to all the attributes are concatenated, returns data (365,24*my_n_attributes)
    my_data_points = a_data

    if type(my_data_points) is list:
        my_data_points = np.concatenate(a_data, axis=1)

    # runs the K-means clustering
    kmeans = KMeans(a_n_clusters,init='k-means++', max_iter=10**3).fit(my_data_points)
    
    # gets an array where for each index i, its return the days index associated to the cluster i
    my_cluster_attribution = [np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)]

    return sort_clustering_attribution_with_regards_to_number_data_points(my_cluster_attribution)

# -----------------#
# Gaussian Mixture #
# -----------------#

def get_gaussian_mixture_clusters_attribution(a_data,a_n_clusters,a_method):
    
    # Get the data
    my_data_points = a_data

    my_fitted_mixture_model = 1
    if a_method == 'gmm':
        print("Fit: Gaussian Mixture")
        # Gaussian Mixture
        my_fitted_mixture_model = mixture.GaussianMixture(n_components=a_n_clusters, covariance_type="full", max_iter=10**6).fit(my_data_points)
    elif a_method == 'dpgmm':
        print("Fit: Bayesian Gaussian Mixture with a Dirichlet process prior")
        # Bayesian Gaussian Mixture with a Dirichlet process prior
        my_fitted_mixture_model = mixture.BayesianGaussianMixture(n_components=a_n_clusters, covariance_type="full", max_iter=10**6).fit(my_data_points)
    else:
        myErrorMessage = "Method '{}' not found".format(a_method)
        raise NameError(myErrorMessage)

    # the number of clusters after the fitting
    my_n_clusters = my_fitted_mixture_model.n_components
    
    # gets the cluster attribution of each data point
    my_labels = my_fitted_mixture_model.predict(my_data_points)
    
    # gets an array where for each index i, its return the days index associated to the cluster i
    my_cluster_attribution = [np.where(my_labels == i)[0] for i in range(my_n_clusters)]

    return sort_clustering_attribution_with_regards_to_number_data_points(my_cluster_attribution)

# ---------- #
# Clustering #
# ---------- #

def get_clusters_attribution(a_data,a_n_clusters,a_method='kmeans',a_seed=None):

    # Fix the seed
    if a_seed is not None:
        np.random.seed(a_seed)

    if a_method in ['gmm','dpgmm']:
        return get_gaussian_mixture_clusters_attribution(a_data,a_n_clusters,a_method)
    elif a_method == "kmeans":
        return get_kmeans_clusters(a_data,a_n_clusters)
    else:
        raise NameError(f"Method '{a_method}' unknown. Please select one of these options: 'kmeans', 'gmm' or 'dpgmm'.")
    return None


##################
# Plot functions #
##################

def generates_plot_days_cluster_attribution(a_cluster_attribution,a_title='',a_n_bins=50,a_x_ticks=None,a_x_ticks_labels=None,a_file='None',a_dpi=None):

    # gets the number of clusters
    n_clusters = len(a_cluster_attribution)
        
    myMinIndex = min([min(sub_arr) for sub_arr in a_cluster_attribution])
    myMaxIndex = max([max(sub_arr) for sub_arr in a_cluster_attribution])

    # Define the bin edges
    my_n_bins = a_n_bins
    bins = np.linspace(myMinIndex, myMaxIndex, my_n_bins+1)
    widths = np.diff(bins)

   # Compute the histograms
    counts = []
    for data in a_cluster_attribution:
        count, bins = np.histogram(data, bins=bins)
        counts.append(count)

    # Compute the new counts
    counts_transposed = np.transpose(np.array(counts))
    counts_frequency = [(lambda x: x/sum(q))(q) for q in counts_transposed]
    new_counts = np.transpose(counts_frequency.copy())
    for i in range(1,n_clusters):
        new_counts[i] += new_counts[i-1]

    # Create histogram
    colors = list(dict(mcolors.TABLEAU_COLORS, **mcolors.CSS4_COLORS).values())
    fig, ax = plt.subplots()
    for i in list(np.arange(n_clusters,0,-1)-1):
        ax.bar(bins[:-1], new_counts[i], widths, align='edge',color=colors[i])

    plt.title(a_title, size=18)
    plt.xlabel(r'Year', size=16)
    plt.ylabel(r'Proportion of days', size=16)
    if (a_x_ticks is not None) & (a_x_ticks_labels is not None):
        plt.xticks(a_x_ticks, a_x_ticks_labels)
    elif (a_x_ticks is not None):
        plt.xticks(a_x_ticks, a_x_ticks)
    plt.xlim(myMinIndex,myMaxIndex)
    plt.ylim((0, 1.0))

    if a_file is not None:
        # Create the output folder if it does not exist
        Path(os.path.dirname(a_file)).mkdir(parents=True, exist_ok=True)

        if a_dpi is not None:
            plt.savefig(a_file, dpi=a_dpi, bbox_inches='tight')
        else:
            plt.savefig(a_file, bbox_inches='tight')

    return fig, ax

def plot_quantiles(a_plot,aY):
    a_plot.plot(range(24),np.median(aY,axis=1),color='k',linewidth=1)
    myQuantilesToPlot = [25,50,74]
    my_colours = sns.color_palette("tab20c")
    for i in range(len(myQuantilesToPlot)):
        q = myQuantilesToPlot[i]
        myUpperQuant = np.array(np.percentile(aY, 100.0 - q/2.0, axis=1), dtype=float)
        myBottomQuant = np.array(np.percentile(aY, q/2.0, axis=1), dtype=float)
        a_plot.fill_between(range(24),myUpperQuant,myBottomQuant,color=my_colours[i])

def plot_clustering(a_cluster_attribution,
                    a_data,
                    a_attribute,
                    a_x_ticks,
                    a_x_ticks_labels,
                    a_n_bins=25,
                    a_file=None,
                    a_title=None,
                    a_n_cols=2
                    ):
    
    ####################################
    # Generates one plot per attribute #
    ####################################
    
    # Get the number of clusters
    my_n_clusters = len(a_cluster_attribution)

    # Get the data related to the attribute to plot
    my_attribute_data = a_data[a_attribute]
    my_n_data_points = my_attribute_data.shape[0]
    my_n_values_per_data_point = my_attribute_data.shape[1]
    
    # Get the number of lines for the plot
    my_n_lines = int(np.ceil(my_n_clusters / float(a_n_cols)))  
    
    # Initialise the plot which is a grid with the representation of the attributes for each cluster
    fig, axs = plt.subplots(my_n_lines,2*a_n_cols,figsize=(20,4*my_n_lines))

    # Add the title to the plot
    if a_title is not None:
        fig.suptitle(a_title)
    
    # Get the minimum and maximum of the attributes
    my_attr_min  = np.amin(my_attribute_data)
    my_attr_max  = np.amax(my_attribute_data)

     # Loop on the clusters
    for i, cluster_loc in enumerate(a_cluster_attribution):

        my_n_data_points_in_cluster = len(cluster_loc)

        # Title of the subplot
        myTitle = r'Cluster '+format(i)+': # days = '+format(my_n_data_points_in_cluster)
        
        my_line = i // a_n_cols 
        my_col = i % a_n_cols
        
        # Loop on the days in the cluster
        myY = np.empty(my_n_values_per_data_point*my_n_data_points_in_cluster,dtype=object)
        myYHour = np.empty((my_n_data_points_in_cluster,my_n_values_per_data_point),dtype=object)
        for j, my_data_point_index in enumerate(cluster_loc):
            myY[j*my_n_values_per_data_point:(j+1)*my_n_values_per_data_point] = my_attribute_data[my_data_point_index,:]
            myYHour[j,:]       = my_attribute_data[my_data_point_index,:]
        myYHour = myYHour.T
    
        # Get the axis on which to plot
        if my_n_lines == 1: 
            my_ax_values = axs[2*i]
            my_ax_hist   = axs[2*i+1]
        else:
            my_ax_values = axs[my_line,2*my_col]
            my_ax_hist   = axs[my_line,2*my_col+1]

        # Plot the quantiles of the data points 
        my_ax_values.set_title(myTitle)
        my_ax_values.set_ylim([my_attr_min,my_attr_max])
        my_ax_values.set_xlim([0,my_n_values_per_data_point-1])
        plot_quantiles(my_ax_values,myYHour)
        
        # Plot histogram with days 
        my_ax_hist.hist(a_cluster_attribution[i], density=False, bins=a_n_bins, range=(0,my_n_data_points-1))
        my_ax_hist.set_title(myTitle)
        my_ax_hist.set_xticks(a_x_ticks)
        my_ax_hist.set_xticklabels(a_x_ticks_labels)
    
    if a_file is not None:
        Path(os.path.dirname(a_file)).mkdir(parents=True, exist_ok=True)
        plt.savefig(a_file, bbox_inches='tight')

    # Close the figure
    plt.close(fig)

    return fig, axs

def plot_relationship_num_scenarios_prob_threshold(a_size_reduction_via_pca,
                                                   a_cluster_attribution,
                                                   a_attributes,
                                                   a_prob_threshold_vec,
                                                   a_prob_threshold_to_scatter=None,
                                                   a_file=None):

    # Compute the number of scenarios retained
    my_n_retained_scenarios_vec = [(lambda x: get_scenarios(a_size_reduction_via_pca,a_cluster_attribution,a_attributes,x)[1])(p) for p in a_prob_threshold_vec]

    if a_prob_threshold_to_scatter is not None:
        _, my_n_retained_scenarios = get_scenarios(a_size_reduction_via_pca,a_cluster_attribution,a_attributes,a_prob_threshold_to_scatter)

    # create the plot
    fig, ax = plt.subplots()
    ax.plot(a_prob_threshold_vec,my_n_retained_scenarios_vec)
    if a_prob_threshold_to_scatter is not None:
        ax.scatter([a_prob_threshold_to_scatter],[my_n_retained_scenarios],color='red')
    ax.set_xlabel("Probability threshold $ \\tilde{p} $",size=16)
    ax.set_ylabel("# of scenarios retained",size=16)

    if a_file is not None:
        # Create the output folder if it does not exist
        Path(os.path.dirname(a_file)).mkdir(parents=True, exist_ok=True)

        fig.savefig(a_file,bbox_inches='tight')

    return fig, ax
