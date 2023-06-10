# imports packages

from collections import Counter, defaultdict
import copy
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numbers
import numpy as np
from numpy import unravel_index
import pandas as pd
from pandas import *
from pathlib import Path
import progressbar
from pyomo.environ import *
import pyomo.environ as pyo
import random
from scipy import linalg
from scipy.optimize import linprog
from scipy.stats.stats import pearsonr
import seaborn as sns
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import seaborn as sns
# import pypoman

import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

import pca_size_reduction as psr

#############################################################################################
# Functions to get standardized form for the constraints of a linear program, i.e. A*x <= b #
#############################################################################################

def get_variable_constraint_standard_form(a_var,a_variable_names):
    
    my_var_index = a_variable_names.index(a_var.name)
    
    my_A_indices = []
    my_A_coeffs  = []
    my_A_names   = []
    my_b         = []
    
    if a_var.has_ub():
        my_A_indices.append([my_var_index])
        my_A_coeffs.append([1.0])
        my_A_names.append([a_var.name])
        my_b.append(float(a_var.ub))
    if a_var.has_lb():
        my_A_indices.append([my_var_index])
        my_A_coeffs.append([-1.0])
        my_A_names.append([a_var.name])
        my_b.append(-1.0*float(a_var.lb))
    return my_A_indices, my_A_coeffs, my_A_names, my_b

def get_constraint_standard_form(a_con,a_variable_names):
    
    # Function to help to extract the coefficents of the constraint
    
    def get_monomial(a_monomial_expression,a_coeff=1.0):
        my_args = a_monomial_expression.args
        my_variable_name_loc = my_args[1].name
        my_variable_coefficient_loc = a_coeff * float(my_args[0])
        return {my_variable_name_loc: my_variable_coefficient_loc}
    
    def get_general_var_data(a_general_var_data_expression,a_coeff=1.0):
        my_variable_name_loc = a_general_var_data_expression.name
        my_variable_coefficient_loc = a_coeff
        return {my_variable_name_loc: my_variable_coefficient_loc}
    
    def add_local_var_to_general(a_variables,a_variables_loc):
        
        if a_variables is None:
            return a_variables_loc
        
        my_variables = a_variables.copy()
        my_variables_names = list(my_variables.keys())
        
        for var_loc in a_variables_loc:
            if var_loc in my_variables_names:
                my_variables[var_loc] += a_variables_loc[var_loc]
            else:
                my_variables[var_loc] = a_variables_loc[var_loc]
        return my_variables  

    def get_sum_expression(a_sum_expression,a_allow_product=True):
        
        my_variables = None
        my_constant = 0.0
        
        my_args = a_sum_expression.args
        for arg in my_args:
            my_constant_loc, my_variables_loc = get_coeff_expression(arg)
            my_constant += my_constant_loc
            my_variables = add_local_var_to_general(my_variables,my_variables_loc)
               
        return my_constant, my_variables
        
    def get_product_expression(a_product_expression):
        
        my_args = a_product_expression.args
        
        if (not isinstance(my_args[0],numbers.Number)) & (not isinstance(my_args[1],numbers.Number)):
            raise ValueError(f'The product expression {a_product_expression} must have at least one term that is a number in a linear program. The elements are of type {type(my_args[0])} and {type(my_args[1])}.')
            
        if (not 'SumExpression' in str(type(my_args[0]))) & (not 'SumExpression' in str(type(my_args[1]))):
            raise ValueError(f'The product expression {a_product_expression} must have at least one term that is a SumExpression in a linear program. The elements are of type {type(my_args[0])} and {type(my_args[1])}.')
        
        my_prod_coeff = None
        my_constant_loc = None
        my_variables_loc = None
        if (isinstance(my_args[0],numbers.Number)):
            my_prod_coeff = float(my_args[0])
            my_constant_loc, my_variables_loc = get_sum_expression(my_args[1],a_allow_product=False)
        else:
            my_prod_coeff = float(my_args[1])
            my_constant_loc, my_variables_loc = get_sum_expression(my_args[0],a_allow_product=False)
            
        # Compute the product
        my_constant_loc *= my_prod_coeff
        
        for var in my_variables_loc:
            my_variables_loc[var] *= my_prod_coeff
            
        return my_constant_loc, my_variables_loc
    
    def get_coeff_expression(a_expression,a_first_level=True):
        my_constant = 0.0
        my_variables = {}
        arg = a_expression
        if isinstance(arg,numbers.Number):
            my_constant += float(arg)
        elif 'GeneralVarData' in str(type(arg)):
            my_variables_loc = get_general_var_data(arg)
            my_variables = add_local_var_to_general(my_variables,my_variables_loc)
        elif 'MonomialTermExpression' in str(type(arg)):
            my_variables_loc = get_monomial(arg)
            my_variables = add_local_var_to_general(my_variables,my_variables_loc)
        elif 'SumExpression' in str(type(arg)):
            my_constant_loc, my_variables_loc = get_sum_expression(arg)
            my_constant += my_constant_loc
            my_variables = add_local_var_to_general(my_variables,my_variables_loc)
        elif 'ProductExpression' in str(type(arg)):
            my_constant_loc, my_variables_loc = get_product_expression(arg)
            my_constant += my_constant_loc
            my_variables = add_local_var_to_general(my_variables,my_variables_loc)
        elif 'NegationExpression' in str(type(arg)):
                my_constant_loc, my_variables_loc = get_coeff_expression(arg.args[0])
                 
                # Add the negative term
                my_constant_loc *= -1.0
                for k in my_variables_loc:
                    my_variables_loc[k] *= -1.0
                    
                my_constant += my_constant_loc
                my_variables = add_local_var_to_general(my_variables,my_variables_loc)
        else:
            raise NameError(f'The term {arg} is not taken into account in the NegationExpression.')
        
        return my_constant, my_variables
        
    # Extract the coefficents of the constraint
    my_constant, my_variables = get_coeff_expression(a_con.body)
    
    my_variables_names = list(my_variables.keys())
    my_variables_coefficients = [my_variables[var] for var in my_variables_names]
    my_variables_indices = [a_variable_names.index(var) for var in my_variables_names]
    
    my_A_indices = []
    my_A_coeffs  = []
    my_A_names   = []
    my_b         = []
    
    if a_con.has_ub():
        my_A_indices.append(my_variables_indices)
        my_A_coeffs.append(np.array(my_variables_coefficients))
        my_A_names.append(my_variables_names)
        my_b.append(float(a_con.upper())-my_constant)
    if a_con.has_lb():
        my_A_indices.append(my_variables_indices)
        my_A_coeffs.append(-1.0*np.array(my_variables_coefficients))
        my_A_names.append(my_variables_names)
        my_b.append(-1.0*(float(a_con.lower())-my_constant))
                
    return my_A_indices, my_A_coeffs, my_A_names, my_b

def get_standardized_constraints(a_model):
    
    # Get all the active constraints in the model
    my_constraints = list(a_model.component_data_objects(pyo.Constraint, active=True))

    # Get all the variables in the model
    my_variables = list(a_model.component_data_objects(pyo.Var, active=True))
    
    # Get the number of variables
    my_n_variables = len(my_variables)

    # Extract the variable names
    my_variable_names = [var.getname() for var in my_variables]
    
    # Initialise the output variables
    my_A_indices = []
    my_A_coeffs  = []
    my_A_names   = []
    my_b         = []

    # Get the standardised constraints related to the constraints
    for _, var in enumerate(my_variables):
        my_A_indices_loc, my_A_coeffs_loc, my_A_names_loc, my_b_loc = get_variable_constraint_standard_form(var,my_variable_names)
        my_A_indices += my_A_indices_loc
        my_A_coeffs  += my_A_coeffs_loc
        my_A_names   += my_A_names_loc
        my_b         += my_b_loc
    
    # Get the standardised constraints related to the constraints
    for _, con in enumerate(my_constraints):
        my_A_indices_loc, my_A_coeffs_loc, my_A_names_loc, my_b_loc = get_constraint_standard_form(con,my_variable_names)
        my_A_indices += my_A_indices_loc
        my_A_coeffs  += my_A_coeffs_loc
        my_A_names   += my_A_names_loc
        my_b         += my_b_loc
        
    # Get the number of constraints
    my_n_constraints = len(my_A_indices)
        
    # Create a matrix A_all and a vector b_all in order to have the constraint A_all*variables <= b_all
    my_A_all = np.zeros((my_n_constraints,my_n_variables))
    my_b_all = np.zeros(my_n_constraints)
    for i, _ in enumerate(my_A_indices):
        my_A_all[i,my_A_indices[i]] = my_A_coeffs[i]
        my_b_all[i] = my_b[i] 
    
    return my_A_indices, my_A_coeffs, my_A_names, my_b, my_A_all, my_b_all


#################################################
# Class related to a polyhedral uncertainty set #
#################################################

class PolyhedralUncertaintySet:
    """
    A class defining a polyhedral uncertainty set (PUS).
    
    Attributes:
    -----------
    data : array
        The data based on which the PUS will be generated, represented as a 2-dimensional numpy array.
    n_directions_pus : int
        The number of values (i.e. columns in the data), used to generate the PUS.
    α : float in (0,0.5)
        Quantiles for the estimate of the lower and upper bounds along each direction.
    cumulated_budget : float
        Budget for the cumulated dispersion.
    pairwise_budget : float
        Budget for the pairwise dispersion.
    n_dir_pairwise_budget : int
        The number of (first) directions concerned by the pairwise dispersion budget.
    """

    def __init__(self,
                 a_data,
                 a_n_directions_pus,
                 a_α,
                 a_cumulated_budget,
                 a_pairwise_budget,
                 a_n_dir_pairwise_budget,
                 a_matrix_projection_on_other_space=None,
                 a_A_to_original_basis = None,
                 a_b_to_original_basis = None,
                 a_description_var_original_basis = None
                 ):

        self.data = a_data
        self.n_directions_pus = a_n_directions_pus
        self.matrix_projection_on_other_space = np.eye(self.n_directions_pus) if ( a_matrix_projection_on_other_space is None ) else a_matrix_projection_on_other_space.copy()
        self.α = a_α 
        self.cumulated_budget = a_cumulated_budget# the budget allowed
        self.pairwise_budget = a_pairwise_budget # the pairwise budget allowed
        self.n_dir_pairwise_budget = a_n_dir_pairwise_budget # the number of directions concerned by the pairwise budget allowed

        self.lower_quantiles = None
        self.upper_quantiles = None

        # The matrices to get the standardized linear program of the form A_w * w + A_z * z <= b with w is the uncertainty realisation in the pus and z = [z_neg, z_pos, lam] are helping variables
        self.A_w = None
        self.A_z = None
        self.b   = None

        # The matrices such that we map with a linear transformation to the data original basis, i.e. u = w' * A_to_original_basis + b_to_original_basis
        self.A_to_original_basis = np.eye(a_n_directions_pus) if (a_A_to_original_basis is None) else a_A_to_original_basis 
        self.b_to_original_basis = np.zeros(a_n_directions_pus) if (a_b_to_original_basis is None) else a_b_to_original_basis 

        # Tells how u must be split in order to have attribute in the original basis  
        self.description_var_original_basis = a_description_var_original_basis

        # Variables to generate distinct points in the PUS, it defines the matrices to impose the distance to previously generated points
        self.points_w_in_pus = None
        self.points_z_in_pus = None

        # Feasibility optimisation problem to solve in order to get a feasible point in the PUS
        self.model_find_points_pus = None

        # Compute the linear constraint
        self.compute_pus_linear_constraints()

    def get_linear_constraints_for_optimisation(self):
        return {'A_w':self.A_w, 'A_z':self.A_z, 'b':self.b, 'A_to_original_basis':self.A_to_original_basis, 'b_to_original_basis':self.b_to_original_basis, 'description_var_original_basis':self.description_var_original_basis}

    def compute_lower_upper_quantiles(self):
        my_lower_quantiles = np.zeros(self.n_directions_pus)
        my_upper_quantiles = np.zeros(self.n_directions_pus)
        for i in range(self.n_directions_pus):
            my_values = self.data[:,i]
            my_lower_quantiles[i] = np.quantile(my_values, self.α)
            my_upper_quantiles[i] = np.quantile(my_values, 1.0-self.α)
        self.lower_quantiles = my_lower_quantiles
        self.upper_quantiles = my_upper_quantiles

    def compute_pus_linear_constraints(self):

        # Compute the quantiles
        self.compute_lower_upper_quantiles()

        # Dimension of the PUS
        n = self.n_directions_pus

        # Create a Pyomo model
        my_model = ConcreteModel()
        
        ########################
        # Variables definition #
        ########################

        my_model.w     = Var(range(n))
        my_model.z_neg = Var(range(n),bounds=(0, 1))
        my_model.z_pos = Var(range(n),bounds=(0, 1))
        my_model.lam   = Var(range(n))

        ##########################
        # Constraints definition #
        ##########################

        # 1) Cumulated budget constraint
        def my_constraint_rule_cumulated_budget(a_model):
            expr = sum(a_model.z_neg[i] for i in range(n)) + sum(a_model.z_pos[i] for i in range(n)) <= self.cumulated_budget
            return expr

        my_model.cumulated_budget = Constraint(rule=my_constraint_rule_cumulated_budget)

        # 2) Pairwise budget constraint
        def my_constraint_rule_pairwise_budget(a_model,i,j):
            expr = a_model.z_neg[i] + a_model.z_pos[i] + a_model.z_neg[j] + a_model.z_pos[j] <= self.pairwise_budget
            return expr

        # Define constraints list for the pairwise budget constraint
        my_model.pairwise_budget = ConstraintList()
        for i in range(min(self.n_dir_pairwise_budget,n)):
            for j in range(min(self.n_dir_pairwise_budget,n)):
                if i < j:
                    my_model.pairwise_budget.add(my_constraint_rule_pairwise_budget(my_model,i,j))

        # # Extract the coefficients of the standardized linear program when only considering the constraints on z which defines the polytope
        # _, _, _, _, my_A_poly, my_b_poly = get_standardized_constraints(my_model)
        # my_A_poly = my_A_poly[:,n:n+2*n]

        # # Get the vertices of the polytope of the feasible set
        # self.vertices = pypoman.compute_polytope_vertices(my_A_poly, my_b_poly)

        # 3) Constraint: lambda = 1/2 * ( z_neg + z_pos + 1)  
        def my_constraint_rule_link_lambda_z(a_model,i):
            expr = 2*a_model.lam[i] == a_model.z_pos[i] - a_model.z_neg[i] + 1.0
            return expr

        # Define constraints list for the constraint on the link between lambda and z_neg, z_pos
        my_model.link_lambda_z = ConstraintList()
        for i in range(n):
            my_model.link_lambda_z.add(my_constraint_rule_link_lambda_z(my_model,i))

        # 4) Constraint: w = (1-lam) * lower_quant + lam * upper_quant  
        def my_constraint_rule_link_w_quantiles(a_model,i):
            expr = a_model.w[i] == (1 - a_model.lam[i]) * self.lower_quantiles[i] + a_model.lam[i] * self.upper_quantiles[i]
            return expr

        # Define constraints list for the constraint on the link between w and the quantiles
        my_model.link_w_quantiles = ConstraintList()
        for i in range(n):
            my_model.link_w_quantiles.add(my_constraint_rule_link_w_quantiles(my_model,i))

        # Extract the coefficients of the standardized linear program
        my_A_indices, my_A_coeffs, my_A_names, my_b, my_A_all, my_b_all = get_standardized_constraints(my_model)

        # Split the matrix to get the standardized linear program of the form A_w * w + A_z * z <= b with z = [z_neg, z_pos, lam]
        my_A_w = my_A_all[:,:n]
        my_Az = my_A_all[:,n:]

        # Add the values to the object
        self.A_w = my_A_w.copy()
        self.A_z = my_Az.copy()
        self.b   = my_b_all.copy()

        # Re-initialise the variables needed to generate distinct points in the PUS, it defines the matrices to impose the distance to previously generated points
        self.points_w_in_pus = []
        self.points_z_in_pus = []

    def linear_constraint(self):
        return

    def check_points_in_pus(self,a_points_w,a_points_z,a_tol=1.0e-5):

        my_A_w = self.A_w
        my_A_z = self.A_z
        my_b   = self.b

        for i, _ in enumerate(a_points_w):
            my_lhs = np.dot(my_A_w, a_points_w[i]) + np.dot(my_A_z, a_points_z[i])
            if np.max(my_lhs-my_b) > a_tol:
                return False, f"The {i+1}th point, i.e. \nw = {a_points_w[i]}\nz = {a_points_z[i]}\nis not in the PUS. The maximum violation distance is {np.max(my_lhs-my_b)} which is above the given tolerance 'a_tol = {a_tol}'. Try increasing the tolerance.", my_lhs, my_b
        return True, f"All points are in the PUS with a tolerance of {a_tol}", None, None

    def initialise_model_to_find_pus_point(self,a_epsilon):
    
        self.points_w_in_pus = []
        self.points_z_in_pus = []
        
        # Create a Pyomo model
        my_model = ConcreteModel()
        
        ########################
        # Variables definition #
        ########################
        
        # Get the number of of constraints and the number of variables w and z
        n_con = self.A_w.shape[0]
        n_w = self.A_w.shape[1]
        n_z = self.A_z.shape[1]
        
        # Initialise the variables
        my_model.w = Var(range(n_w),initialize=0)
        my_model.z = Var(range(n_z),initialize=0)

        ########################
        # Parameter definition #
        ########################

        # Define a parameter epsilon
        my_model.epsilon = Param(initialize=0,mutable=True)
        
        # Fix the value of epsilon to 'a_epsilon'
        my_model.epsilon.value = a_epsilon
        
        ######################
        # Objective function #
        ######################
        
        # Define the original objective function:
        #   We are just tying to find a feasible point here so the objective has no importance, it just helps to solver to look in a given direction
        my_model.obj = Objective(expr=sum(my_model.w[i] for i in range(n_w)), sense=pyo.minimize)
        
        ##########################
        # Constraints definition #
        ##########################
        
        # Define the i-th linear constraint of the matrix linear constraint A_w * w + A_z * z <= b
        def my_ith_linear_constraint(a_model,i):
            expr = sum( (self.A_w[i,j] *a_model.w[j])  for j in range(n_w)) + sum( (self.A_z[i,j] *a_model.z[j])  for j in range(n_z)) <= self.b[i]
            return expr
        
        # Define constraints list for the linear matrix constraint A_w * w + A_z * z <= b 
        my_model.initial_constraints = ConstraintList()
        for i in range(n_con):
            my_model.initial_constraints.add(my_ith_linear_constraint(my_model,i))

        ############################################################################
        # Constraints list to impose a minimum distance to previously found points #
        ############################################################################

        my_model.minimum_distance_constraints = ConstraintList()

        # Define the function to add a point and impose a minimum distance in 1-norm
        def my_constraint_distance(a_model,a_w):
            expr = sum( np.abs(a_model.w[i] - a_w[i])/(self.upper_quantiles[i] - self.lower_quantiles[i]) for i in range(len(a_w)) )  >= len(a_w) * a_model.epsilon
            return expr

        # Set the function in order to add a constraint to impose a minimum distance with previous points found in the PUS
        self.constraint_distance = my_constraint_distance
        
        # Update the model in the object
        self.model_find_points_pus = my_model

    def get_points_in_pus(self,
                          a_n_points,
                          a_recompute=False,
                          a_epsilon=None,
                          a_minimum_epsilon=1.0e-4,
                          a_max_n_trials=10**1,
                          a_tol=1.0e-5):

        # Get the value of the epsilon
        my_epsilon = a_epsilon
        my_epsilon = my_epsilon if (a_epsilon is not None) else 10/a_n_points

        # Check if one does have to recompute the points in the PUS
        if a_recompute | (self.model_find_points_pus is None):
            self.initialise_model_to_find_pus_point(a_epsilon)
    
        # Get the number of distinct points already generated in the PUS
        my_n_points_already_generated = len(self.points_w_in_pus) 

        if my_n_points_already_generated >= a_n_points:
            my_points_to_return = (np.array(self.points_w_in_pus[:a_n_points]), np.array(self.points_z_in_pus[:a_n_points])) 

            # Check that all the points are in the PUS given the tolerance
            my_check_in_pus_given_tol = self.check_points_in_pus(my_points_to_return[0], my_points_to_return[1],a_tol=a_tol)    

            if my_check_in_pus_given_tol[0]:
                return my_points_to_return[0]
            else:
                raise ValueError(my_check_in_pus_given_tol[1])

        # Get the model
        my_model = self.model_find_points_pus

        # Update the value of epsilon (if did not had to re-initialise the model) 
        my_model.epsilon.value = my_epsilon
    
        # Get the dimension of the PUS
        my_n = self.n_directions_pus
        
        bar = progressbar.ProgressBar(maxval=a_n_points-my_n_points_already_generated, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i in range(a_n_points-my_n_points_already_generated):

            # Update the value for the progress bar
            bar.update(i)
    
            # Solver: it needs to be able to handle non linear constraints has we will had constaints | x - x_old | >= epsilon
            my_solver = SolverFactory('ipopt')

            # Details on whether a point has been found
            my_iter = 0
            my_has_found_point_in_pus = False

            while (my_iter < a_max_n_trials) & (not my_has_found_point_in_pus):

                # Get all the variables in the model
                my_variables = list(my_model.component_data_objects(Var, active=True))

                for var in my_variables:
                    var.value = np.random.rand()

                # Solve the optimization problem
                results = my_solver.solve(my_model)
    
                # Get the termination status
                my_termination_condition = results.solver.termination_condition
    
                if my_termination_condition == 'optimal':

                    # Tell that a point in the PUS has been found
                    my_has_found_point_in_pus = True
    
                    # Get the new w in PUS
                    my_w = np.zeros(my_n)
                    for i, _ in enumerate(my_w):
                        my_w[i] = value(my_model.w[i])
    
                    # Get the z solution
                    my_z = np.zeros(self.A_z.shape[1])
                    for i, _ in enumerate(my_z):
                        my_z[i] = value(my_model.z[i])
    
                    # Increment the number of points found in the PUS
                    my_n_points_already_generated += 1
    
                    # Add the new points in the vector with all the points
                    self.points_w_in_pus.append(my_w)
                    self.points_z_in_pus.append(my_z)
    
                    # Update the values force to generate distinct points
                    my_model.minimum_distance_constraints.add(self.constraint_distance(my_model,my_w))

                else:
                    if my_iter < a_max_n_trials:
                        my_iter += 1
                    else:
                        if my_epsilon > a_minimum_epsilon:
                            my_iter = 0
                            my_epsilon = my_epsilon / 2.0
                            print(f"Decrease the size of 'epsilon' to {my_epsilon}")

                            # Update the value of epsilon (if did not had to re-initialise the model) 
                            my_model.epsilon.value = my_epsilon
                        else:
                            raise ValueError(f"A {i}th point in the PUS has been found with final value for epsilon = {my_epsilon}. The termination condition returned is 'my_termination_condition'.")

        bar.finish()

        my_points_to_return = (np.array(self.points_w_in_pus), np.array(self.points_z_in_pus)) 

        # Check that all the points are in the PUS given the tolerance
        my_check_in_pus_given_tol = self.check_points_in_pus(my_points_to_return[0], my_points_to_return[1],a_tol=a_tol)    

        if my_check_in_pus_given_tol[0]:
            return my_points_to_return[0]
        else:
            raise ValueError(my_check_in_pus_given_tol[1])

        return my_points_to_return[0]
