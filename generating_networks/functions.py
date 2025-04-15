"""
functions.py

Module containing functions to generate the obligate and facultative cross-feeding networks, across a range of parameters. 

CONTAINS:
    - make_uptake_fac: generates the facultative uptake matrix
    - make_uptake_oblig: generates the obligate uptake matrix
    - make_spec_met: generates the species-specific metabolic matrix
    - make_met_fac: generates the facultative metabolic matrix
    - make_met_oblig: generates the obligate metabolic matrix

"""

import numpy as np

#------------------------------------------------------------------------------------------------------------------------------------------------------------

def make_uptake_fac(n_s,n_r,n_consumed,PCS_sigma,mu,sigma):

    """
    n_s: number of species, int
    n_r: number of resources (primary+secondary), int
    n_consumed: number of consumed resources, int
    PCS_sigma: standard deviation of the uptake rate of the PCS, float
    mu: mean of the uptake rate of the SCS, float
    sigma: standard deviation of the uptake rate of the SCS, float

    RETURNS: the uptake matrix of the facultative network, n_sxn_r matrix

    """

    uptake = np.zeros((n_s,n_r))                      # initialize matrix
    uptake[:,0] = np.random.normal(1,PCS_sigma,n_s)   # uptake of the primary resource

    for species in range(n_s):
        consumed_resources = np.random.choice(range(1,n_r),n_consumed,replace=False)  # choose n_consumed resources
        uptake[species,consumed_resources] = np.random.normal(mu,sigma,n_consumed)    # assign uptake rates to secondary resources      
    
    return uptake

def make_uptake_oblig(uptake_fac):

    """
    uptake_fac: uptake matrix of the facultative network, n_sxn_r matrix

    RETURNS: the uptake matrix of the obligate network, n_sxn_r matrix

    """

    uptake = uptake_fac.copy()                        # copy the facultative uptake matrix
    uptake[:, 1:][uptake_fac[:, 1:] != 0] = 1   

    return uptake

#------------------------------------------------------------------------------------------------------------------------------------------------------------

def make_spec_met(uptake_oblig, n_producers):

    """
    uptake_oblig: uptake matrix of the obligate network, n_sxn_r matrix
    n_producers: number of producers per resource, int
    
    RETURNS: the species-specific metabolic matrix, n_sxn_r matrix

    """

    spec_met = np.zeros(uptake_oblig.shape)                              # initialize matrix

    for col in range(1, uptake_oblig.shape[1]):
        
        non_zero_indices = np.where(uptake_oblig[:, col] != 1)[0]  # Get the indices where the potential_spec_met is non-zero
        
        if len(non_zero_indices) >= n_producers:                         # Randomly select `n_producers` indices from the non-zero positions
            selected_indices = np.random.choice(non_zero_indices, n_producers, replace=False)
        else:
            selected_indices = non_zero_indices
        
        # Set the selected indices to 1 in the potential_spec_met matrix
        spec_met[selected_indices, col] = 1

    return spec_met

#------------------------------------------------------------------------------------------------------------------------------------------------------------

def fill_dirichlet(mat,cost_spread):

    """
    mat: matrix to be filled with Dirichlet distribution
    cost_spread: the parameters for the Dirichlet distribution, determins how different costs of production of different things are, float
    
    RETURNS: the matrix filled with Dirichlet distribution

    """

    for column in range(mat.shape[1]):
        # Find indices where matrix[:, col_idx] == 1 (non-zero entries)
        non_zero_indices = np.where(mat[:, column] == 1)[0]  
        if len(non_zero_indices) > 0:
            # Sample from Dirichlet distribution for non-zero entries
            dirichlet_values = np.random.dirichlet(np.ones(len(non_zero_indices))*cost_spread)
            mat[non_zero_indices, column] = dirichlet_values
    return mat


def make_met_fac(n_r,cost_spread):

    """
    n_r: number of resources (primary+secondary), int
    cost_spread: the parameters for the Dirichlet distribution, determins how different costs of production of different things are, float
    
    RETURNS: the facultative metabolic matrix, n_rxn_r matrix

    """

    met_fac = np.ones((n_r,n_r))
    met_fac[0,:] = 0                           # set the first row to zero (no PCS production)
    np.fill_diagonal(met_fac, 0)               # Set the diagonal to 0

    return fill_dirichlet(met_fac, cost_spread)

def make_met_oblig(met_fac):

    """
    met_fac: metabolic matrix of the facultative network, n_rxn_r matrix
    
    RETURNS: the obligate metabolic matrix, n_rxn_r matrix

    """

    met_oblig = np.zeros(met_fac.shape)                     # initialize matrix

    np.fill_diagonal(met_oblig, 1)                          # Set the diagonal to 1 (to leak back)
    met_oblig[:,0] = met_fac[:,0].copy()                    # copy PCS from facultative matrix

    return met_oblig

#------------------------------------------------------------------------------------------------------------------------------------------------------------






