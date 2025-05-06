
"""
R_dynamics.py: file containing the functions to calculate the reaction equation of resources 

CONTAINS: - f: reaction part of the RD equation, calculates uptake and production, given the 
               concentration at each site
          - f_partial: reaction parto of the RD equation, with partial regulation on uptake
          - f_maslov: reaction part of the RD equation, in the case of regulation limited to
                      building blocks, and energy resources not regulated

"""

import numpy as np

#-----------------------------------------------------------------------------------------------------------
# f vectorial function to calculate the sources, given the nxnxn_r concentrations matrix

def f(R, N, param, mat):

    """
    R:     matrix, nxnxn_r, state of resources
    N:     matrix, nxnxn_s, state of population
    param: dictionary, parameters
    mat:   dictionary, matrices

    RETURNS inn-upp: matrix, nxnxn_r, reaction part of RD equation 
            upp:     matrix, nxnxn_r, consumption
            inn:     matrix, nxnxn_r, production
    """
    species = np.argmax(N, axis=2)
    
    # Calculate MM at each site
    upp = R / (R + 1)
    uptake_species = mat['uptake'][species]
    up = upp * uptake_species
    
    # Calculate production
    spec_met_species = mat['spec_met'][species]
    l_w = param['l'] * param['w']
    met_transposed = mat['met'].T
    
    # Compute met_grid 
    met_grid = met_transposed[np.newaxis, np.newaxis, :, :] * spec_met_species[:, :, np.newaxis, :]
    
    # Sum along the met_dim axis
    col_sums = np.sum(met_grid, axis=-2, keepdims=True)
    col_sums[col_sums == 0] = 1e-13  # Avoid division by zero
    
    met_grid_normalized = met_grid / col_sums
    
    # Calculate inn using einsum for efficient summation
    inn = np.einsum('ijk,ijkl->ijl', up * l_w, met_grid_normalized) / param['w']
    
    return inn - up, upp, inn

#-----------------------------------------------------------------------------------------------------------
# f vectorial function to calculate the sources, given the nxnxn_r concentrations matrix

def f_partial(R, N, param, mat):

    """
    R:     matrix, nxnxn_r, state of resources
    N:     matrix, nxnxn_s, state of population
    param: dictionary, parameters
    mat:   dictionary, matrices

    RETURNS inn-upp: matrix, nxnxn_r, reaction part of RD equation 
            upp:     matrix, nxnxn_r, consumption
            inn:     matrix, nxnxn_r, production
    """

    n, _, n_r = R.shape
    n_s = N.shape[2]
    
    species = np.argmax(N, axis=2)
    growth_matrix = np.zeros((n, n))

    # Identify auxotrophies on the grid
    mask = (mat['ess'][species] != 0).astype(int)
    
    # Calculate Michaelis-Menten at each site and mask for essential resources
    upp = R / (R + 1)
    up_ess = np.where(mask == 0, 1, upp)
    
    # Find limiting nutrient and calculate corresponding mu modulation
    lim = np.argmin(up_ess, axis=2)
    mu_lim = np.min(up_ess, axis=2)
    
    # Create modulation mask
    mu = np.ones_like(R) * mu_lim[:, :, np.newaxis]
    mu[np.arange(n)[:, None], np.arange(n), lim] = 1
    
    
    # Modulate uptake and insert uptake rates
    uptake = upp * (param['alpha'] +(1-param['alpha'])*mu) * mat['uptake'][species]
    
    # Calculate production
    spec_met_species = mat['spec_met'][species]
    l_w = param['l'] * param['w']
    met_transposed = mat['met'].T
    
    # Compute met_grid 
    met_grid = met_transposed[np.newaxis, np.newaxis, :, :] * spec_met_species[:, :, np.newaxis, :]
    
    # Sum along the met_dim axis
    col_sums = np.sum(met_grid, axis=-2, keepdims=True)
    col_sums[col_sums == 0] = 1e-13  # Avoid division by zero
    
    met_grid_normalized = met_grid / col_sums
    
    # Calculate inn using einsum for efficient summation
    inn = np.einsum('ijk,ijkl->ijl', uptake * l_w, met_grid_normalized) / param['w']
    
    return inn - uptake, upp, inn

#----------------------------------------------------------------------------------------------------------------------
# f_maslov: R dynamics with eccess nutrients being leaked back into the environment

def f_maslov(R, N, param, mat):

    """
    R:     matrix, nxnxn_r, state of resources
    N:     matrix, nxnxn_s, state of population
    param: dictionary, parameters
    mat:   dictionary, matrices

    RETURNS inn-upp: matrix, nxnxn_r, reaction part of RD equation 
            upp:     matrix, nxnxn_r, consumption
            inn:     matrix, nxnxn_r, production
    """

    n = R.shape[0]

    species = np.argmax(N, axis=2)

    # Identify auxotrophies on the grid
    mask = (mat['ess'][species] != 0).astype(int)
    
    # Calculate Michaelis-Menten at each site and mask for essential resources
    upp = R / (R + 1)
    up_ess = np.where(mask == 0, 1, upp)
    uptake_species = mat['uptake'][species]
    up = upp * uptake_species
    
    # Find limiting nutrient and calculate corresponding mu modulation
    lim = np.argmin(up_ess, axis=2)
    mu_lim = np.min(up_ess, axis=2)
    
    # Create modulation mask
    mu = np.ones_like(R) * mu_lim[:, :, np.newaxis]
    mu[np.arange(n)[:, None], np.arange(n), lim] = 1
        
    # Modulate uptake and insert uptake rates
    leakage = param['l']*mu + (1-mu) 

    spec_met_species = mat['spec_met'][species]
    l_w = leakage * param['w']
    met_transposed = mat['met'].T
    
    # Compute met_grid 
    met_grid = met_transposed[np.newaxis, np.newaxis, :, :] * spec_met_species[:, :, np.newaxis, :]
    
    # Sum along the met_dim axis
    col_sums = np.sum(met_grid, axis=-2, keepdims=True)
    col_sums[col_sums == 0] = 1e-13  # Avoid division by zero
    
    met_grid_normalized = met_grid / col_sums
    
    # Calculate inn using einsum for efficient summation
    inn = np.einsum('ijk,ijkl->ijl', up * l_w, met_grid_normalized) / param['w']

    return inn - up, upp, inn

