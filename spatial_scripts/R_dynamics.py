
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

def f_maslov_nocost(R, N, param, mat):

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
    # Reset leakage for the limiting nutrient to its original parameter
    leakage[np.arange(n)[:, None], np.arange(n), lim] = param['l'][lim]

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

#----------------------------------------------------------------------------------------------------------------------
# f_maslov: R dynamics with eccess nutrients being leaked back into the environment

def f_maslov(R, N, param, mat):
    """
    Spatial version of dR_dt_maslov with essential resource maintenance cost.

    R:     array (n, n, n_r) - resource concentrations on grid
    N:     array (n, n, n_s) - species abundances on grid
    param: dict              - model parameters (incl. Rstar, l, w, tau, ext)
    mat:   dict              - matrices (uptake, ess, met, spec_met)

    Returns:
        inn - upp: resource rate of change (reaction term)
        upp: consumption
        inn: production
    """
    n = R.shape[0]
    n_r = R.shape[2]
    n_s = N.shape[2]

    # Identify dominant species
    species = np.argmax(N, axis=2)  # shape (n, n)

    # Extract species-specific matrices
    uptake = mat['uptake'][species]         # (n, n, n_r)
    ess_mask = mat['ess'][species]          # (n, n, n_r)
    non_ess_mask = 1 - ess_mask
    spec_met = mat['spec_met'][species]     # (n, n, n_r)

    # Michaelis-Menten uptake
    uptake_frac = R / (R + 1)

    # Compute Rlim: min of essential resources at each grid point
    R_ess_masked = np.where(ess_mask == 1, R, np.inf)  # mask non-ess
    Rlim_val = np.min(R_ess_masked, axis=2)            # (n, n)
    lim_mask = Rlim_val > param['Rstar']               # (n, n)

    # Compute essential uptake
    Rlim_uptake = Rlim_val / (1 + Rlim_val)                                # (n, n)
    Rlim_uptake_bcast = Rlim_uptake[:, :, np.newaxis]                      # (n, n, 1)
    R_bb = np.minimum(R, param['Rstar'])                                   # (n, n, n_r)
    up_ess = np.where(
        lim_mask[:, :, np.newaxis],
        ess_mask * uptake * Rlim_uptake_bcast,
        ess_mask * uptake * R_bb / (1 + R_bb)
    )

    # Non-essential uptake remains standard
    up_non_ess = non_ess_mask * uptake * R / (1 + R)

    # Total uptake
    upp = up_ess + up_non_ess  # (n, n, n_r)

    # Compute modulation mu (same as non-cost version)
    mu = np.maximum((Rlim_val - param['Rstar']) / ((Rlim_val - param['Rstar']) + 1), 0)  # (n, n)

    # Compute limiting index per grid point
    Rlim_for_index = np.where(ess_mask == 1, R / (1 + R), np.inf)
    lim = np.argmin(Rlim_for_index, axis=2)  # (n, n)

    # Compute l_eff
    l_eff = param['l'] * mu[:, :, np.newaxis] + (1 - mu[:, :, np.newaxis])  # (n, n, n_r)
    idx = np.arange(n)
    l_eff[idx[:, None], idx, lim] = param['l'][lim].copy()  # restore l for limiting nutrient

    # Final modulated uptake for production
    up_weighted = uptake * uptake_frac * param['w'] * l_eff  # (n, n, n_r)

    # Normalize specific metabolism matrix
    met_grid = mat['met'].T[np.newaxis, np.newaxis, :, :] * spec_met[:, :, np.newaxis, :]  # (n, n, n_r, n_r)
    col_sums = np.sum(met_grid, axis=-2, keepdims=True)
    col_sums[col_sums == 0] = 1e-13
    met_grid_norm = met_grid / col_sums

    # Compute production
    inn = np.einsum('ijk,ijkl->ijl', up_weighted, met_grid_norm) / param['w']  # (n, n, n_r)

    return inn - upp, upp, inn

