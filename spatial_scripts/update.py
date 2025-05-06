"""
update.py: file containing the functions to run the simulations with given conditions

CONTAINS: - simulate_3D: function to run a simulation using the SOR_3D algorithm for PBC
          - simulate_3D_NBC: function to run a simulation using the SOR_3D algorithm for no flux BC on top
          - simulate_MG: function to perform multi-grid iteration
          - simulate_3D_nobiom: no biomass model
          - simulate 2D_nobiom: no biomass model in 2D
          - shannon:     function to calculate shannon diversity in real time to check for convergence
          - abundances:  function to compute relative abundance of a frame
          - change_grid_R/N: functions for the multigrid solver, to change grids 

"""

import numpy as np
import numba

from time import time
from scipy.interpolate import RegularGridInterpolator
from numba import jit

from SOR import *
from N_dynamics import *
from R_dynamics import *

# t_max for real time chck in order to avoid timeout of execution, remember to change this based on job requirements
t_max=600 # minutes

#---------------------------------------------------------------------------------------------
# simulate_3D: functiln to run a simulation with PBC, in a quasi-3D setting

def simulate_3D(steps, source, growth_model, initial_guess, initial_N, initial_b, param, mat):

    """
    steps:         int, number of steps we want to run the simulation for
    source:        function, (*args: R,N,param,mat; out: nxn source matrix), reaction in RD equation
    growth_model:  function, (*args: R, N, param, mat; out: nxn growth matrix, nxn modulation matrix)
    initial_guess: matrix, nxnxn_r, initial guess for eq. concentrations
    initial_N:     matrix, nxnxn_s, initial composition of species grid
    initial_b:     matrix, nxn, initial biomass (floats from 0 to 2)
    param:         dictionary, parameters
    mat:           dictionary, matrices

    RETURNS last_2_frames_N, mod, current_R, current_N, g_rates, s_list, abundances, t_list, new_biomass 

    """

    # start timing simulation
    t0 = time()

    # extract list of all possible species
    n_s = len(param['g'])
    all_species = list(range(n_s))

    # lists to store time steps 
    last_2_frames_N  = [decode(initial_N)] 
    abundances = [calc_abundances(initial_N)]
    s_list = [shannon(initial_N)]
    t_list = [0]

    # first iteration
    print('Solving iteration zero, finding equilibrium from initial guess')

    # computing equilibrium concentration at ztep zero
    current_R, _, _ = SOR_3D(initial_N, param, mat, source, initial_guess)
    # computing growth rates on all the grid
    g_rates, mod  = growth_model(current_R,initial_N,param,mat)
    # performing BD dynamics
    check, most_present, decoded_N, new_biomass, t_div = birth_death_biomass(decode(initial_N), g_rates, initial_b)
    current_N = encode(decoded_N, all_species)

    # store time step
    last_2_frames_N.append(decoded_N)
    abundances.append(calc_abundances(current_N))
    t_list.append(t_div)

    convergence_count = 0

    for i in range(steps):

        print("Step %i" % (i+1))

        # compute new equilibrium, initial guess is previous equilibrium
        current_R, _, _ = SOR_3D(current_N, param, mat, source, current_R)

        # compute growth rates
        g_rates, mod  = growth_model(current_R,current_N,param,mat)
        # performe DB dynamics
        check, most_present, decoded_N, new_biomass, t_div = birth_death_biomass(decode(current_N), g_rates, new_biomass)
        # check that there is more than one species
        if check == 'vittoria':
            print('winner species is: ', most_present)
            break

        current_N = encode(decoded_N, all_species)

        # save time step
        last_2_frames_N  = [last_2_frames_N[1], decoded_N]
        abundances.append(calc_abundances(current_N))
        t_list.append(t_div)

        # check if shannon diversity has converged
        s = shannon(current_N)
        s_list.append(s)

        if len(s_list)>1000:
            recent_abundances = np.array(abundances[-300:])  # average of last 300 time steps
            avg = np.mean(recent_abundances, axis=0) 
            dev = np.std(recent_abundances, axis=0)

            converged = np.all(np.abs(abundances[-1] - avg) < dev)

            if converged:
                convergence_count += 1
            else:
                convergence_count = 0
            if convergence_count > 500:
                print('Abundances have converged for all species')
                break

        t1 = time()
        if round((t1-t0)/60,4)>590:
            break

    # end timing
    t1 = time()
    print(f'\n Time taken to solve for {steps} steps: ', round((t1-t0)/60,4), ' minutes \n')

    return last_2_frames_N, mod, current_R, current_N, g_rates, s_list, abundances, t_list, new_biomass 


#---------------------------------------------------------------------------------------------
# simulate_3D_NBC: functiln to run a simulation with PBC, in a quasi-3D setting, with no flux on top

def simulate_3D_NBC(steps, source, growth_model, initial_guess, initial_N, initial_b, param, mat):

    """
    steps:         int, number of steps we want to run the simulation for
    source:        function, (*args: R,N,param,mat; out: nxn source matrix), reaction in RD equation
    growth_model:  function, (*args: R, N, param, mat; out: nxn growth matrix, nxn modulation matrix)
    initial_guess: matrix, nxnxn_r, initial guess for eq. concentrations
    initial_N:     matrix, nxnxn_s, initial composition of species grid
    initial_b:     matrix, nxn, initial biomass (floats from 0 to 2)
    param:         dictionary, parameters
    mat:           dictionary, matrices

    RETURNS last_2_frames_N, mod, current_R, current_N, g_rates, s_list, abundances, t_list, new_biomass 

    """

    # start timing simulation
    t0 = time()

    # extract list of all possible species
    n_s = len(param['g'])
    all_species = list(range(n_s))

    # lists to store time steps 
    last_2_frames_N  = [decode(initial_N)] 
    abundances = [calc_abundances(initial_N)]
    s_list = [shannon(initial_N)]
    t_list = [0]

    # first iteration
    print('Solving iteration zero, finding equilibrium from initial guess')

    # computing equilibrium concentration at ztep zero
    current_R, _, _ = SOR_3D_noflux(initial_N, param, mat, source, initial_guess)
    # computing growth rates on all the grid
    g_rates, mod  = growth_model(current_R,initial_N,param,mat)
    # performing BD dynamics
    check, most_present, decoded_N, new_biomass, t_div = birth_death_biomass(decode(initial_N), g_rates, initial_b)
    current_N = encode(decoded_N, all_species)

    # store time step
    last_2_frames_N.append(decoded_N)
    abundances.append(calc_abundances(current_N))
    t_list.append(t_div)

    convergence_count = 0

    for i in range(steps):

        print("Step %i" % (i+1))

        # compute new equilibrium, initial guess is previous equilibrium
        current_R, _, _ = SOR_3D_noflux(current_N, param, mat, source, current_R)

        # compute growth rates
        g_rates, mod  = growth_model(current_R,current_N,param,mat)
        # performe DB dynamics
        check, most_present, decoded_N, new_biomass, t_div = birth_death_biomass(decode(current_N), g_rates, new_biomass)
        # check that there is more than one species
        if check == 'vittoria':
            print('winner species is: ', most_present)
            break

        current_N = encode(decoded_N, all_species)

        # save time step
        last_2_frames_N  = [last_2_frames_N[1], decoded_N]
        abundances.append(calc_abundances(current_N))
        t_list.append(t_div)

        # check if shannon diversity has converged
        s = shannon(current_N)
        s_list.append(s)

        if len(s_list)>1000:
            recent_abundances = np.array(abundances[-300:])  # average of last 300 time steps
            avg = np.mean(recent_abundances, axis=0) 
            dev = np.std(recent_abundances, axis=0)

            converged = np.all(np.abs(abundances[-1] - avg) < dev)

            if converged:
                convergence_count += 1
            else:
                convergence_count = 0
            if convergence_count > 500:
                print('Abundances have converged for all species')
                break

        t1 = time()
        if round((t1-t0)/60,4)>1100:
            break

    # end timing
    t1 = time()
    print(f'\n Time taken to solve for {steps} steps: ', round((t1-t0)/60,4), ' minutes \n')

    return last_2_frames_N, mod, current_R, current_N, g_rates, s_list, abundances, t_list, new_biomass 


#-------------------------------------------------------------------------------------------------
# function to perform simulations in multigrid iterations

def simulate_MG(steps, source, growth_model, initial_guess, initial_N, initial_b, param, mat, t):

    """
    steps:         int, number of steps we want to run the simulation for
    source:        function, (*args: R,N,param,mat; out: nxn source matrix), reaction in RD equation
    growth_model:  function, (*args: R, N, param, mat; out: nxn growth matrix, nxn modulation matrix)
    initial_guess: matrix, nxnxn_r, initial guess for eq. concentrations
    initial_N:     matrix, nxnxn_s, initial composition of species grid
    initial_b:     matrix, nxn, initial biomass (floats from 0 to 2)
    param:         dictionary, parameters
    mat:           dictionary, matrices
    t:             int, number of times to refine

    RETURNS last_2_frames_N, mod, current_R, current_N, g_rates, s_list, abundances, t_list, new_biomass 

    """

    # start timing simulation
    t0 = time()

    # extract list of all possible species
    all_species = list(range(len(param['g'])))

    # lists to store time steps 
    last_2_frames_N  = [decode(initial_N)] 
    abundances = [calc_abundances(initial_N)]
    s_list = [shannon(initial_N)]
    t_list = [0]

    # initial grid size
    n = initial_N.shape[0]

    # first iteration
    print('Solving iteration zero, finding equilibrium from initial guess')

    # computing equilibrium concentration at ztep zero
    current_R, _, _ = SOR_3D(initial_N, param, mat, source, initial_guess)

    # first refinement
    finer_R, finer_up, finer_prod = SOR_3D(change_grid_N(initial_N,n*2), param, mat, source, change_grid_R(current_R,n*2))
    # subsequent refinements
    for ref in range(3,t+2):
        m = n*ref
        finer_R, finer_up, finer_prod = SOR_3D(change_grid_N(initial_N,m), param, mat, source, change_grid_R(finer_R,m))
    coarser_R = change_grid_R(finer_R,n)

    # computing growth rates on all the grid
    g_rates, mod  = growth_model(current_R,initial_N,param,mat)
    # performing BD dynamics
    check, most_present, decoded_N, new_biomass, t_div = birth_death_biomass(decode(initial_N), g_rates,  initial_b)
    current_N = encode(decoded_N, all_species)

    # store time step
    last_2_frames_N.append(decoded_N)
    abundances.append(calc_abundances(current_N))
    t_list.append(t_div)

    convergence_count = 0

    for i in range(steps):

        print("Step %i" % (i+1))

        # compute new equilibrium, initial guess is previous equilibrium
        finer_R, _, _ = SOR_3D(change_grid_N(current_N,(t+1)*n), param, mat, source, finer_R)
        coarser_R = change_grid_R(finer_R,n)

        # compute growth rates
        g_rates, mod  = growth_model(current_R,current_N,param,mat)
        # performe DB dynamics
        check, most_present, decoded_N, new_biomass, t_div = birth_death_biomass(decode(current_N), g_rates, new_biomass)
        # check that there is more than one species
        if check == 'vittoria':
            print('winner species is: ', most_present)
            break
        current_N = encode(decoded_N, all_species)

        # save time step
        last_2_frames_N  = [last_2_frames_N[1], decoded_N]
        abundances.append(calc_abundances(current_N))

        # check if shannon diversity has converged
        s = shannon(current_N)
        s_list.append(s)

        if len(s_list)>1000:
            recent_abundances = np.array(abundances[-300:])  # average of last 300 time steps
            avg = np.mean(recent_abundances, axis=0) 
            dev = np.std(recent_abundances, axis=0)

            converged = np.all(np.abs(abundances[-1] - avg) < dev)

            if converged:
                convergence_count += 1
            else:
                convergence_count = 0
            if convergence_count > 500:
                print('Abundances have converged for all species')
                break

        t1 = time()
        if round((t1-t0)/60,4)>590:
            break

    # end timing
    t1 = time()
    print(f'\n Time taken to solve for {steps} steps: ', round((t1-t0)/60,4), ' minutes \n')

    return last_2_frames_N, mod, current_R, current_N, g_rates, s_list, abundances, t_list, new_biomass 


#-------------------------------------------------------------------------------------------------
# simulate_3D: functiln to run a simulation with PBC, in a quasi-3D setting

def simulate_3D_nobiom(steps, source, growth_model, initial_guess, initial_N, param, mat):

    """
    steps:         int, number of steps we want to run the simulation for
    source:        function, (*args: R,N,param,mat; out: nxn source matrix), reaction in RD equation
    growth_model:  function, (*args: R, N, param, mat; out: nxn growth matrix, nxn modulation matrix)
    initial_guess: matrix, nxnxn_r, initial guess for eq. concentrations
    initial_N:     matrix, nxnxn_s, initial composition of species grid
    param:         dictionary, parameters
    mat:           dictionary, matrices

    RETURNS frames_N:  list, all the frames of the population grid, in a decoded species matrix form
            frames_R:  list, all the frames of the chemicals grid
            frames_up: list, all the frames of the chemicals grid for uptake
            frames_in: list, all the frames of the chemicals grid for the production
            frames_mu: list, all the frames of the liminting modulation of the growth matrix 
            current_R: matrix, nxnxn_r, final configuration of nutrients grid
            current_N: matrix, nxnxn_s, final configuration of population grid
            g_rates:   matrix, nxn, final growth rates
            s_list:    list, time series of shannon diversity

    """

    # start timing simulation
    t0 = time()

    # extract list of all possible species
    n_s = len(param['g'])
    all_species = list(range(n_s))

    # lists to store time steps 
    last_2_frames_N  = [decode(initial_N)] 
    abundances = [calc_abundances(initial_N)]
    s_list = [shannon(initial_N)]

    # first iteration
    print('Solving iteration zero, finding equilibrium from initial guess')

    # computing equilibrium concentration at ztep zero
    current_R, _, _ = SOR_3D(initial_N, param, mat, source, initial_guess)
    # computing growth rates on all the grid
    g_rates, mod  = growth_model(current_R,initial_N,param,mat)
    # performing DB dynamics
    decoded_N, check, most_present = death_birth_periodic(decode(initial_N),g_rates)
    current_N = encode(decoded_N, all_species)

    # store time step
    last_2_frames_N.append(decoded_N)
    abundances.append(calc_abundances(current_N))

    convergence_count = 0

    for i in range(steps):

        print("Step %i" % (i+1))

        # compute new equilibrium, initial guess is previous equilibrium
        current_R, _, _ = SOR_3D(current_N, param, mat, source, current_R)

        # compute growth rates
        g_rates, mod  = growth_model(current_R,current_N,param,mat)
        # performe DB dynamics
        decoded_N,check,most_present = death_birth_periodic(decode(current_N),g_rates)
        # check that there is more than one species
        if check == 'vittoria':
            print('winner species is: ', most_present)
            break

        current_N = encode(decoded_N, all_species)

        # save time step
        last_2_frames_N  = [last_2_frames_N[1], decoded_N]
        abundances.append(calc_abundances(current_N))

        # check if shannon diversity has converged
        s = shannon(current_N)
        s_list.append(s)

        if len(s_list)>1000:
            recent_abundances = np.array(abundances[-300:])  # average of last 300 time steps
            avg = np.mean(recent_abundances, axis=0) 
            dev = np.std(recent_abundances, axis=0)

            converged = np.all(np.abs(abundances[-1] - avg) < dev)

            if converged:
                convergence_count += 1
            else:
                convergence_count = 0
            if convergence_count > 500:
                print('Abundances have converged for all species')
                break

        t1 = time()
        if round((t1-t0)/60,4)>1670:
            break

    # end timing
    t1 = time()
    print(f'\n Time taken to solve for {steps} steps: ', round((t1-t0)/60,4), ' minutes \n')

    return last_2_frames_N, mod, current_R, current_N, g_rates, s_list, abundances   


#----------------------------------------------------------------------------------------------------
# simulate_2D: functiln to run a simulation with DBC, in a 2D setting

def simulate_2D_nobiom(steps, source, growth_model, initial_guess, initial_N, param, mat):

    """
    steps:         int, number of steps we want to run the simulation for
    source:        function, (*args: R,N,param,mat; out: nxn source matrix), reaction in RD equation
    growth_model:  function, (*args: R, N, param, mat; out: nxn growth matrix, nxn modulation matrix)
    initial_guess: matrix, nxnxn_r, initial guess for eq. concentrations
    initial_N:     matrix, nxnxn_s, initial composition of species grid
    param:         dictionary, parameters
    mat:           dictionary, matrices

    RETURNS frames_N:  list, all the frames of the population grid, in a decoded species matrix form
            frames_R:  list, all the frames of the chemicals grid
            frames_up: list, all the frames of the chemicals grid for uptake
            frames_in: list, all the frames of the chemicals grid for the production
            frames_mu: list, all the frames of the liminting modulation of the growth matrix 
            current_R: matrix, nxnxn_r, final configuration of nutrients grid
            current_N: matrix, nxnxn_s, final configuration of population grid
            g_rates:   matrix, nxn, final growth rates
            s_list:    list, time series of shannon diversity

    """

    # start timing simulation
    t0 = time()

    # extract list of all possible species
    all_species = list(range(len(param['g'])))

    # lists to store time steps 
    last_2_frames_N  = [decode(initial_N)] 
    abundances = [calc_abundances(initial_N)]
    s_list = [shannon(initial_N)]

    # first iteration
    print('Solving iteration zero, finding equilibrium from initial guess')

    # computing equilibrium concentration at ztep zero
    current_R, up, prod = SOR_2D(initial_N, param, mat, source, initial_guess)
    # computing growth rates on all the grid
    g_rates, mod  = growth_model(current_R,initial_N,param,mat)
    # performing DB dynamics
    decoded_N, check, most_present = death_birth_periodic(decode(initial_N),g_rates)
    current_N = encode(decoded_N, all_species)

    # store time step
    last_2_frames_N.append(decoded_N)
    abundances.append(calc_abundances(current_N))

    convergence_count = 0

    for i in range(steps):

        print("Step %i" % (i+1))

        # compute new equilibrium, initial guess is previous equilibrium
        current_R, up, prod = SOR_2D(current_N, param, mat, source, current_R)

        # compute growth rates
        g_rates, mod  = growth_model(current_R,current_N,param,mat)
        # performe DB dynamics
        decoded_N,check,most_present = death_birth_periodic(decode(current_N),g_rates)
        # check that there is more than one species
        if check == 'vittoria':
            print('winner species is: ', most_present)
            break

        current_N = encode(decoded_N, all_species)

        # save time step
        last_2_frames_N  = [last_2_frames_N[1], decoded_N]
        abundances.append(calc_abundances(current_N))

        # check if shannon diversity has converged
        s = shannon(current_N)
        s_list.append(s)

        if len(s_list)>1000:
            recent_abundances = np.array(abundances[-300:])  # average of last 300 time steps
            avg = np.mean(recent_abundances, axis=0) 
            dev = np.std(recent_abundances, axis=0)

            converged = np.all(np.abs(abundances[-1] - avg) < dev)

            if converged:
                convergence_count += 1
            else:
                convergence_count = 0
            if convergence_count > 100:
                print('Abundances have converged for all species')
                break
        
        t1 = time()
        if round((t1-t0)/60,4)>15:
            break

    # end timing
    t1 = time()
    print(f'\n Time taken to solve for {steps} steps: ', round((t1-t0)/60,4), ' minutes \n')

    return last_2_frames_N, mod, current_R, current_N, g_rates, s_list, abundances

#-------------------------------------------------------------------------------------------------
# function to quickly compute relative abundances from grid

def calc_abundances(frame):

    """
    frame:   matrix, nxnxn_s, species frame

    RETURNS ab: vector, n_s, contains relative abundances of each species

    """

    total_cells = frame.shape[0] * frame.shape[1]  
    species_counts = np.sum(frame, axis=(0, 1)) 
    ab = species_counts / total_cells

    return ab

#-------------------------------------------------------------------------------------------------
# function to calculate Shannon diversity

def shannon(N):

    """
    N: matrix, nxnxn_s, encoded population grid

    RETURNS shannon_entropy, float, value for shannon diversity

    """

    n_s = N.shape[-1]
    n   = N.shape[0]

    # frequencies
    counts = np.zeros((n_s))
    for i in range(n_s):
        counts[i] = np.sum(N[:,:,i])
    freq = counts/(n*n)

    # calculate shannon diversity
    shannon_entropy = -np.sum(freq * np.log(freq + 1e-14))  # avoid log(0)

    return shannon_entropy

#----------------------------------------------------------------------------------------------------
# functions to change grid for multigrid solvers

def change_grid_R(R, m):

    """
    R: matrix, nxnxn_r, resource matrix
    m: int, new number of grid points

    RETURNS resized_R: matrix, mxmxn_r, resized resources matrix

    """

    n, _, n_r = R.shape

    old_x, old_y = np.arange(n), np.arange(n)
    new_x, new_y = np.linspace(0, n-1, m), np.linspace(0, n-1, m)
    
    resized_R = np.zeros((m, m, n_r))

    if m > n:
        # Interpolation case: New grid is finer
        for i in range(n_r):
            interpolator = RegularGridInterpolator((old_x, old_y), R[:, :, i])
            new_grid_points = np.array(np.meshgrid(new_x, new_y, indexing='ij')).T.reshape(-1, 2)
            resized_R[:, :, i] = interpolator(new_grid_points).reshape(m, m)
    else:
        # Averaging case: New grid is coarser
        scale_factor = n / m
        for i in range(m):
            for j in range(m):
                x_start = int(i * scale_factor)
                x_end = int((i + 1) * scale_factor)
                y_start = int(j * scale_factor)
                y_end = int((j + 1) * scale_factor)
                
                for k in range(n_r):
                    resized_R[i, j, k] = np.mean(R[x_start:x_end, y_start:y_end, k])

    return resized_R

def change_grid_N(N, m):

    """
    N: matrix, nxnxn_s, old species grid
    m: int, new size for grid

    RETURNS new_N: matrix, mxmxn_s, reshaped species grid
    
    """

    n, _, n_s = N.shape
    new_N = np.zeros((m, m, n_s))

    if m == n:
        return N  # No changes needed

    if m > n:
        # If the new grid is larger, copy the values from the old grid
        for i in range(m):
            for j in range(m):
                new_N[i, j] = N[i * n // m, j * n // m]
    else:
        # If the new grid is smaller, use the central point value from the finer grid
        scale_factor = n / m
        half_scale = scale_factor / 2
        for i in range(m):
            for j in range(m):
                center_i = int(i * scale_factor + half_scale)
                center_j = int(j * scale_factor + half_scale)
                new_N[i, j] = N[center_i, center_j]
    
    return new_N

