"""
SOR.py: file to store the algorithm to solve for equilibrium concentrations of the chemicals
        on a discretized-space grid. It contains versions with different details for a successive
        overrelaxation algorithm: an iterative method to solve the non-linear Poisson problem
        one is left with after assuming that chemicals reach equilibrium in space much faster than
        typical time scales for cell reproduction. See report for more information.

CONTAINS: - SOR_2D: a Dirichlet BC setting where a frame of the grid has fixed chemicals concentrations
          - SOR_3D: a PBC setting; to make implementation possible, 3 layers are staked on the z axis,
                    with central layer corresponding to the chemicals layer in SOR_2D and has toroidal
                    boundary conditions; bottom layer has fixed constant concentrations and top layer 
                    has fixed zero concentrations
          - SOR_3D_noflux: like SOR_3D, but the top boundary condition is Newmann

"""

import numpy as np

import sys


from time import time

#------------------------------------------------------------------------------------------------------
# SOR_2D algorithm to iteratively solve a non linear Poisson problem of the form ∇^2*R = f(R) with PBC

def SOR_2D(N, param, mat, source, initial_guess):
    """
    N:             matrix, nxnxn_s, population grid
    param:         dictionary, parameters
    mat:           dictionary, matrices
    source:        function, (*args: R,N,param,mat; out: nxn source matrix), reaction in RD equation
    initial_guess: matrix, nxnxn_r, initial guess for eq. concentrations

    RETURNS R_eq: matrix, nxnxn_r, equilibrium concentration
            up  : matrix, nxnxn_r, equilibrium uptake
            prod: matrix, nxnxn_r, equilibrium production

    """

    # parameters needed for SOR implementation
    L     = param['L']                                    # float, length of the side 
    D     = param['D']                                    # float, diffusion constant
    stop  = param['acc']                                  # float, convergence criterion
    bound = param['ext']                                  # vector, n_r, bottom D. BC

    n     = N.shape[0]                                    # int, number of grid points (currently) 
    h     = L/n                                           # float, finite difference step
    n_r   = len(bound)                                    # int, number of resources

    # updates initialization
    delta_list = [1]                                      # list, store the biggest update at each step
    delta = np.zeros((n,n,n_r))                           # matrix, nxnxn_r, store updates                      

    # create frame for fixed BC
    BC = np.tile(bound, (n+2, n+2, 1))
    BC[1:n+1,1:n+1,:] = initial_guess
    best_BC = BC

    # start timing
    t0 = time()

    # SOR algorithm, convergence condition based on update relative to current absolute value
    while ((delta_list[-1]>best_BC[1:n+1,1:n+1,:]*stop).any()):

        current_source, up , prod = source(best_BC[1:n+1,1:n+1,:],N,param,mat)

        # prepare grid, give red and black colors        
        i, j = np.mgrid[1:n+1, 1:n+1]
        checkerboard  = (i + j) % 2
        i_red, j_red = i[checkerboard == 0], j[checkerboard == 0]
        i_black, j_black = i[checkerboard == 1], j[checkerboard == 1]

        # red dots update
        delta[i_red-1,j_red-1,:] = 0.25*(best_BC[i_red,j_red+1,:]+best_BC[i_red,j_red-1,:]+best_BC[i_red+1,j_red,:]+best_BC[i_red-1,j_red,:]+(h**2/D)*current_source[i_red-1,j_red-1,:])-best_BC[i_red,j_red,:]
        best_BC[i_red,j_red] += param['sor']*delta[i_red-1,j_red-1]

        # black dots update
        delta[i_black-1,j_black-1,:] = 0.25*(best_BC[i_black,j_black+1,:]+best_BC[i_black,j_black-1,:]+best_BC[i_black+1,j_black,:]+best_BC[i_black-1,j_black,:]+(h**2/D)*current_source[i_black-1,j_black-1,:])-best_BC[i_black,j_black,:] 
        best_BC[i_black,j_black] += param['sor']*delta[i_black-1,j_black-1]

        if (np.abs(delta[1:-1,1:-1]/(best_BC[2:n,2:n]+1e-12))<=0.001).all():
                break

        if (padded_R<0).any():
             print('numerical instabilities')
             sys.exit()

        # extract biggest update
        delta_list.append(np.max(np.abs(delta)))

        print("N_iter %d delta_max %e\r" % (len(delta_list)-1, delta_list[-1]), end='')

        # check for very small deltas
        if (np.abs(delta_list[-1])<2e-10):
            break

    R_eq = best_BC[1:n+1,1:n+1,:]

    # end timing and print time
    t1 = time()
    print('\n Time taken to solve for equilibrium: ', round((t1-t0)/60,4), ' minutes')


    return R_eq, up, prod 

#------------------------------------------------------------------------------------------------------
# SOR_3D algorithm to iteratively solve a non linear Poisson problem of the form ∇^2*R = f(R) with DBC

def SOR_3D(N, param, mat, source, initial_guess):
    """
    N:             matrix, nxnxn_s, population grid
    param:         dictionary, parameters
    mat:           dictionary, matrices
    source:        function, (*args: R,N,param,mat; out: nxn source matrix), reaction in RD equation
    initial_guess: matrix, nxnxn_r, initial guess for eq. concentrations

    RETURNS R_eq: matrix, nxnxn_r, equilibrium concentration
            up  : matrix, nxnxn_r, equilibrium uptake
            prod: matrix, nxnxn_r, equilibrium production

    """

    # parameters needed for SOR implementation
    L     = param['L']                                    # float, length of the side 
    D     = param['D']                                    # float, diffusion constant
    rapp  = param['Dz']/D                                 # float, ratio between Dz and Dxy
    stop  = param['acc']                                  # float, convergence criterion
    bound = param['ext']                                  # vector, n_r, bottom D. BC

    n     = N.shape[0]                                    # int, number of grid points (currently) 
    h     = L/n                                           # float, finite difference step
    n_r   = len(bound)                                    # int, number of resources

    # updates initialization
    delta_list = [1]                                      # list, store the biggest update at each step
    delta = np.zeros((n,n,n_r))                           # matrix, nxnxn_r, store updates                      

    # create frame for quasi 3D BC
    padded_R = np.zeros((n+2,n+2,3,n_r))                  # fill top layer with D. zero BC
    padded_R[:,:,0,:] = np.tile(bound, (n+2, n+2, 1))     # fill bottom layer with D. BC
    padded_R[1:n+1,1:n+1,1,:] = initial_guess             # fill middle layer with initial guess

    # start timing
    t0 = time()

    # SOR algorithm, convergence condition based on update relative to current absolute value
    while (True):

        # periodic BC in middle layer
        padded_R[0,1:n+1,1,:]=padded_R[-2,1:n+1,1,:]
        padded_R[-1,1:n+1,1,:]=padded_R[1,1:n+1,1,:]
        padded_R[1:n+1,0,1,:]=padded_R[1:n+1,-2,1,:]
        padded_R[1:n+1,-1,1,:]=padded_R[1:n+1,1,1,:]

        # computing source
        padded_source = np.zeros((n+2,n+2,3,n_r))
        s, up, prod = source(padded_R[1:n+1,1:n+1,1,:],N,param,mat)
        padded_source[1:n+1,1:n+1,1] = s
          
        # prepare grid, assign red and black colors        
        i, j = np.mgrid[1:n+1, 1:n+1]
        checkerboard  = (i + j) % 2
        i_red, j_red = i[checkerboard == 0], j[checkerboard == 0]
        i_black, j_black = i[checkerboard == 1], j[checkerboard == 1]

        # red dots update
        delta[i_red-1,j_red-1,:] = 1/(4+rapp*2)*(padded_R[i_red,j_red+1,1,:]+padded_R[i_red,j_red-1,1,:]+padded_R[i_red+1,j_red,1,:]+padded_R[i_red-1,j_red,1,:]
                                   +rapp*(padded_R[i_red,j_red,0,:]+padded_R[i_red,j_red,2,:])
                                   +(h**3/D)*padded_source[i_red,j_red,1,:])-padded_R[i_red,j_red,1,:]
        padded_R[i_red,j_red,1,:] += param['sor']*delta[i_red-1,j_red-1]

        # repeat PBC
        # periodic BC in middle layer
        padded_R[0,1:n+1,1,:]=padded_R[-2,1:n+1,1,:]
        padded_R[-1,1:n+1,1,:]=padded_R[1,1:n+1,1,:]
        padded_R[1:n+1,0,1,:]=padded_R[1:n+1,-2,1,:]
        padded_R[1:n+1,-1,1,:]=padded_R[1:n+1,1,1,:]

        # black dots update
        delta[i_black-1,j_black-1,:] = 1/(4+rapp*2)*(padded_R[i_black,j_black+1,1,:]+padded_R[i_black,j_black-1,1,:]+padded_R[i_black+1,j_black,1,:]+padded_R[i_black-1,j_black,1,:]
                                   +rapp*(padded_R[i_black,j_black,0,:]+padded_R[i_black,j_black,2,:])
                                   +(h**3/D)*padded_source[i_black,j_black,1,:])-padded_R[i_black,j_black,1,:]
        padded_R[i_black,j_black,1,:] += param['sor']*delta[i_black-1,j_black-1]

        if (np.abs(delta[1:-1,1:-1]/(padded_R[2:n,2:n,1]+1e-14))<=stop).all():
                break

        if (padded_R<0).any():
             print('numerical instabilities')
             sys.exit()

        # extract biggest update
        delta_list.append(np.max(np.abs(delta)))

        print("N_iter %d delta_max %e\r" % (len(delta_list)-1, delta_list[-1]), end='')

        # check for very small deltas
        if (np.abs(delta_list[-1])<1e-10):
            break
    
    R_eq = padded_R[1:n+1,1:n+1,1,:]

    # end timing and print time
    t1 = time()
    print('\n Time taken to solve for equilibrium: ', round((t1-t0)/60,4), ' minutes')

    return R_eq, up, prod

#------------------------------------------------------------------------------------------------------
# SOR_3D algorithm to iteratively solve a non linear Poisson problem of the form ∇^2*R = f(R) with NBC

def SOR_3D_noflux(N, param, mat, source, initial_guess):
    """
    N:             matrix, nxnxn_s, population grid
    param:         dictionary, parameters
    mat:           dictionary, matrices
    source:        function, (*args: R,N,param,mat; out: nxn source matrix), reaction in RD equation
    initial_guess: matrix, nxnxn_r, initial guess for eq. concentrations

    RETURNS R_eq: matrix, nxnxn_r, equilibrium concentration
            up  : matrix, nxnxn_r, equilibrium uptake
            prod: matrix, nxnxn_r, equilibrium production

    """

    # parameters needed for SOR implementation
    L     = param['L']                                    # float, length of the side 
    D     = param['D']                                    # float, diffusion constant
    rapp  = param['Dz']/D                                 # float, ratio between Dz and Dxy
    stop  = param['acc']                                  # float, convergence criterion
    bound = param['ext']                                  # vector, n_r, bottom D. BC

    n     = N.shape[0]                                    # int, number of grid points (currently) 
    h     = L/n                                           # float, finite difference step
    n_r   = len(bound)                                    # int, number of resources

    # updates initialization
    delta_list = [1]                                      # list, store the biggest update at each step
    delta = np.zeros((n,n,n_r))                           # matrix, nxnxn_r, store updates                      

    # create frame for quasi 3D BC
    padded_R = np.zeros((n+2,n+2,3,n_r))                  # initialize
    padded_R[1:n+1,1:n+1,2,:] = initial_guess             # newmann BC (no flux)
    padded_R[:,:,0,:] = np.tile(bound, (n+2, n+2, 1))     # fill bottom layer with D. BC
    padded_R[1:n+1,1:n+1,1,:] = initial_guess             # fill middle layer with initial guess

    # start timing
    t0 = time()

    # SOR algorithm, convergence condition based on update relative to current absolute value
    while (True):

        # periodic BC in middle layer
        padded_R[0,1:n+1,1,:]=padded_R[-2,1:n+1,1,:]
        padded_R[-1,1:n+1,1,:]=padded_R[1,1:n+1,1,:]
        padded_R[1:n+1,0,1,:]=padded_R[1:n+1,-2,1,:]
        padded_R[1:n+1,-1,1,:]=padded_R[1:n+1,1,1,:]

        # computing source
        padded_source = np.zeros((n+2,n+2,3,n_r))
        s, up, prod = source(padded_R[1:n+1,1:n+1,1,:],N,param,mat)
        padded_source[1:n+1,1:n+1,1] = s
          
        # prepare grid, assign red and black colors        
        i, j = np.mgrid[1:n+1, 1:n+1]
        checkerboard  = (i + j) % 2
        i_red, j_red = i[checkerboard == 0], j[checkerboard == 0]
        i_black, j_black = i[checkerboard == 1], j[checkerboard == 1]

        # red dots update
        delta[i_red-1,j_red-1,:] = 1/(4+rapp*2)*(padded_R[i_red,j_red+1,1,:]+padded_R[i_red,j_red-1,1,:]+padded_R[i_red+1,j_red,1,:]+padded_R[i_red-1,j_red,1,:]
                                   +rapp*(padded_R[i_red,j_red,0,:]+padded_R[i_red,j_red,2,:])
                                   +(h**3/D)*padded_source[i_red,j_red,1,:])-padded_R[i_red,j_red,1,:]
        padded_R[i_red,j_red,1,:] += param['sor']*delta[i_red-1,j_red-1]

        # repeat PBC
        # periodic BC in middle layer
        padded_R[0,1:n+1,1,:]=padded_R[-2,1:n+1,1,:]
        padded_R[-1,1:n+1,1,:]=padded_R[1,1:n+1,1,:]
        padded_R[1:n+1,0,1,:]=padded_R[1:n+1,-2,1,:]
        padded_R[1:n+1,-1,1,:]=padded_R[1:n+1,1,1,:]

        # black dots update
        delta[i_black-1,j_black-1,:] = 1/(4+rapp*2)*(padded_R[i_black,j_black+1,1,:]+padded_R[i_black,j_black-1,1,:]+padded_R[i_black+1,j_black,1,:]+padded_R[i_black-1,j_black,1,:]
                                   +rapp*(padded_R[i_black,j_black,0,:]+padded_R[i_black,j_black,2,:])
                                   +(h**3/D)*padded_source[i_black,j_black,1,:])-padded_R[i_black,j_black,1,:]
        padded_R[i_black,j_black,1,:] += param['sor']*delta[i_black-1,j_black-1]

        if (np.abs(delta[1:-1,1:-1]/(padded_R[2:n,2:n,1]+1e-14))<=stop).all():
                break

        if (padded_R<0).any():
             print('numerical instabilities')
             sys.exit()

        # no flux for the next iteration
        padded_R[1:n+1,1:n+1,2,:]=padded_R[1:n+1,1:n+1,1,:]

        # extract biggest update
        delta_list.append(np.max(np.abs(delta)))

        print("N_iter %d delta_max %e\r" % (len(delta_list)-1, delta_list[-1]), end='')

        # check for very small deltas
        if (np.abs(delta_list[-1])<1e-10):
            break
        if (len(delta_list)>1000):
            break
    
    R_eq = padded_R[1:n+1,1:n+1,1,:]

    # end timing and print time
    t1 = time()
    print('\n Time taken to solve for equilibrium: ', round((t1-t0)/60,4), ' minutes')

    return R_eq, up, prod
