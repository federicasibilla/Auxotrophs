"""
run.py: script to run the spatial version of a given network
        takes in same input as the well-mixed run.py file (a dataframe collecting the networks to simulate NB it is not
        the same dataframe because there needs to be subsampling aftyer well-mixed simulations)

"""

import os
import sys
import pickle

import numpy as np
import pandas as pd

import R_dynamics
import N_dynamics
import SOR
import update

# find base path
# Path of the current file
current_file_path = os.path.abspath(__file__)
# Go up 2 levels (adjust as needed)
base_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))


def spatial_O(df_index, leakage):

    """
    i: int, index in the dataframe of the network to run

    """

    # change thsi to the subsampoled dataframe one available
    networks_df = pd.read_pickle(f'{base_path}/Auxotrophs/generating_networks/generated_networks_df.pkl')
    # extract parameters
    network     = networks_df.iloc[df_index] # select the network from the dataframe

    
    n_s, n_r, n_consumed, n_producers = networks_df['network_id'].str.split('_', expand=True).iloc[df_index, :4].astype(int)
    PCS_sigma, mu, sigma = networks_df['network_id'].str.split('_', expand=True).iloc[df_index, 4:7].astype(float)
    cost, replica = networks_df['network_id'].str.split('_', expand=True).iloc[df_index, 7:].astype(int)

    # extract matrices
    up_mat_F  = network['uptake_fac']
    up_mat_O  = network['uptake_oblig']
    met_mat_F = network['met_fac']
    met_mat_O = network['met_oblig']
    spec_met  = network['spec_met']

    # create paths to save results
    path = os.path.splitext(os.path.abspath(__file__))[0]
    results_dir = f"{path}_results/{df_index}_{leakage}"
    # Check if the directory already exists and is not empty
    if os.path.exists(results_dir) and os.listdir(results_dir):
        print(f"Directory {results_dir} exists and is not empty. Skipping...")
        return

    os.makedirs(results_dir, exist_ok=True)

    # more parameters: number of grid points
    n_ext = 1          # one supplied CS everyone can consume
    n = 100

    # generate uptake and metabolic matrices
    # essential specific
    mat_ess_F  = np.zeros((n_s,n_r))
    mat_ess_O  = np.where(up_mat_O!=0,1,0)
    mat_ess_O[:,0] = 0
    # all nutrients apport positive contributions to gr
    sign_mat_F = np.ones((n_s,n_r))
    sign_mat_O = np.zeros((n_s,n_r))
    sign_mat_O[:,0]=1 

    mat_O = {
        'uptake'  : up_mat_O,
        'met'     : met_mat_O,
        'ess'     : mat_ess_O,
        'spec_met': spec_met,
        'sign'    : sign_mat_O
    }

    # set dilution
    tau = 100

    # growth and maintainence
    g_F = np.ones((n_s))*1
    g_O = np.ones((n_s))*1
    m = np.zeros((n_s))+1/tau

    # reinsertion of chemicals
    tau = np.zeros((n_r))+tau 
    ext_F = np.zeros((n_r))
    ext_O = np.zeros((n_r))
    # primary carbon sources replenished to saturation
    ext_F[:n_ext] = 10000
    ext_O[:n_ext] = 10000

    # initial guess for resources
    guess = np.ones((n_r))*10000

    # leakage
    l_F = np.ones((n_r))*leakage
    l_O = np.zeros((n_r))
    l_O[0] = leakage

    # maint. cost 
    Rstar = (1/100)/100

    # define parameters
    param = {
        'w'  : np.ones((n_r))/(n_consumed+1),              # energy conversion     [energy/mass]
        'l'  : l_O,                                        # leakage               [adim]
        'g'  : g_O,                                        # growth conv. factors  [1/energy]
        'm'  : m,                                          # maintainance requ.    [energy/time]
        'ext': ext_O,                                      # external replenishment  
        'tau' : tau,                                       # chemicals dilution                                            
        'guess_wm': guess,                                 # initial resources guess
        'Rstar': Rstar                                     # initial resources guess
    }

    # definition of the rest of the model parameters
    
    param['n']=n                                 
    param['sor']=1.55
    param['L']=100
    param['D']=100
    param['Dz']=1e-4
    param['acc']=1e-5             

    # rescale influx so that wm and space compare in terms of supplied energy
    param['ext']=param['ext']/param['tau']

    # simulate in space
    # initial guesses and conditions
    R_space_ig = np.zeros((n,n,n_r))
    R_space_ig[:,:,param['ext']>0.]=param['ext'][0]/2
    N0_space   = np.zeros((n,n))
    N0_space   = N_dynamics.encode(np.random.randint(0, n_s, size=(n,n)),np.array(np.arange(n_s)))
    biomass = np.random.uniform(0, 2, (n, n)) 

    # define functions
    fR = R_dynamics.f_maslov
    gr = N_dynamics.growth_rates_maslov

    # spatial
    last_2_frames_N, _, current_R, current_N, g_rates, s_list, abundances, t_list, biomass  = update.simulate_3D_NBC(5, fR, gr, R_space_ig, N0_space, biomass, param, mat_O)

    data = {
                                        'n_consumed': n_consumed,
                                        'n_produced': n_producers,
                                        'parameters':param,
                                        'replica':replica,
                                        'C_O':up_mat_O,
                                        'D_O':met_mat_O,
                                        'spec':spec_met,
                                        'mat': mat_O
                                    }

    data['last_2_frames_N'] = last_2_frames_N
    data['current_R']       = current_R
    data['current_N']       = current_N
    data['g_rates']         = g_rates
    data['s_list']          = s_list
    data['abundances']      = abundances[::]#change later to only save some steps and make it lighter
    data['t_list']          = t_list
    data['biomass']         = biomass

    # output file path
    output_file = f'{results_dir}/all_data.pkl'

    # save as pickle
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)

    return 

    
#------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Check if there are command-line arguments
    if len(sys.argv) == 3:
        i = int(sys.argv[1])
        l = float(sys.argv[2])

        # Run the simulation with the provided parameters
        spatial_O(i,l)
        print(f"Simulation completed for row number {i} and leakage {l}")

    else:
        print("Usage: python run_O.py")