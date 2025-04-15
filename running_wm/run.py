"""
script for the sensitivity analysis preliminar to the obligate/facultative cross-feeding comparison
the goal is to obtain curves in the diversity-n_producers graph, at changing number of n_consumed, and 
across a range of conditions (i.e. parameter collections)

n_supplied is fixed to 1 and the supply regime is high
uptake is binary, energy content is ridcaled for changing n_consumed (in the F, for the O CF doesn't bring energy)

we vary: - leakage
         - PCS_var
         - PCS_bias

"""

import os
import sys
import pickle

import numpy as np
import pandas as pd

import well_mixed


def run_wm(df_index,leakage):

    """
    df_index: index in the networks dataframe, int
    leakage: leakage value, float

    RETURNS  pkl file with simulation dynamics of the selected index network and in the given leakage regime

    """

    # change path to where the dataframe is stored
    networks_df = pd.read_pickle('/Users/federicasibilla/Downloads/Auxotrophs/generating_networks/generated_networks_df.pkl')
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

    # externally supplied PCS
    n_ext = 1

    # all nutrients apport positive contributions to gr
    sign_mat_F = np.ones((n_s,n_r))
    sign_mat_O = np.zeros((n_s,n_r))
    sign_mat_O[:,0]=1  

    # essential specific
    mat_ess_F  = np.zeros((n_s,n_r))
    mat_ess_O  = np.where(up_mat_O!=0,1,0)
    mat_ess_O[:,0] = 0

    # recapitulate in dictionary
    mat_F = {
        'uptake'  : up_mat_F,
        'met'     : met_mat_F,
        'ess'     : mat_ess_F,
        'spec_met': spec_met,
        'sign'    : sign_mat_F
    }

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

    # define parameters
    param_F = {
        'w'  : np.ones((n_r))/(n_consumed+1),              # energy conversion     [energy/mass]
        'l'  : l_F,                                        # leakage               [adim]
        'g'  : g_F,                                        # growth conv. factors  [1/energy]
        'm'  : m,                                          # maintainance requ.    [energy/time]
        'ext': ext_F,                                      # external replenishment  
        'tau' : tau,                                       # chemicals dilution                                            
        'guess_wm': guess                                  # initial resources guess
    }
    # define parameters
    param_O = {
        'w'  : np.ones((n_r)),                             # energy conversion     [energy/mass]
        'l'  : l_O,                                        # leakage               [adim]
        'g'  : g_O,                                        # growth conv. factors  [1/energy]
        'm'  : m,                                          # maintainance requ.    [energy/time]
        'ext': ext_O,                                      # external replenishment  
        'tau' : tau,                                                                       
        'guess_wm': guess                                  # initial resources guess
    }

    initial_condition = np.random.normal(loc=1, scale=0.1, size=n_s)

    # run CR model for 200000 steps 
    N_fin_F,R_fin_F=well_mixed.run_wellmixed(initial_condition,param_F,mat_F,well_mixed.dR_dt_nomod,well_mixed.dN_dt,1000)
    # run CR model for 200000 steps 
    N_fin_O,R_fin_O=well_mixed.run_wellmixed(initial_condition,param_O,mat_O,well_mixed.dR_dt_maslov,well_mixed.dN_dt_maslov,1000)

    # ---------------------------------------------------------------------------------------------------------------------

    # save results
    data = {
        'n_producers':n_producers,
        'n_consumed':n_consumed,
        'replica':replica,
        'C_F':up_mat_F,
        'C_O':up_mat_O,
        'D_F':met_mat_F,
        'D_O':met_mat_O,
        'CR_R_F':R_fin_F[::200], # only save one every 10 time steps to make it lighter
        'CR_R_O':R_fin_O[::200], # only save one every 10 time steps to make it lighter
        'CR_N_F':N_fin_F[::200], # only save one every 10 time steps to make it lighter
        'CR_N_O':N_fin_O[::200], # only save one every 10 time steps to make it lighter
        'initial_condition':initial_condition
    }

    # output file path
    output_file = f'{results_dir}/all_data.pkl'

    # save as pickle
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)

    return 


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Check if there are command-line arguments
    if len(sys.argv) == 3:
        df_index  = int(sys.argv[1])
        leakage   = float(sys.argv[2])

        # Run the simulation with the provided parameters
        run_wm(df_index, leakage)
        print(f"Simulation completed for index {df_index} with leakage {leakage}.")

    else:
        print("Usage: python run_wm.py")