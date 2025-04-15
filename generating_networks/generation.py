"""
generation.py

script for the actual generation of networks

IPUT: parameters range for parameters referring to the community matrices (number of auxotrophies, redundancy, number of species, number of resources, etc.)
OUTPUT: dataframe containing a network identificator and all the matrices for the obligate and facultative case

Requires the functions.py module

"""

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

import functions as fun

#------------------------------------------------------------------------------------------------------------------------------------------------------------

def generate_network(n_s, n_r, n_consumed, n_producers, PCS_sigma, mu, sigma, cost, replica):

    """
    n_s: number of species, int
    n_r: number of resources (primary+secondary), int
    n_consumed: number of consumed resources per species, int
    n_producers: number of producers per resource, int
    PCS_sigma: standard deviation of the uptake rate of the PCS, float
    mu: mean of the uptake rate of the SCS, float
    sigma: standard deviation of the uptake rate of the SCS, float
    cost: cost of the Dirichlet distributions, int
    replica: number of the replica, int
    
    RETURNS: a dictionary containing the network identificator and all the matrices for the obligate and facultative case
    
    """

    np.random.seed(replica)                                                           # set the seed for reproducibility

    spec_met = np.zeros((n_s, n_r))                                                   # initialize species-specific metabolic matrix
    attempts = 0                                                                      # counter for how many times the while loop runs

    while(np.any(np.sum(spec_met[:,1:],axis=0)!=n_producers)):                              # make sure n_producers is deterministic

        attempts += 1                                                                 # increment counter
        uptake_fac = fun.make_uptake_fac(n_s, n_r, n_consumed, PCS_sigma, mu, sigma)  # facultative uptake matrix
        uptake_oblig = fun.make_uptake_oblig(uptake_fac)                              # obligate uptake matrix

        spec_met = fun.make_spec_met(uptake_oblig, n_producers)                       # species-specific metabolic matrix

    met_fac = fun.make_met_fac(n_r, cost)                                  # facultative metabolic matrix
    met_oblig = fun.make_met_oblig(met_fac)                            # obligate metabolic matrix

    network_id = f'{n_s}_{n_r}_{n_consumed}_{n_producers}_{PCS_sigma}_{mu}_{sigma}_{cost}_{replica}'  
    print(f"Generating: {network_id} | Attempts: {attempts}", end='\r')               # inline print to track progress

    mat_dict = {
        'network_id':   network_id,
        'attempts':     attempts,
        'uptake_fac':   uptake_fac,
        'uptake_oblig': uptake_oblig,
        'spec_met':     spec_met,
        'met_fac':      met_fac,
        'met_oblig':    met_oblig
    }

    return mat_dict

#------------------------------------------------------------------------------------------------------------------------------------------------------------

n_s = 16
n_r = 20
PCS_sigma = 0.1
mu = 1.
sigma = 0.1
cost = 10

networks = []

for n_consumed in range(1,11):
    for n_producers in range(1,7):
        for replica in range(11):
            mat_dict = generate_network(n_s,n_r,n_consumed,n_producers,PCS_sigma,mu,sigma,cost,replica) # generate the network
            networks.append(mat_dict)

print("\nDone generating all networks.")                                                           # clean end of generation message

# Create DataFrame from list of dictionaries
df = pd.DataFrame(networks)

# Save DataFrame as pickle
df.to_pickle("/Users/federicasibilla/Downloads/Auxotrophs/generating_networks/generated_networks_df.pkl")