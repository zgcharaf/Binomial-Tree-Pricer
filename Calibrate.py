import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

# Example real_data DataFrame
# real_data = pd.read_csv('path_to_your_real_data.csv')  # Load your real data
# For illustration purposes, let's create a dummy real_data DataFrame
np.random.seed(42)
real_data = pd.DataFrame({
    'Time': np.arange(0, 5000, 1),
    'Mid price': np.random.normal(loc=500, scale=5, size=5000)
})

# Define the likelihood function
def likelihood_function(params, real_data):
    setup = {
        "initbidpricetick" : params[0],
        "initaskpricetick" : params[1],
        "tickscale" : 0.01,
        "Nlattice" : 1000,
        "Nproviders" : 50,
        "Ntakers" : 50,
        "LOaggrA" : 0.5,
        "LOaggrB" : 5.0,
        "MOaggrA" : 5.0,
        "MOaggrB" : 0.5,
        "LOsignalpha" : 0.5,
        "MOvolpower" : 0.5,
        "MOvolcutoff" : 100,
        "meanLOratebid" : params[2],
        "meanLOrateask" : params[3],
        "meanMOratebid" : params[4],
        "meanMOrateask" : params[5],
        "meanCOratebid" : params[6],
        "meanCOrateask" : params[7],
        "meanHOrate" : params[8],
        "meanLOdecay" : params[9],
    }

    los = lobsim(setup, agentens=agentens)
    burn_in = 50.0
    tend, t = 5000.0, 0.0
    midps = []
    while t < tend:
        los.iterate()
        t = los.time
        midps.append([t, los.market_state_info["midprice"]])

    df = pd.DataFrame(midps, columns=['Time', 'Mid price'])
    df = df[df.Time > burn_in]
    simulated_mid_prices = df['Mid price'].values
    
    # Calculate negative log-likelihood
    nll = -np.sum(norm.logpdf(real_data['Mid price'].values, loc=np.mean(simulated_mid_prices), scale=np.std(simulated_mid_prices)))
    return nll

# Initial parameter guesses
initial_guess = [498, 502, 4.5, 4.5, 4.0, 4.0, 4.0, 4.0, 10.0, 5.0]

# Perform the optimization
result = minimize(likelihood_function, initial_guess, args=(real_data,), method='L-BFGS-B')

# Extract the estimated parameters
estimated_params = result.x
print(f"Estimated Parameters: {estimated_params}")

# Visualize the fitted results
setup = {
    "initbidpricetick" : estimated_params[0],
    "initaskpricetick" : estimated_params[1],
    "tickscale" : 0.01,
    "Nlattice" : 1000,
    "Nproviders" : 50,
    "Ntakers" : 50,
    "LOaggrA" : 0.5,
    "LOaggrB" : 5.0,
    "MOaggrA" : 5.0,
    "MOaggrB" : 0.5,
    "LOsignalpha" : 0.5,
    "MOvolpower" : 0.5,
    "MOvolcutoff" : 100,
    "meanLOratebid" : estimated_params[2],
    "meanLOrateask" : estimated_params[3],
    "meanMOratebid" : estimated_params[4],
    "meanMOrateask" : estimated_params[5],
    "meanCOratebid" : estimated_params[6],
    "meanCOrateask" : estimated_params[7],
    "meanHOrate" : estimated_params[8],
    "meanLOdecay" : estimated_params[9],
}

los = lobsim(setup, agentens=agentens)
burn_in = 50.0
tend, t = 5000.0, 0.0
midps = []
while t < tend:
    los.iterate()
    t = los.time
    midps.append([t, los.market_state_info["midprice"]])

df = pd.DataFrame(midps, columns=['Time', 'Mid price'])
df = df[df.Time > burn_in]
simulated_mid_prices = df['Mid price'].values

plt.figure(figsize=(10, 6))
plt.hist(real_data['Mid price'], bins=50, density=True, alpha=0.6, color='g', label='Real Data')
plt.hist(simulated_mid_prices, bins=50, density=True, alpha=0.6, color='b', label='Simulated Data')
plt.legend()
plt.show()
