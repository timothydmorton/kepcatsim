#!/usr/bin/env python
from __future__ import print_function, division

import sys
import os, os.path, shutil
import pandas as pd
import numpy as np
import logging

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from kepcatsim import simulate_survey

folder = sys.argv[1]
if os.path.exists(folder):
    shutil.rmtree(folder)
os.makedirs(folder)

if len(sys.argv) < 6:
    theta_true = [-0.3, -0.8, -1.5, 0.5, 0.3]
else:
    theta_true = sys.argv[2:]

print('Simulating survey...')

survey = simulate_survey(theta_true)
np.savetxt(os.path.join(folder,'theta.txt'), theta_true)

survey.stars[['mass_B', 'radius_B', 'flux_ratio']].to_hdf(os.path.join(folder,'binaries.h5'),'df')
survey.planets.to_hdf(os.path.join(folder,'kois.h5'),'df')

print('Binaries & Planets ({}) saved to {} folder.'.format(len(survey.planets),folder))

period_rng = survey.period_rng
rp_rng = survey.rp_rng
kois = survey.planets

# grab completeness & grid info
comp = survey.completeness
period_grid = survey.period_grid
rp_grid = survey.rp_grid

# A double power law model for the population.
def population_model(theta, period, rp):
    lnf0, beta, alpha = theta
    v = np.exp(lnf0) * np.ones_like(period)
    for x, rng, n in zip((period, rp),
                         (period_rng, rp_rng),
                         (beta, alpha)):
        n1 = n + 1
        v *= x**n*n1 / (rng[1]**n1-rng[0]**n1)
    return v

# The ln-likelihood function given at the top of this post.
koi_periods = np.array(kois.koi_period)
koi_rps = np.array(kois.koi_prad)
vol = np.diff(period_grid, axis=0)[:, :-1] * np.diff(rp_grid, axis=1)[:-1, :]
def lnlike(theta):
    pop = population_model(theta, period_grid, rp_grid) * comp
    pop = 0.5 * (pop[:-1, :-1] + pop[1:, 1:])
    norm = np.sum(pop * vol)
    ll = np.sum(np.log(population_model(theta, koi_periods, koi_rps))) - norm
    return ll if np.isfinite(ll) else -np.inf

# The ln-probability function is just propotional to the ln-likelihood
# since we're assuming uniform priors.
bounds = [(-5, 5), (-5, 5), (-5, 5)]
def lnprob(theta):
    # Broad uniform priors.
    for t, rng in zip(theta, bounds):
        if not rng[0] < t < rng[1]:
            return -np.inf
    return lnlike(theta)

# The negative ln-likelihood is useful for optimization.
# Optimizers want to *minimize* your function.
def nll(theta):
    ll = lnlike(theta)
    return -ll if np.isfinite(ll) else 1e15



# Optimize, and then run the chain
from scipy.optimize import minimize
theta_0 = np.array(theta_true[:3])
r = minimize(nll, theta_0, method="L-BFGS-B", bounds=bounds)

import emcee

ndim, nwalkers = len(r.x), 16
pos = [r.x + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

print('Running chains...')

# Burn in.
pos, _, _ = sampler.run_mcmc(pos, 1000)
sampler.reset()

# Production.
pos, _, _ = sampler.run_mcmc(pos, 4000)


import corner
corner.corner(sampler.flatchain, labels=[r"$\ln F$", r"$\beta$", r"$\alpha$"],
                truths=(theta_true[:3]))

plt.savefig(os.path.join(folder,'corner.png'))
                
np.save(os.path.join(folder,'chains.npy'), sampler.flatchain)

print('Done.  Chains and plots saved to {}.'.format(folder))
