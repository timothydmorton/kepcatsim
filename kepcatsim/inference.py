from __future__ import print_function, division

import numpy as np
from scipy.optimize import minimize
import emcee

class DoublePowerLaw(object):
    """
    Double Power Law in period/radius

    theta = [lnf0, beta, alpha]
    """
    ndim = 3
    def __init__(self, period_rng, rp_rng):
        self.period_rng = period_rng
        self.rp_rng = rp_rng

    def __call__(self, theta, period, rp):
        lnf0, beta, alpha = theta
        v = np.exp(lnf0) * np.ones_like(period)
        for x, rng, n in zip((period, rp),
                             (self.period_rng, self.rp_rng),
                             (beta, alpha)):
            n1 = n + 1
            v *= x**n*n1 / (rng[1]**n1-rng[0]**n1)
        return v

    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    def lnprior(self, theta):
        for t, rng in zip(theta, self.bounds):
            if not rng[0] < t < rng[1]:
                return -np.inf
        return 0

class ProbabilisticModel(object):
    def __init__(self, survey, popmodel):
        self.survey = survey
        self.popmodel = popmodel

        self.period_grid = self.survey.period_grid
        self.rp_grid = self.survey.rp_grid
        self.comp = self.survey.completeness

        self.vol = np.diff(self.period_grid, axis=0)[:, :-1] *\
                     np.diff(self.rp_grid, axis=1)[:-1, :]

        self.koi_periods = self.survey.planets.koi_period
        self.koi_rps = self.survey.planets.koi_prad

    def lnlike(self, theta):
        pop = self.popmodel(theta, self.period_grid, self.rp_grid) * self.comp
        pop = 0.5 * (pop[:-1, :-1] + pop[1:, 1:])
        norm = np.sum(pop * self.vol)
        ll = np.sum(np.log(self.popmodel(theta, self.koi_periods, self.koi_rps))) - norm
        return ll if np.isfinite(ll) else -np.inf

    def nll(self, theta):
        ll = self.lnlike(theta)
        return -ll if np.isfinite(ll) else 1e15        

    def lnprob(self, theta):
        return self.lnlike(theta) + self.popmodel.lnprior(theta)

def run_mcmc(theta_0, model, nwalkers=16, nburn=1000, niter=4000):

    assert len(theta_0) == model.popmodel.ndim

    r = minimize(model.nll, theta_0, method="L-BFGS-B", 
                 bounds=model.popmodel.bounds)

    ndim = len(r.x)
    pos = [r.x + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model.lnprob)

    print('Running chains...')

    # Burn in.
    pos, _, _ = sampler.run_mcmc(pos, 1000)
    sampler.reset()

    # Production.
    pos, _, _ = sampler.run_mcmc(pos, 4000)

    return sampler
