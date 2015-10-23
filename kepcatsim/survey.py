from __future__ import print_function, division

import os, os.path
import numpy as np
import pandas as pd
import logging

from .utils import get_stellar, BURKE_QUERY
from .utils import get_completeness, BASEPATH
from .catalog import sim_binaries, sim_planets

def simulate_survey(theta, stars=None, query=BURKE_QUERY, 
                period_rng=(50,300), rp_rng=(0.75,20),
                mes_threshold=10, per_star=False):
    lnF0, beta, alpha, fB, gamma = theta

    if stars is None:
        stars = get_stellar(query)

    sim_binaries(stars, fB=fB, gamma=gamma)
    kois = sim_planets(theta[:3], stars, 
                        period_rng=period_rng, rp_rng=rp_rng,
                        mes_threshold=mes_threshold, per_star=per_star)

    return Survey(stars, kois, period_rng, rp_rng, mes_threshold)

def calc_completeness(stars, period_rng, rp_rng, 
                    npts_period=51, npts_rp=101, mes_threshold=10,
                    savefile=None):
    period = np.linspace(period_rng[0], 
                         period_rng[1], npts_period)
    rp = np.linspace(rp_rng[0], 
                     rp_rng[1], npts_rp)

    period_grid, rp_grid = np.meshgrid(period, rp, indexing="ij")
    comp = np.zeros_like(period_grid)
    for _, star in stars.iterrows():
        try:
            comp += get_completeness(star, period_grid, rp_grid, 
                                     0.0, with_geom=True, 
                                     thresh=mes_threshold)
        except ValueError:
            continue 

    if savefile is not None:
        np.savez(savefile, comp=comp, period_grid=period_grid,
                 rp_grid=rp_grid, inds=stars.index, 
                 thresh=mes_threshold)

    return comp, period_grid, rp_grid


class Survey(object):
    def __init__(self, stars, planets, 
                 period_rng, rp_rng, mes_threshold,
                 comp_filename='Burke'):
        """
        stars, planets are DataFrames
        period_rng, rp_rng are self-explanatory
        mes_threshold is the threshold at which planets were detected
        comp_filename is the path to the saved completeness data
        """

        self.stars = stars
        self.planets = planets
        self.period_rng = period_rng
        self.rp_rng = rp_rng
        self.mes_threshold = mes_threshold

        if comp_filename=='Burke':
            comp_filename = os.path.join(BASEPATH,'burke_completeness.npz')
        self.comp_filename = comp_filename

        self._comp_data = None
        self._comp_verified = False

    def calculate_completeness(self):
        print('Calculating completeness...')
        comp, period_grid, rp_grid = calc_completeness(self.stars,
                                        self.period_rng,
                                        self.rp_rng, 
                                        mes_threshold=self.mes_threshold)
        self._comp_data = {'comp':comp,
                            'period_grid':period_grid,
                            'rp_grid':rp_grid,
                            'inds':self.stars.index,
                            'thresh':self.mes_threshold}

        return self._comp_data

    @property
    def comp_data(self):
        if self._comp_data is None:
            self._comp_data = np.load(self.comp_filename)

        # Check to make sure completeness is for the correct stars/threshold
        if not self._comp_verified:
            if not np.all(self._comp_data['inds']==self.stars.index):
                raise ValueError('Completeness stars do not match survey stars!')
            if not self.mes_threshold==self._comp_data['thresh']:
                raise ValueError('Completeness MES threshold does not match survey!')
            self._comp_verified = True


        return self._comp_data

    @property
    def completeness(self):
        return self.comp_data['comp']

    @property
    def rp_grid(self):
        return self.comp_data['rp_grid']
    
    @property
    def period_grid(self):
        return self.comp_data['period_grid']

