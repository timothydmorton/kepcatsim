from __future__ import print_function, division

import pandas as pd
import numpy as np
import logging
from scipy.stats import poisson
from numba import jit

from isochrones.dartmouth import Dartmouth_Isochrone

from .utils import draw_powerlaw 
from .utils import get_duration, get_a, get_delta, get_pgeom

dar = Dartmouth_Isochrone()
#because
dar.radius(1,9.8,0.0)

R_EARTH = 0.009171 # in solar units

# Some default ranges
P_RANGE = (50, 300) 
R_RANGE = (0.75, 20)


def sim_binaries(stars, fB=0.5, gamma=0.3, 
              qmin=0.1, minmass=None, band='Kepler', ic=dar):
    """
    Generates binary companions to stars



    Adds the following columns to DataFrame:
    mass_B, radius_B, flux_ratio
    """

    if minmass is None:
        minmass = ic.minmass

    N = len(stars)

    # which ones are binaries?
    b =  np.random.random(N) < fB
    nb = b.sum()

    # Simulate secondary masses
    qmin = np.maximum(qmin, minmass/stars.mass[b])
    q = draw_powerlaw(gamma, (qmin, 1), N=nb)
    M2 = q * stars.mass[b]

    # Stellar catalog doesn't have ages, so let's make them up.
    minage, maxage = ic.agerange(stars.mass[b], stars.feh[b])
    maxage -= 0.05
    age = np.random.random(size=nb) * (maxage - minage) + minage

    # Secondary properties (don't let secondary be bigger than primary)
    #  First, re-arrange arrays so they're organized nicely in memory 
    #   (This is a big time-saver!)
    M2 = np.ascontiguousarray(M2)
    feh = np.ascontiguousarray(stars.feh[b])

    R2 = ic.radius(M2, age, feh)
    R1 = stars.radius[b].values #to suppress pandas warninge
    toobig = R2 > R1
    R2[toobig] = R1[toobig]

    # Calculate secondary/primary flux ratio
    M1 = np.ascontiguousarray(stars.mass[b])
    dmag = (ic.mag[band](M2, age, feh) - 
            ic.mag[band](M1, age, feh))
    flux_ratio = 10**(-0.4 * dmag)

    # Assign columns appropriately.  Nans are those without binaries.
    #  Convoluted shit to avoid pandas "setting value on copy of slice" warning
    N = len(stars)
    mass_B = np.zeros(N)
    mass_B[b] = M2
    stars['mass_B'] = mass_B 

    radius_B = np.zeros(N)
    radius_B[b] = R2
    stars['radius_B'] = radius_B 

    fluxrat = np.zeros(N)
    fluxrat[b] = flux_ratio
    stars['flux_ratio'] = fluxrat

def draw_planet(theta_P, rp_rng=R_RANGE, period_rng=P_RANGE, N=1):
    """
    Returns radius and period for a planet, given parameters
    """
    _, beta, alpha = theta_P
    return (draw_powerlaw(alpha, rp_rng, N=N), 
            draw_powerlaw(beta, period_rng, N=N))
    

def get_sigma(tau, cdpp, cdpp_vals):
    return np.interp(tau, cdpp_vals, cdpp)
    

def sim_planets(theta_P, stars, 
                rp_rng=R_RANGE, period_rng=P_RANGE,
                mes_threshold=10, per_star=False):
    """
    Adds synthetic planets to stars

    stars needs to have binaries added already

    if per_star is True, then the occurrence rate is
    simulated to be *per star* and not *per system*.
    """
    lnf0, beta, alpha = theta_P

    if per_star:
        # Simulate planets for each *star*
        n_planets_A = poisson(np.exp(lnf0)).rvs((len(stars)))
        n_planets_B = poisson(np.exp(lnf0)).rvs((len(stars)))
        kepid = []
        idx = []
        which = []
        for nA, nB, kid, ix, mB in zip(n_planets_A, n_planets_B, 
                                   stars.kepid, stars.index, stars.mass_B):
            kepid += [kid]*nA
            idx += [ix]*nA
            which += ['A']*nA
            if mB > 0:
                kepid += [kid]*nB
                idx += [ix]*nB
                which += ['B']*nB
        kepid = np.array(kepid)
        which = np.array(which)
        N = len(kepid)

        has_binary = stars.ix[idx, 'mass_B'].values > 0
        A = has_binary & (which=='A')
        B = has_binary & (which=='B')
        single = ~has_binary

    else:     
        # Simulate planets for each *system*
        n_planets = poisson(np.exp(lnf0)).rvs(len(stars))

        kepid = []
        idx = []
        for n,kid,ix in zip(n_planets, stars.kepid, stars.index):
            kepid += [kid]*n
            idx += [ix]*n
        kepid = np.array(kepid)

        N = len(kepid)

        u = np.random.random(N)
        has_binary = stars.ix[idx, 'mass_B'].values > 0
        B = (u <= 0.5) & has_binary
        A = (u > 0.5) & has_binary
        single = ~has_binary
        which = np.array(['A']*N)
        which[B] = 'B'

    radius, period = draw_planet(theta_P, rp_rng=rp_rng,
                                period_rng=period_rng, N=N)

    Xr = np.ones(N)
    fluxrat = stars.ix[idx, 'flux_ratio']
    rad_A = stars.ix[idx, 'radius'].values
    rad_B = stars.ix[idx, 'radius_B'].values
    Xr[A] = np.sqrt(1 + fluxrat[A])
    Xr[B] = rad_A[B] / rad_B[B] * np.sqrt((1 + fluxrat[B]) / fluxrat[B])

    radius_observed = radius / Xr

    # Figure out which are transiting
    mass_A = stars.ix[idx, 'mass'].values
    mass_B = stars.ix[idx, 'mass_B'].values
    host_mass = mass_A * (A | single) + mass_B * B
    host_radius = rad_A * (A | single) + rad_B * B
    aor = get_a(period, host_mass)
    transit_prob = get_pgeom(aor / host_radius, 0.) # no ecc.

    u = np.random.random(N)
    transiting = u < transit_prob
    
    # Calculate the *observed* depth and duration of each planet
    depth = get_delta(radius_observed * R_EARTH / rad_A)
    tau = get_duration(period, aor, 0.) * 24 # no ecc.

    # Grab cdpp information from stars
    cdpp_cols = [k for k in stars.keys() if k.startswith("rrmscdpp")]
    cdpp_vals = np.array([k[-4:].replace("p", ".") for k in cdpp_cols], dtype=float)

    # Make access to cdpp values easier/faster:
    cdpp_array = stars.ix[idx, cdpp_cols].values

    # Interpolate cdpp at each tau
    sigma = np.zeros(N)
    for i in xrange(N):
        try:
            sigma[i] = get_sigma(tau[i], cdpp_array[i,:], cdpp_vals)
        except TypeError:
            logging.warning('No sigma for planet {}: tau={}'.format(i,tau[i]))

    # Estimate MES for each planet
    snr = depth * 1e6 / sigma
    ntrn = stars.ix[idx, 'dataspan'].values * \
                stars.ix[idx, 'dutycycle'].values / period
    mes = snr * np.sqrt(ntrn)

    # Select the planets that both transit and are detected
    detected = (mes > mes_threshold) & transiting

    planets = pd.DataFrame({'kepid':kepid[detected], 
                            'koi_prad':radius_observed[detected], 
                            'koi_period':period[detected],
                            'koi_prad_true':radius[detected], 
                            'koi_max_mult_ev':mes[detected],
                            'which':which[detected],
                            'star_index':np.array(idx)[detected]})


    return planets




