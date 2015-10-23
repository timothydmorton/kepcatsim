from __future__ import print_function, division

import pandas as pd
import numpy as np
import os, os.path
import requests
from cStringIO import StringIO
import logging

BASEPATH = os.getenv('KEPCATSIM',os.path.expanduser('~/.kepcatsim'))
if not os.path.exists(BASEPATH):
    os.makedirs(BASEPATH)

R_EARTH = 0.009171 # in solar units

"""
# From dan's blog:

# Select G and K dwarfs.
m = (4200 <= stlr.teff) & (stlr.teff <= 6100)
m &= stlr.radius <= 1.15

# Only include stars with sufficient data coverage.
m &= stlr.dataspan > 365.25*2.
m &= stlr.dutycycle > 0.6
m &= stlr.rrmscdpp07p5 <= 1000.

# Only select stars with mass estimates.
m &= np.isfinite(stlr.mass)
"""


BURKE_QUERY = "4200 <= teff <= 6100 and radius <= 1.15 " +\
              "and dataspan > 365.25*2. and dutycycle > 0.6 "+\
              "and rrmscdpp07p5 <= 1000. and mass > 0"

def get_burke_completeness():
    return np.load(os.path.join(BASEPATH, 'burke_completeness.npz'))

def get_catalog(name, basepath=BASEPATH):
    fn = os.path.join(basepath, "{0}.h5".format(name))
    if os.path.exists(fn):
        return pd.read_hdf(fn, name)
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    print("Downloading {0}...".format(name))
    url = ("http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/"
           "nph-nstedAPI?table={0}&select=*").format(name)
    r = requests.get(url)
    if r.status_code != requests.codes.ok:
        r.raise_for_status()
    fh = StringIO(r.content)
    df = pd.read_csv(fh)
    df.to_hdf(fn, name, format="t")
    return df

def get_stellar(query=BURKE_QUERY, basepath=BASEPATH):
    stlr = get_catalog('q1_q17_dr24_stellar', basepath=BASEPATH)
    return stlr.query(query)

def draw_powerlaw(alpha, rng, N=1):
    """
    Returns random variate according to x^alpha, between rng[0] and rng[1]
    """
    if alpha == -1:
        alpha = -1.0000001
    # Normalization factor
    x0, x1 = rng
    C = (alpha + 1) / (x1**(alpha + 1) - x0**(alpha + 1))
    
    if N==1:
        u = np.random.random()
    else:
        u = np.random.random(N)
    x = ((u * (alpha + 1)) / C + x0**(alpha + 1))**(1./(alpha + 1))

    return x

def get_duration(period, aor, e):
    """
    Equation (1) from Burke et al. This estimates the transit
    duration in the same units as the input period. There is a
    typo in the paper (24/4 = 6 != 4).
    
    :param period: the period in any units of your choosing
    :param aor:    the dimensionless semi-major axis (scaled
                   by the stellar radius)
    :param e:      the eccentricity of the orbit
    
    """
    return 0.25 * period * np.sqrt(1 - e**2) / aor

def get_a(period, mstar, Go4pi=2945.4625385377644/(4*np.pi*np.pi)):
    """
    Compute the semi-major axis of an orbit in Solar radii.
    
    :param period: the period in days
    :param mstar:  the stellar mass in Solar masses
    
    """
    return (Go4pi*period*period*mstar) ** (1./3)

def get_delta(k, c=1.0874, s=1.0187):
    """
    Estimate the approximate expected transit depth as a function
    of radius ratio. There might be a typo here. In the paper it
    uses c + s*k but in the public code, it is c - s*k:
    https://github.com/christopherburke/KeplerPORTs
    
    :param k: the dimensionless radius ratio between the planet and
              the star
    
    """
    delta_max = k*k * (c + s*k)
    return 0.84 * delta_max

def get_pgeom(aor, e):
    """
    The geometric transit probability.
    
    See e.g. Kipping (2014) for the eccentricity factor
    http://arxiv.org/abs/1408.1393
    
    :param aor: the dimensionless semi-major axis (scaled
                by the stellar radius)
    :param e:   the orbital eccentricity

    """
    return 1. / (aor * (1 - e*e)) * (aor > 1.0)

def get_pdet(star, aor, period, rp, e, thresh=None):
    """
    Equation (5) from Burke et al. Estimate the detection efficiency
    for a transit.
    
    :param star:   a pandas row giving the stellar properties
    :param aor:    the dimensionless semi-major axis (scaled
                   by the stellar radius)
    :param period: the period in days
    :param rp:     the planet radius in Earth radii
    :param e:      the orbital eccentricity
    
    """
    tau = get_duration(period, aor, e) * 24.
    depth = get_delta(rp * R_EARTH / star.radius)
    mes = get_mes(star, period, depth, tau)
    if thresh is not None:
        return mes > thresh
    
    mest = np.interp(tau, mesthres_vals,
                     np.array(star[mesthres_cols],
                              dtype=float))
    x = mes - 4.1 - (mest - 7.1)
    return pgam.cdf(x)

def get_pwin(star, period):
    """
    Equation (6) from Burke et al. Estimates the window function
    using a binomial distribution.
    
    :param star:   a pandas row giving the stellar properties
    :param period: the period in days
    
    """
    M = star.dataspan / period
    f = star.dutycycle
    omf = 1.0 - f
    pw = 1 - omf**M - M*f*omf**(M-1) - 0.5*M*(M-1)*f*f*omf**(M-2)
    msk = (pw >= 0.0) * (M >= 2.0)
    return pw * msk

def get_mes(star, period, depth, tau, re=0.009171):
    """
    Estimate the multiple event statistic value for a transit.
    
    :param star:   a pandas row giving the stellar properties
    :param period: the period in days
    :param depth:     the dimensionless depth of the transit signal
    :param tau:    the transit duration in hours
    
    """
    
    # Yes, this is goofy to do this every time, but so it goes.
    cdpp_cols = [k for k in star.keys() if k.startswith("rrmscdpp")]
    cdpp_vals = np.array([k[-4:].replace("p", ".") for k in cdpp_cols], dtype=float)
    
    # Interpolate to the correct CDPP for the duration.
    cdpp = np.array(star[cdpp_cols], dtype=float)
    sigma = np.interp(tau, cdpp_vals, cdpp)
    if np.any(sigma==0) or np.any(np.isnan(sigma)):
        raise ValueError('sigma is nan or zero!')
    
    # Compute the radius ratio and estimate the S/N.
    #k = rp * re / star.radius
    snr = depth * 1e6 / sigma
    
    # Scale by the estimated number of transits.
    ntrn = star.dataspan * star.dutycycle / period 
    return snr * np.sqrt(ntrn)



def get_completeness(star, period, rp, e, with_geom=True, thresh=None):
    """
    A helper function to combine all the completeness effects.
    
    :param star:      a pandas row giving the stellar properties
    :param period:    the period in days
    :param rp:        the planet radius in Earth radii
    :param e:         the orbital eccentricity
    :param with_geom: include the geometric transit probability?
    
    """
    aor = get_a(period, star.mass) / star.radius
    pdet = get_pdet(star, aor, period, rp, e, thresh=thresh)
    pwin = get_pwin(star, period)
    if not with_geom:
        return pdet * pwin
    pgeom = get_pgeom(aor, e)
    return pdet * pwin * pgeom