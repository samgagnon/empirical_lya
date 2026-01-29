"""
Comparing my LyA model to that of the expanding shell formulation employed by
https://arxiv.org/pdf/2510.18946
"""

import os

import numpy as np
import py21cmfast as p21c

from tqdm import tqdm

from halomod.halo_model import MassFunction

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p, e, m_e, G

from scipy.integrate import trapezoid
from scipy.special import gamma, erf
from scipy.optimize import curve_fit, differential_evolution
from scipy.interpolate import RegularGridInterpolator

from ref_uvlf import get_ref_uvlf

import matplotlib.pyplot as plt
rc = {"font.family" : "serif", 
    "mathtext.fontset" : "stix"}
plt.rcParams.update(rc) 
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 14})
import matplotlib as mpl
label_size = 20
font_size = 30
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

plt.style.use('dark_background')

mstar_range = np.linspace(3, 13, 100)
sfr_range = np.linspace(-5, 5, 100)
muv_range = np.linspace(-24, -16, 100)
mh_range = np.linspace(8, 15, 100)

def get_auv(Muv):
    # from Kar+25, based on the fit from some 1999 paper and beta-Muv relation from Bouwens+15
    beta = -0.2*(Muv + 19.5) - 2.05 # Hubble
    # beta = -0.17 * Muv - 5.40     # JWST
    Auv = 4.43 + 1.99 * beta
    # NOTE this allows for unphysical negative dust attenuations
    # I am keeping this to avoid a weird kink in the UVLF at the faint end
    # return np.clip(Auv,-5,5)
    return Auv

def interp_kuv(SFR, Mstar, z, bounds_error=False, fill_value=1.15e28, \
            interpolation_table_loc = '../data/interpolation_table.npy'):
    """
    Trilinear interpolation on a regular grid.
    """
    table = np.load(interpolation_table_loc)
    SFR_grid = np.logspace(-5,5,100)
    Ms_grid = np.logspace(3,13,100)
    z_grid = np.linspace(5,15,10)

    interp = RegularGridInterpolator(
        (z_grid, Ms_grid, SFR_grid),
        table,
        method="linear",
        bounds_error=bounds_error,
        fill_value=fill_value,
    )
    return interp((z,Mstar,SFR))

def gaussian(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu)/sigma)**2)

def get_p_stellar_mass(mh, params):
    mh = 10**mh
    f_star10, m_pivot = params
    
    sigma_star = 0.2393
    m_turn = 10**8.7
    alpha_1 = 0.4709
    alpha_2 = -0.61
    baryon_frac = Planck18.Ob0 / Planck18.Om0

    high_mass_turnover_numerator = (m_pivot/1e10)**alpha_1 + (m_pivot/1e10)**alpha_2
    high_mass_turnover_denominator = (mh/m_pivot)**(-alpha_1) + (mh/m_pivot)**(-alpha_2)
    high_mass_turnover = high_mass_turnover_numerator / high_mass_turnover_denominator
    low_mass_turnover = np.exp(-m_turn/mh)
    mean_stellar_mass = f_star10 * baryon_frac * mh * (high_mass_turnover * low_mass_turnover)
    p_mstar = gaussian(mstar_range[np.newaxis,:], np.log10(mean_stellar_mass)[:,np.newaxis], sigma_star)
    p_mstar[:, mstar_range > np.log10(mh * Planck18.Ob0 / Planck18.Om0)] = 0
    return mstar_range, p_mstar

def get_p_sfr(stellar_mass, params, redshift=9.0):
    sigma_sfr_lim, sigma_sfr_idx = 0.0929, -0.0184
    stellar_mass = 10**stellar_mass
    # t_star = 0.1676
    t_star = params[0]
    t_h = 1/Planck18.H(redshift).to('yr**-1').value
    sfr_mean = stellar_mass / (t_star * t_h)
    sigma_sfr = np.maximum(
        sigma_sfr_lim + sigma_sfr_idx * np.log10(stellar_mass / 1e10),
        sigma_sfr_lim
    )
    p_sfr = gaussian(sfr_range[np.newaxis,:], np.log10(sfr_mean)[:,np.newaxis], sigma_sfr)
    return sfr_range, p_sfr

def get_p_muv(sfr, stellar_mass, redshift=9.0):
    sfr = 10**sfr
    stellar_mass = 10**stellar_mass
    kuv = interp_kuv(sfr[:,np.newaxis], stellar_mass[np.newaxis,:], redshift)
    muv_mean = -2.5 * np.log10(sfr[:,np.newaxis] * kuv) + 51.64
    sigma_kuv = 0.245
    p_muv = gaussian(muv_range[np.newaxis,np.newaxis,:], muv_mean[:,:, np.newaxis], sigma_kuv)
    return muv_range, p_muv

def get_p_muv_1d(sfr, stellar_mass, redshift=9.0):
    sfr = 10**sfr
    stellar_mass = 10**stellar_mass
    kuv = interp_kuv(sfr, stellar_mass, redshift)
    muv_mean = -2.5 * np.log10(sfr * kuv) + 51.64
    sigma_kuv = 0.245
    p_muv = gaussian(muv_range[np.newaxis,:], muv_mean[:, np.newaxis], sigma_kuv)
    return muv_range, p_muv

# from Bouwens 2021 https://arxiv.org/pdf/2102.07775
def uvlf_params(z):
    muv_star = -21.03 - 0.04 * (z - 6)
    phi = 4e-4 * 10**(-0.33*(z - 6) - 0.024*(z - 6)**2)
    alpha = -1.94 - 0.11 * (z - 6)
    return phi, muv_star, alpha

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha + 1)*\
        np.exp(-10**(0.4*(muv_star - muv)))

dndlog10m_list = []
muv_b21_dict = {}
logphi_b21_dict = {}
logphi_err_b21_up_dict = {}
logphi_err_b21_low_dict = {}

auv = get_auv(muv_range)
muv_obs = muv_range + auv
d_obs_dmuv = np.gradient(muv_obs, muv_range)
dust_correction = np.log10(d_obs_dmuv)

redshift = 6.0

# get halo mass function
hmf_ST = MassFunction(z=redshift, Mmin=5, Mmax=15, dlog10m=0.01, hmf_model='SMT')
m, dndlog10m = hmf_ST.m/Planck18.h, \
    hmf_ST.dndlog10m*Planck18.h**3*np.exp(-5e8/(hmf_ST.m/Planck18.h) )  # Msun, comoving Mpc^-3 Msun^-1
dndlog10m = np.interp(mh_range, np.log10(m), dndlog10m)

# result = {'x': [10**-3.0229,  10**12,  0.3, -0.0522]}
result = {'x': [10**-2.5156,  10**12.147,  0.378299]}
stellar_params = result['x'][:2]
sfr_params = result['x'][2:]

mstar_range, p_mstar = get_p_stellar_mass(mh_range, stellar_params)
sfr_range, p_sfr = get_p_sfr(mstar_range, sfr_params, redshift=redshift)
muv_range, p_muv = get_p_muv(sfr_range, mstar_range, redshift=redshift)

# propagate through to get P(M_UV|M_h)
p_muv_mh = np.einsum(
    'hs,sf,fsu -> hu',
    p_mstar,      # (Nh, Ns)
    p_sfr,        # (Ns, Nf)
    p_muv         # (Nf, Ns, Nu)
)

p_muv_mh /= 100

plt.figure(figsize=(8,6), constrained_layout=True)
plt.contourf(muv_range, mh_range, p_muv_mh, levels=50, cmap='hot')
plt.xlabel(r'$M_{UV}$', fontsize=font_size)
plt.ylabel(r'$\log_{10} M_h$ [M$_\odot$]', fontsize=font_size)
plt.title(f'Redshift z={redshift}', fontsize=font_size)
plt.gca().invert_xaxis()
plt.show()

# now show conditionals for select UV magnitudes
for muv_select in [-22, -20, -18, -16]:
    idx = np.argmin(np.abs(muv_range - muv_select))
    p_mh_given_muv = p_muv_mh[:, idx]
    p_mh_given_muv /= trapezoid(p_mh_given_muv, x=mh_range)
    plt.plot(mh_range, p_mh_given_muv, label=f'$M_{{UV}}={muv_select}$')
plt.xlabel(r'$\log_{10} M_h$ [M$_\odot$]', fontsize=font_size)
plt.ylabel(r'$P(M_h | M_{UV})$', fontsize=font_size)
plt.title(f'Redshift z={redshift}', fontsize=font_size)
plt.legend(fontsize=font_size)
plt.show()