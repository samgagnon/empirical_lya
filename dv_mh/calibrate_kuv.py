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

mstar_range = np.linspace(5, 12, 100)
sfr_range = np.linspace(-5, 5, 100)
muv_range = np.linspace(-24, -17, 100)

def get_auv(Muv):
    # from Kar+25, based on the fit from some 1999 paper and beta-Muv relation from Bouwens+15
    beta = -0.2*(Muv + 19.5) - 2.05
    Auv = 4.43 + 1.99 * beta
    return np.clip(Auv,0,5)

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
    f_star10, m_pivot, alpha_1, alpha_2, sigma_star = params
    
    m_turn = 10**8.7
    # alpha_1 = 0.4709
    # alpha_2 = -0.61
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
    t_star, sigma_sfr_lim, sigma_sfr_idx = params
    stellar_mass = 10**stellar_mass
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
    auv = get_auv(muv_mean)
    muv_mean += auv
    sigma_kuv = 0.245
    p_muv = gaussian(muv_range[np.newaxis,np.newaxis,:], muv_mean[:,:, np.newaxis], sigma_kuv)
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

def get_ref_uvlf(redshift):
    # BOUWENS 2021
    b21_mag = [[-22.52, -22.02, -21.52, -21.02, -20.52, -20.02, -19.52, -18.77, -17.77, -16.77],
            [-22.19, -21.69, -21.19, -20.68, -20.19, -19.69, -19.19, -18.69, -17.94, -16.94],
            [-21.85, -21.35, -20.85, -20.10, -19.35, -18.6, -17.6]]
    b21_phi = [[2e-6, 1.4e-5, 5.1e-5, 1.69e-4, 3.17e-4, 7.24e-4, 1.124e-3, 2.82e-3, 8.36e-3, 1.71e-2],
            [1e-6, 4.1e-5, 4.7e-5, 1.98e-4, 2.83e-4, 5.89e-4, 1.172e-3, 1.433e-3, 5.76e-3, 8.32e-3],
            [3e-6, 1.2e-5, 4.1e-5, 1.2e-4, 6.57e-4, 1.1e-3, 3.02e-3]]
    b21_phi_err = [[2e-6, 5e-6, 1.1e-5, 2.4e-5, 4.1e-5, 8.7e-5, 1.57e-4, 4.4e-4, 1.66e-3, 5.26e-3],
                [2e-6, 1.1e-5, 1.5e-5, 3.6e-5, 6.6e-5, 1.26e-4, 3.36e-4, 4.19e-4, 1.44e-3, 2.9e-3],
                [2e-6, 4e-6, 1.1e-5, 4e-5, 2.33e-4, 3.4e-4, 1.14e-3]]

    b21_6 = np.array(b21_phi[0])
    b21_7 = np.array(b21_phi[1])
    b21_8 = np.array(b21_phi[2])

    b21_6_err = np.array(b21_phi_err[0])
    b21_7_err = np.array(b21_phi_err[1])
    b21_8_err = np.array(b21_phi_err[2])

    logphi_b21_6 = np.log10(b21_6)
    logphi_b21_7 = np.log10(b21_7)
    logphi_b21_8 = np.log10(b21_8)

    logphi_err_b21_6_up = np.log10(b21_6 + b21_6_err) - logphi_b21_6
    logphi_err_b21_7_up = np.log10(b21_7 + b21_7_err) - logphi_b21_7
    logphi_err_b21_8_up = np.log10(b21_8 + b21_8_err) - logphi_b21_8

    logphi_err_b21_6_low = logphi_b21_6 - np.log10(b21_6 - b21_6_err)
    logphi_err_b21_7_low = logphi_b21_7 - np.log10(b21_7 - b21_7_err)
    logphi_err_b21_8_low = logphi_b21_8 - np.log10(b21_8 - b21_8_err)

    logphi_err_b21_6_low[np.isinf(logphi_err_b21_6_low)] = np.abs(logphi_b21_6[np.isinf(logphi_err_b21_6_low)])
    logphi_err_b21_7_low[np.isinf(logphi_err_b21_7_low)] = np.abs(logphi_b21_7[np.isinf(logphi_err_b21_7_low)])
    logphi_err_b21_8_low[np.isinf(logphi_err_b21_8_low)] = np.abs(logphi_b21_8[np.isinf(logphi_err_b21_8_low)])

    logphi_err_b21_6_low[np.isnan(logphi_err_b21_6_low)] = np.abs(logphi_b21_6[np.isnan(logphi_err_b21_6_low)])
    logphi_err_b21_7_low[np.isnan(logphi_err_b21_7_low)] = np.abs(logphi_b21_7[np.isnan(logphi_err_b21_7_low)])
    logphi_err_b21_8_low[np.isnan(logphi_err_b21_8_low)] = np.abs(logphi_b21_8[np.isnan(logphi_err_b21_8_low)])

    logphi_b21 = [logphi_b21_6, logphi_b21_7, logphi_b21_8]
    logphi_err_b21_up = [logphi_err_b21_6_up, logphi_err_b21_7_up, logphi_err_b21_8_up]
    logphi_err_b21_low = [logphi_err_b21_6_low, logphi_err_b21_7_low, logphi_err_b21_8_low]

    if redshift == 6.0:
        return b21_mag[0], logphi_b21[0], logphi_err_b21_up[0], logphi_err_b21_low[0]
    elif redshift == 7.0:
        return b21_mag[1], logphi_b21[1], logphi_err_b21_up[1], logphi_err_b21_low[1]
    elif redshift == 8.0:
        return b21_mag[2], logphi_b21[2], logphi_err_b21_up[2], logphi_err_b21_low[2]
    else:
        raise ValueError('Redshift not in B21 data.')

redshift = 6.0
uvlf_p = uvlf_params(redshift)
mh_range = np.linspace(8, 15, 100)

# get halo mass function
little_h = Planck18.h
hmf_ST = MassFunction(z=redshift, Mmin=7, Mmax=15, dlog10m=0.01, hmf_model='SMT')
m, dndlog10m = hmf_ST.m/Planck18.h, hmf_ST.dndlog10m*Planck18.h**3  # Msun, comoving Mpc^-3 Msun^-1
p_mh = np.interp(mh_range, np.log10(m), dndlog10m)

muv_b21, logphi_b21, logphi_err_b21_up, logphi_err_b21_low = get_ref_uvlf(redshift)

def objective(params):

    stellar_params = params[:5]
    sfr_params = params[5:]
    stellar_params[0] = 10**stellar_params[0]
    stellar_params[1] = 10**stellar_params[1]

    mstar_range, p_mstar = get_p_stellar_mass(mh_range, stellar_params)
    sfr_range, p_sfr = get_p_sfr(mstar_range, sfr_params, redshift=redshift)
    muv_range, p_muv = get_p_muv(sfr_range, mstar_range, redshift=redshift)

    p_mstar = trapezoid(p_mh[:, np.newaxis]*p_mstar, x=mh_range, axis=0)  # comoving Mpc^-3 dex^-2
    p_mstar_muv = trapezoid(p_mstar[np.newaxis,:,np.newaxis]*p_sfr.T[..., np.newaxis]*p_muv, x=sfr_range, axis=0)  # comoving Mpc^-3 dex^-2
    p_muv = trapezoid(p_mstar_muv, x=mstar_range, axis=0)  # comoving Mpc^-3 dex^-1

    # p_muv_ref = schechter(muv_range, *uvlf_p)
    log_p_muv = np.interp(muv_b21, muv_range, np.log10(p_muv))

    sigma = np.ones_like(logphi_b21)
    sigma[log_p_muv > logphi_b21] = logphi_err_b21_up[log_p_muv > logphi_b21]
    sigma[log_p_muv <= logphi_b21] = logphi_err_b21_low[log_p_muv <= logphi_b21]
    chi2 = np.sum(((log_p_muv - logphi_b21)/sigma)**2)
    # kl_divergence = np.abs(np.sum(p_muv_ref * np.log(p_muv_ref / p_muv)))
    # print(f'KL divergence: {kl_divergence}')
    return chi2

bounds = [(-3, -1), (11, 13), (0, 1), (-1, 0), (0.1, 0.5),\
          (0.1, 0.2), (0.01, 0.2), (-1.0, 0.0)]

# best fit parameters
# weird result, alpha 1 and alpha 2 should not be at the boundaries
# look for simulations to get a better prior on those
# [-2.53853777e+00  1.19069177e+01  1.00000000e+00 -1.00000000e+00
#   2.29793512e-01  1.80387025e-01  1.00000000e-02 -3.69578696e-01]

result = differential_evolution(objective, bounds, maxiter=20, disp=True)
print('Optimal parameters:', result.x)

stellar_params = result.x[:5]
sfr_params = result.x[5:]
# default parameters from ivan
# stellar_params = [-2.1, 11.6, 0.2393]
# sfr_params = [0.1676, 0.09, -0.09]
stellar_params[0] = 10**stellar_params[0]
stellar_params[1] = 10**stellar_params[1]

mstar_range, p_mstar = get_p_stellar_mass(mh_range, stellar_params)
sfr_range, p_sfr = get_p_sfr(mstar_range, sfr_params, redshift=redshift)
muv_range, p_muv = get_p_muv(sfr_range, mstar_range, redshift=redshift)

p_mstar = trapezoid(p_mh[:, np.newaxis]*p_mstar, x=mh_range, axis=0)  # comoving Mpc^-3 dex^-2
p_mstar_muv = trapezoid(p_mstar[np.newaxis,:,np.newaxis]*p_sfr.T[..., np.newaxis]*p_muv, x=sfr_range, axis=0)  # comoving Mpc^-3 dex^-2
p_muv = trapezoid(p_mstar_muv, x=mstar_range, axis=0) # comoving Mpc^-3 dex^-1

muv_b21, logphi_b21, logphi_err_b21_up, logphi_err_b21_low = get_ref_uvlf(redshift)

plt.figure(figsize=(8,6), constrained_layout=True)
# plt.plot(muv_range, p_muv_ref, label=r'${\it Hubble}$', color='cyan')
plt.errorbar(muv_b21, logphi_b21, yerr=[logphi_err_b21_low, logphi_err_b21_up], fmt='o', color='red', label='Bouwens+2021')
plt.plot(muv_range, np.log10(p_muv), label='This Work', color='orange')
# plt.yscale('log')
plt.xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
plt.ylabel(r'$\phi(M_{\rm UV})$ [cMpc$^{-3}$ mag$^{-1}$]', fontsize=font_size)
plt.legend(fontsize=int(font_size*0.8))
plt.show()