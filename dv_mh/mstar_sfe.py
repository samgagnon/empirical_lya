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
mgas_range = np.linspace(3, 14, 100)

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
    # f_star10, m_pivot = params

    m_pivot = 14.44
    f_star10 = 10**-2.0
    
    sigma_star = 0.24
    m_turn = 10**5.0
    alpha_1 = 0.47
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

def get_p_gas_mass(mh, params):
    mh = 10**mh
    # f_star10, m_pivot = params

    m_pivot = 10**14.44
    f_star10 = 10**-2.0
    
    sigma_star = 0.24
    m_turn = 10**5.0
    alpha_1 = 0.47
    alpha_2 = -0.61
    baryon_frac = Planck18.Ob0 / Planck18.Om0

    high_mass_turnover_numerator = (m_pivot/1e10)**alpha_1 + (m_pivot/1e10)**alpha_2
    high_mass_turnover_denominator = (mh/m_pivot)**(-alpha_1) + (mh/m_pivot)**(-alpha_2)
    high_mass_turnover = high_mass_turnover_numerator / high_mass_turnover_denominator
    low_mass_turnover = np.exp(-m_turn/mh)
    mean_stellar_mass = f_star10 * baryon_frac * mh * (high_mass_turnover * low_mass_turnover)
    mean_gas_mass = mh * baryon_frac - mean_stellar_mass
    p_mgas = gaussian(mgas_range[np.newaxis,:], np.log10(mean_gas_mass)[:,np.newaxis], sigma_star)
    p_mgas[:, mgas_range > np.log10(mean_gas_mass)] = 0
    return mgas_range, p_mgas

def get_p_sfr(stellar_mass, params, redshift=9.0):
    sigma_sfr_lim, sigma_sfr_idx = 0.093, -0.019
    stellar_mass = 10**stellar_mass
    t_star = 0.17
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

# TODO what redshift again?
redshift = 13.0

# get halo mass function
hmf_ST = MassFunction(z=redshift, Mmin=5, Mmax=15, dlog10m=0.01, hmf_model='SMT')
m, dndlog10m = hmf_ST.m/Planck18.h, \
    hmf_ST.dndlog10m*Planck18.h**3*np.exp(-5e8/(hmf_ST.m/Planck18.h) )  # Msun, comoving Mpc^-3 Msun^-1
dndlog10m = np.interp(mh_range, np.log10(m), dndlog10m)
p_mh = dndlog10m / trapezoid(dndlog10m, mh_range)

# result = {'x': [10**-3.0229,  10**12,  0.3, -0.0522]}
result = {'x': [10**-2.5156,  10**12.147,  0.378299]} # what are the fit params Ivan found?
stellar_params = result['x'][:2]
sfr_params = result['x'][2:]

mstar_range, p_mstar = get_p_stellar_mass(mh_range, stellar_params)
sfr_range, p_sfr = get_p_sfr(mstar_range, sfr_params, redshift=redshift)
muv_range, p_muv = get_p_muv(sfr_range, mstar_range, redshift=redshift)
mgas_range, p_mgas = get_p_gas_mass(mh_range, stellar_params)

plt.figure(figsize=(8,6), constrained_layout=True)
plt.contourf(mh_range, mstar_range, p_mstar.T, levels=50, cmap='hot')
plt.plot(mh_range, mh_range + np.log10(Planck18.Ob0 / Planck18.Om0), color='white', linestyle='-', label='baryon fraction limit')
plt.xlabel(r'$M_h$ [$M_\odot$]', fontsize=font_size)
plt.ylabel(r'$M_*$ [$M_\odot$]', fontsize=font_size)
plt.title(f'Redshift z={redshift}', fontsize=font_size)
plt.gca().invert_xaxis()
# plt.show()
plt.close()

plt.figure(figsize=(8,6), constrained_layout=True)
plt.contourf(mh_range, mgas_range, p_mgas.T, levels=50, cmap='hot')
plt.xlabel(r'$M_h$ [$M_\odot$]', fontsize=font_size)
plt.ylabel(r'$M_g$ [$M_\odot$]', fontsize=font_size)
plt.title(f'Redshift z={redshift}', fontsize=font_size)
plt.gca().invert_xaxis()
# plt.show()
plt.close()

p_sfr_mh = np.einsum(
    'sf,hs -> hf',
    p_sfr,      # (Ns, Nf)
    p_mstar     # (Nh, Ns)
)

p_sfr_mh /= 100
p_sfr_mh /= trapezoid(p_sfr_mh, x=sfr_range, axis=1)[:, np.newaxis]  # Normalize over sfr for each mh

plt.figure(figsize=(8,6), constrained_layout=True)
plt.contourf(mh_range, sfr_range, p_sfr_mh.T, levels=50, cmap='hot')
plt.xlabel(r'$M_h$ [$M_\odot$]', fontsize=font_size)
plt.ylabel(r'$\log_{10}$ SFR [$M_\odot$ yr$^{-1}$]', fontsize=font_size)
plt.title(f'Redshift z={redshift}', fontsize=font_size)
plt.gca().invert_xaxis()
# plt.show()
plt.close()

sfe_range = np.linspace(-1, 0.5, 100)
num_mstar = p_mgas.shape[0]
num_sfe = len(sfe_range)
p_sfe_mh = np.zeros((num_mstar, num_sfe))

# p(sfe|mstar) = \int p(sfr|mh) p(mg=sfe-sfr|mh) dsfr
for i, sfr in enumerate(sfr_range):
    for j, sfe in enumerate(sfe_range):
        mg = 10**sfe / 10**sfr
        p_sfe_mh[:, j] += trapezoid(p_sfr_mh * np.interp(mg, mgas_range, p_mgas[:, j]), x=mh_range, axis=1)

# p(mh|mstar) = p(mstar|mh) p(mh) / p(mstar)
p_mh_mstar = p_mstar * p_mh / trapezoid(p_mstar * p_mh, x=mh_range, axis=0)

for i, mstar in enumerate(mstar_range):
    print(f'mstar={mstar:.2f}, p(mh|mstar)={trapezoid(p_mh_mstar[:,i], x=mh_range):.4e}')

# p(sfe|mstar) = \int p(sfe|mh) p(mh|mstar) dmh
p_sfe_mstar = np.zeros((num_mstar, num_sfe))
for i in range(num_mstar):
    p_sfe_mstar[i, :] = trapezoid(p_sfe_mh * np.interp(mh_range, mh_range, p_mh), x=mh_range)

# compute mean SFE for each M* and plot
mean_sfe = trapezoid(p_sfe_mstar * sfe_range, x=sfe_range, axis=1) / trapezoid(p_sfe_mstar, x=sfe_range, axis=1)
# compute 1 sigma confidence interval for SFE at each M*
sfe_lower = np.zeros((3, num_mstar))
sfe_upper = np.zeros((3, num_mstar))
for i in range(num_mstar):
    cdf = np.cumsum(p_sfe_mstar[i, :]) * np.diff(sfe_range)[0]
    cdf /= cdf[-1]  # Normalize to 1
    sfe_lower[0, i] = np.interp(0.16, cdf, sfe_range)
    sfe_upper[0, i] = np.interp(0.84, cdf, sfe_range)
    sfe_lower[1, i] = np.interp(0.025, cdf, sfe_range)
    sfe_upper[1, i] = np.interp(0.975, cdf, sfe_range)
    sfe_lower[2, i] = np.interp(0.0015, cdf, sfe_range)
    sfe_upper[2, i] = np.interp(0.9985, cdf, sfe_range)

plt.figure(figsize=(8,6), constrained_layout=True)

# plt.contourf(mstar_range, 10**sfe_range, p_sfe_mstar.T, levels=50, cmap='hot')
plt.plot(mstar_range,10**mean_sfe, color='cyan', linestyle='-', linewidth=2, label='mean SFE')
for i in range(3):
    plt.fill_between(mstar_range, 10**sfe_lower[i,:], 10**sfe_upper[i,:], color='cyan', \
                    alpha=0.3, label='1 sigma interval')
plt.xlabel(r'$\log_{10}M_*$ [$M_\odot$]', fontsize=font_size)
# TODO are we computing the ratio of the logs??? that's incorrect
plt.ylabel(r'$SFE$ [yr$^{-1}$]', fontsize=font_size)
# plt.ylabel(r'$\log_{10}$ SFE [yr$^{-1}$]', fontsize=font_size)
plt.title(f'Redshift z={redshift}', fontsize=font_size)
plt.xlim(3, 13)
# plt.gca().invert_xaxis()
plt.show()