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
    # beta = -0.2*(Muv + 19.5) - 2.05
    beta = -0.17 * Muv - 5.40
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
    f_star10, m_pivot = params
    
    sigma_star = 0.2393
    m_turn = 10**8.7
    alpha_1 = 0.4709
    alpha_2 = -0.61
    # m_pivot = 10**11.6
    baryon_frac = Planck18.Ob0 / Planck18.Om0

    high_mass_turnover_numerator = (m_pivot/1e10)**alpha_1 + (m_pivot/1e10)**alpha_2
    high_mass_turnover_denominator = (mh/m_pivot)**(-alpha_1) + (mh/m_pivot)**(-alpha_2)
    high_mass_turnover = high_mass_turnover_numerator / high_mass_turnover_denominator
    low_mass_turnover = np.exp(-m_turn/mh)
    mean_stellar_mass = f_star10 * baryon_frac * mh * (high_mass_turnover * low_mass_turnover)
    # stoc_adjust_term = 0.5*sigma_star**2# * np.log(10)**2
    # mean_stellar_mass *= 10**stoc_adjust_term
    p_mstar = gaussian(mstar_range[np.newaxis,:], np.log10(mean_stellar_mass)[:,np.newaxis], sigma_star)
    p_mstar[:, mstar_range > np.log10(mh * Planck18.Ob0 / Planck18.Om0)] = 0
    return mstar_range, p_mstar

def get_p_sfr(stellar_mass, params, redshift=9.0):
    sigma_sfr_lim, sigma_sfr_idx = params
    stellar_mass = 10**stellar_mass
    t_star = 0.1676
    t_h = 1/Planck18.H(redshift).to('yr**-1').value
    sfr_mean = stellar_mass / (t_star * t_h)
    sigma_sfr = np.maximum(
        sigma_sfr_lim + sigma_sfr_idx * np.log10(stellar_mass / 1e10),
        sigma_sfr_lim
    )
    # stoc_adjust_term = 0.5*sigma_sfr**2# * np.log(10)**2
    # sfr_mean *= 10**stoc_adjust_term
    p_sfr = gaussian(sfr_range[np.newaxis,:], np.log10(sfr_mean)[:,np.newaxis], sigma_sfr)
    return sfr_range, p_sfr

def get_p_muv(sfr, stellar_mass, redshift=9.0):
    sfr = 10**sfr
    stellar_mass = 10**stellar_mass
    kuv = interp_kuv(sfr[:,np.newaxis], stellar_mass[np.newaxis,:], redshift)
    muv_mean = -2.5 * np.log10(sfr[:,np.newaxis] * kuv) + 51.64
    # auv = get_auv(muv_mean)
    # muv_mean += auv
    sigma_kuv = 0.245
    p_muv = gaussian(muv_range[np.newaxis,np.newaxis,:], muv_mean[:,:, np.newaxis], sigma_kuv)
    return muv_range, p_muv

def get_p_muv_1d(sfr, stellar_mass, redshift=9.0):
    sfr = 10**sfr
    stellar_mass = 10**stellar_mass
    kuv = interp_kuv(sfr, stellar_mass, redshift)
    muv_mean = -2.5 * np.log10(sfr * kuv) + 51.64
    auv = get_auv(muv_mean)
    muv_mean += auv
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

redshift = 10.0
uvlf_p = uvlf_params(redshift)

stellar_params = [10**-2.81, 10**14.44]
sfr_params = [0.09297, -0.01884]

mstar_range, p_mstar = get_p_stellar_mass(mh_range, stellar_params)
sfr_range, p_sfr = get_p_sfr(mstar_range, sfr_params, redshift=redshift)
muv_range, p_muv = get_p_muv(sfr_range, mstar_range, redshift=redshift)

# get halo mass function
little_h = Planck18.H0.to("km s-1 Mpc-1").value / 100
hmf_ST = MassFunction(z=redshift, Mmin=5, Mmax=15, dlog10m=0.01, hmf_model='SMT')
m, dndlog10m = hmf_ST.m/Planck18.h, \
    hmf_ST.dndlog10m*Planck18.h**3*np.exp(-5e8/(hmf_ST.m/Planck18.h) )  # Msun, comoving Mpc^-3 Msun^-1
dndlog10m = np.interp(mh_range, np.log10(m), dndlog10m)

# propagate through to get P(M_UV|M_h)
p_muv_mh = np.einsum(
    'hs,sf,fsu -> hu',
    p_mstar,      # (Nh, Ns)
    p_sfr,        # (Ns, Nf)
    p_muv         # (Nf, Ns, Nu)
)

# norm = trapezoid(p_muv_mh, x=muv_range, axis=1)
# plt.plot(mh_range, norm)
# plt.show()
# quit()
# p_muv_mh /= norm[:, np.newaxis]
p_muv_mh /= 100

# propagate dn/dlog10Mh through to get the UVLF
phi_muv = np.einsum(
    'h,hu -> u',
    dndlog10m,    # (Nh,)
    p_muv_mh      # (Nh, Nu)
)

# plt.contourf(mh_range, muv_range, p_muv_mh.T, levels=50,
#                 cmap='hot')
# plt.xlabel(r'$\log_{10}(M_{\rm h}/M_\odot)$', fontsize=font_size)
# plt.ylabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
# plt.xlim(8, 15)
# plt.ylim(-24, -10)
# plt.show()
# quit()

mstar_width = mstar_range[1] - mstar_range[0]
sfr_width = sfr_range[1] - sfr_range[0]
mh_width = mh_range[1] - mh_range[0]
muv_width = muv_range[1] - muv_range[0]

uvlf_sum = trapezoid(phi_muv, x=muv_range)
hmf_sum = trapezoid(dndlog10m, x=mh_range)

print(f'Integral of UVLF: {uvlf_sum} cMpc^-3')
print(f'Integral of HMF: {hmf_sum} cMpc^-3')


x = [-20.817517711609586, -20.262772356824406, -19.795620906231633, -19.29926881450249, -18.598538965541092, -18.043796283828147, -17.605840128227275]
y = [-5.396492084703947, -4.526315789473685, -4.021052631578947, -3.6842105263157894, -3.43157894736842, -3.1228072317023017, -2.8701756527549334]

x = np.array(x)
y = np.array(y)

ysim = np.interp(x, muv_range, np.log10(phi_muv))
print(f'{ysim - y}')

# plt.plot(muv_range, np.log10(phi_muv), linestyle='-', color='cyan')

# auv = get_auv(muv_range)
# muv_range += auv

plt.plot(muv_range, np.log10(phi_muv), linestyle='-', color='cyan')
plt.plot(x, y, 'o', color='red', label='JWST')
plt.xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
plt.ylabel(r'$\frac{dn}{d{\rm M}_{\rm UV}}\;{\rm [cMpc^{-3}mag^{-1}]}$', fontsize=font_size)
plt.xlim(-24, -16)
plt.show()
quit()

fig, axs = plt.subplots(2, 3, figsize=(8,10), constrained_layout=True, sharex='col', sharey='col')

axs[0,0].contourf(mh_range, mstar_range, p_mstar.T, levels=50,
                cmap='hot')
axs[0,0].set_xlabel(r'$\log_{10}(M_{\rm h}/M_\odot)$', fontsize=font_size)
axs[0,0].set_ylabel(r'$\log_{10}(M_{\star}/M_\odot)$', fontsize=font_size)
axs[0,1].contourf(mstar_range, sfr_range, p_sfr.T, levels=50,
                cmap='hot')
axs[0,1].set_xlabel(r'$\log_{10}(M_{\star}/M_\odot)$', fontsize=font_size)
axs[0,1].set_ylabel(r'$\log_{10}({\rm SFR}/M_\odot {\rm yr}^{-1})$', fontsize=font_size)

axs[0,2].contourf(sfr_range, muv_range, p_muv.sum(axis=1).T, levels=50, 
                cmap='hot')
axs[0,2].set_xlabel(r'$\log_{10}({\rm SFR}/M_\odot {\rm yr}^{-1})$', fontsize=font_size)
axs[0,2].set_ylabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)

# get halo mass function
little_h = Planck18.H0.to("km s-1 Mpc-1").value / 100
hmf_ST = MassFunction(z=redshift, Mmin=5, Mmax=15, dlog10m=0.01, hmf_model='SMT')
m, dndlog10m = hmf_ST.m/Planck18.h, hmf_ST.dndlog10m*Planck18.h**3*np.exp(-5e8/(hmf_ST.m/Planck18.h) )  # Msun, comoving Mpc^-3 Msun^-1
p_mh = np.interp(mh_range, np.log10(m), dndlog10m)

# propagate through to get the UVLF
axs[1,0].contourf(mh_range, mstar_range, p_mh[:, np.newaxis]*p_mstar.T, levels=50,
                cmap='hot')
axs[1,0].set_xlabel(r'$\log_{10}(M_{\rm h}/M_\odot)$', fontsize=font_size)
axs[1,0].set_ylabel(r'$\log_{10}(M_{\star}/M_\odot)$', fontsize=font_size)
axs[1,0].set_xlim(8, 15)
axs[1,0].set_ylim(3, 13)

p_mstar_mh = trapezoid(p_mh[:, np.newaxis]*p_mstar.T, x=mh_range, axis=0)  # comoving Mpc^-3 dex^-2
p_sfr_mstar = p_mstar_mh[:, np.newaxis]*p_sfr

axs[1,1].contourf(mstar_range, sfr_range, p_sfr_mstar.T, levels=50,
                cmap='hot')
axs[1,1].set_xlabel(r'$\log_{10}(M_{\star}/M_\odot)$', fontsize=font_size)
axs[1,1].set_ylabel(r'$\log_{10}({\rm SFR}/M_\odot {\rm yr}^{-1})$', fontsize=font_size)
axs[1,1].set_xlim(3, 13)
axs[1,1].set_ylim(-5, 5)

p_mstar = trapezoid(p_mh[:, np.newaxis]*p_mstar, x=mh_range, axis=0) # comoving Mpc^-3 dex^-2
p_muv_sfr = trapezoid(p_mstar[:,np.newaxis]*p_sfr[..., np.newaxis]*p_muv, x=mstar_range, axis=1)

axs[1,2].contourf(sfr_range, muv_range, p_muv_sfr.T, levels=50,
                cmap='hot')
axs[1,2].set_xlabel(r'$\log_{10}({\rm SFR}/M_\odot {\rm yr}^{-1})$', fontsize=font_size)
axs[1,2].set_ylabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[1,2].set_xlim(-5, 5)
axs[1,2].set_ylim(-24, -10)
plt.show()

p_muv = trapezoid(p_muv_sfr, x=sfr_range, axis=0) # comoving Mpc^-3 dex^-1

# muv_b21, logphi_b21, logphi_err_b21_up, logphi_err_b21_low = get_ref_uvlf(redshift)
x = [-20.817517711609586, -20.262772356824406, -19.795620906231633, -19.29926881450249, -18.598538965541092, -18.043796283828147, -17.605840128227275]
y = [-5.396492084703947, -4.526315789473685, -4.021052631578947, -3.6842105263157894, -3.43157894736842, -3.1228072317023017, -2.8701756527549334]

x = np.array(x)
y = np.array(y)

fig, axs = plt.subplots(1, 4, figsize=(8,6), constrained_layout=True)
axs[0].plot(mh_range, p_mh, color='cyan')
axs[0].set_xlabel(r'$\log_{10}(M_{\rm h}/M_\odot)$', fontsize=font_size)
axs[0].set_ylabel(r'$\frac{dn}{d\log_{10}M_h}\;{\rm [cMpc^{-3}dex^{-1}]}$', fontsize=font_size)
axs[0].set_yscale('log')
axs[0].set_xlim(8, 15)

axs[1].plot(mstar_range, p_mstar, color='cyan')
axs[1].set_xlabel(r'$\log_{10}(M_{\star}/M_\odot)$', fontsize=font_size)
axs[1].set_ylabel(r'$\frac{dn}{d\log_{10}M_{\star}}\;{\rm [cMpc^{-3}dex^{-1}]}$', fontsize=font_size)
axs[1].set_yscale('log')
axs[1].set_xlim(3, 13)

p_sfr = trapezoid(p_mstar[:, np.newaxis]*p_sfr, x=mstar_range, axis=1)  # comoving Mpc^-3 dex^-2

axs[2].plot(sfr_range, p_sfr, color='cyan')
axs[2].set_xlabel(r'$\log_{10}({\rm SFR}/M_\odot {\rm yr}^{-1})$', fontsize=font_size)
axs[2].set_ylabel(r'$\frac{dn}{d\log_{10}{\rm SFR}}\;{\rm [cMpc^{-3}dex^{-1}]}$', fontsize=font_size)
axs[2].set_yscale('log')
axs[2].set_xlim(-5, 5)

ysim = np.interp(x, muv_range, np.log10(p_muv))
print(f'{ysim - y}')

axs[3].plot(muv_range, np.log10(p_muv), color='cyan')
axs[3].plot(x, y, 'o', color='red', label='JWST')
# axs[3].errorbar(muv_b21, logphi_b21, yerr=[logphi_err_b21_low, logphi_err_b21_up], fmt='o', color='red', label='Bouwens+2021')
axs[3].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[3].set_ylabel(r'$\frac{dn}{d{\rm M}_{\rm UV}}\;{\rm [cMpc^{-3}mag^{-1}]}$', fontsize=font_size)
axs[3].set_xlim(-24, -10)
# axs[3].set_ylim(-6, -1.75)
plt.show()