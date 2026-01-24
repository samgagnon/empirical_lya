"""
Comparing my LyA model to that of the expanding shell formulation employed by
https://arxiv.org/pdf/2510.18946
"""

import os

import numpy as np
import py21cmfast as p21c

from tqdm import tqdm

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

def change_of_basis(x, mu, sigma):
    """
    Change of basis to standard normal distribution.
    """
    return (x - mu) / sigma

def standard_normal_pdf(x):
    """
    Probability density function for a standard normal distribution.
    """
    return (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)

def standard_normal_cdf(x):
    """
    Cumulative distribution function for a standard normal distribution.
    """
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def skewnormal_pdf(x, mu, sigma, alpha):
    """
    Probability density function for a skew normal distribution.
    """
    x = change_of_basis(x, mu, sigma)
    phi = standard_normal_pdf(x)
    Phi = standard_normal_cdf(alpha * x)
    return 2 * phi * Phi

def double_power(x, a, b1, b2, c):
    """
    Double power law function.
    """
    x = x - c
    return a + x + np.log10(10**(x*b1) + 10**(x*b2))

def sigmoid(x, L, x0, k, b):
    """
    Sigmoid function.
    """
    return L / (1 + np.exp(-k*(x - x0))) + b

def lorentzian(x, x0, gamma, a, b):
    """
    Lorentzian function.
    """
    return a * (gamma**2) / ((x - x0)**2 + gamma**2) + b

def mh_from_muv(muv):
    """
    Get log10 halo mass from UV magnitude using the fitted skew normal parameters.
    """
    # https://en.wikipedia.org/wiki/Skew_normal_distribution
    # https://stats.stackexchange.com/questions/316314/sampling-from-skew-normal-distribution
    popt_mu = [11.50962787, -1.25471915, -2.12854869, -21.99916644]
    popt_std = [-0.50714459, -20.92567604, 1.72699987, 0.72541845]
    popt_alpha = [-2.13037242e+01, 1.83486155e+00, 2.49700612e+00, 8.04770033e-03]
    mu_val = double_power(muv, *popt_mu)
    std_val = sigmoid(muv, *popt_std)
    alpha_val = lorentzian(muv, *popt_alpha)
    standard_normal_samples = np.random.normal(0, 1, size=len(muv))
    p_flip = 0.5 * (1 + erf(-1*alpha_val*standard_normal_samples / np.sqrt(2)))
    u_samples = np.random.uniform(0, 1, size=len(muv))
    standard_normal_samples[u_samples < p_flip] *= -1
    standard_normal_samples *= std_val
    mh_samples = standard_normal_samples + mu_val
    return mh_samples

# TOGGLE HERE
# smsfr = 'sgh'
# smsfr = 'davies'
smsfr = 'nikolic'

if smsfr == 'sgh':
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

    def get_stellar_mass(halo_masses, stellar_rng):

        sigma_star = 4.83e-2  # dex
        mp1 = 1e10
        mp2 = 1.78e11
        M_turn = 10**(8.7)
        a_star = 9.96e-1
        a_star2 = -9.95e-1
        f_star10 = 1.95e-2
        omega_b = Planck18.Ob0
        omega_m = Planck18.Om0
        baryon_frac = omega_b/omega_m
        
        high_mass_turnover = ((mp2/mp1)**a_star + (mp2/mp1)**a_star2)/((halo_masses/mp2)**(-1*a_star)+(halo_masses/mp2)**(-1*a_star2))
        stoc_adjustment_term = 0.5*sigma_star**2
        low_mass_turnover = np.exp(-1*M_turn/halo_masses + stellar_rng*sigma_star - stoc_adjustment_term)
        stellar_mass = f_star10 * baryon_frac * halo_masses * (high_mass_turnover * low_mass_turnover)
        return stellar_mass
    
    def get_sfr(stellar_mass, sfr_rng, z):
        sigma_sfr_lim = 3.82e-1  # dex
        sigma_sfr_idx = -3.10e-1  # dex
        t_h = 1/Planck18.H(z).to('s**-1').value
        t_star = 3.80e-1
        sfr_mean = stellar_mass * 3.1557e7 / (t_star * t_h)
        sigma_sfr = sigma_sfr_idx * np.log10(stellar_mass/1e10) + sigma_sfr_lim
        sigma_sfr[sigma_sfr < sigma_sfr_lim] = sigma_sfr_lim
        stoc_adjustment_term = sigma_sfr * sigma_sfr / 2. # adjustment to the mean for lognormal scatter
        sfr_sample = sfr_mean * np.exp(sfr_rng*sigma_sfr - stoc_adjustment_term)
        return sfr_sample

    def get_muv(sfr_sample, stellar_mass, z):
        kuv = interp_kuv(sfr_sample, stellar_mass, z)
        muv = -2.5 * np.log10(sfr_sample * kuv) + 51.64
        return muv

if smsfr == 'nikolic':
    def get_stellar_mass(halo_masses, stellar_rng):
        sigma_star = 0.2393
        mp1 = 1e10
        mp2 = 10**14.44
        M_turn = 10**(8.7)
        a_star = 0.4709
        a_star2 = -0.61
        f_star10 = np.exp(-2.81)
        omega_b = Planck18.Ob0
        omega_m = Planck18.Om0
        baryon_frac = omega_b/omega_m
        
        high_mass_turnover = ((mp2/mp1)**a_star + (mp2/mp1)**a_star2)/((halo_masses/mp2)**(-1*a_star)+(halo_masses/mp2)**(-1*a_star2))
        stoc_adjustment_term = 0.5*sigma_star**2*np.log(10)**2
        low_mass_turnover = np.exp(-1*M_turn/halo_masses + stellar_rng*sigma_star - stoc_adjustment_term)
        stellar_mass = f_star10 * baryon_frac * halo_masses * (high_mass_turnover * low_mass_turnover)
        return stellar_mass

    def get_sfr(stellar_mass, sfr_rng, z):
        sigma_sfr_lim = 0.09297
        sigma_sfr_idx = -0.01884
        t_h = 1/Planck18.H(z).to('s**-1').value
        t_star = 0.1676
        sfr_mean = stellar_mass / (t_star * t_h)
        sigma_sfr = sigma_sfr_idx * np.log10(stellar_mass/1e10) + sigma_sfr_lim
        sigma_sfr[sigma_sfr < sigma_sfr_lim] = sigma_sfr_lim
        stoc_adjustment_term = 0.5 * (sigma_sfr)**2 * np.log(10)**2  # adjustment to the mean for lognormal scatter
        sfr_sample = sfr_mean * np.exp(sfr_rng*sigma_sfr - stoc_adjustment_term)
        # seconds per year
        return sfr_sample * 3.1557e7

    def interp_kuv(SFR, Mstar, z, bounds_error=True, fill_value=None, \
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

    def get_muv(SFR, Mstar, z):
        SIGMA_UV = 0.244947465229813069
        kuv = interp_kuv(SFR, Mstar, z)
        muv = -2.5 * np.log10(kuv * SFR) + 51.64
        stoc_adjustment_term = 0.5 * (SIGMA_UV)**2 * np.log(10)**2  # adjustment to the mean for lognormal scatter
        muv_sample = muv - (SIGMA_UV * np.random.normal(0, 1, size=len(muv))) -  stoc_adjustment_term
        return muv_sample

elif smsfr == 'davies':

    def get_stellar_mass(halo_masses, stellar_rng):
        sigma_star = 0.5  / np.log(10)
        mp1 = 1e10
        mp2 = 2.8e11
        M_turn = 10**(8.7)
        a_star = 0.5
        a_star2 = -0.61
        f_star10 = 0.05
        omega_b = Planck18.Ob0
        omega_m = Planck18.Om0
        baryon_frac = omega_b/omega_m
        
        high_mass_turnover = ((mp2/mp1)**a_star + (mp2/mp1)**a_star2)/((halo_masses/mp2)**(-1*a_star)+(halo_masses/mp2)**(-1*a_star2))
        stoc_adjustment_term = 0.5*sigma_star**2
        low_mass_turnover = np.exp(-1*M_turn/halo_masses)*10**(stellar_rng*sigma_star - stoc_adjustment_term)
        stellar_mass = f_star10 * baryon_frac * halo_masses * (high_mass_turnover * low_mass_turnover)
        return stellar_mass

    def get_sfr(stellar_mass, sfr_rng, z):
        sigma_sfr_lim = 0.19# / np.log(10)
        sigma_sfr_idx = -0.12
        t_h = 1/Planck18.H(z).to('s**-1').value
        t_star = 0.5
        sfr_mean = stellar_mass / (t_star * t_h)
        sigma_sfr = sigma_sfr_idx * np.log10(stellar_mass/1e10) + sigma_sfr_lim
        sigma_sfr[sigma_sfr < sigma_sfr_lim] = sigma_sfr_lim
        stoc_adjustment_term = sigma_sfr * sigma_sfr / 2. # adjustment to the mean for lognormal scatter
        sfr_sample = sfr_mean * 10**(sfr_rng*sigma_sfr - stoc_adjustment_term)
        return sfr_sample

    def get_muv(SFR, Mstar, z):
        muv = -2.5 * (np.log10(SFR) + np.log10(3.1557e7) + np.log10(1.15e28)) + 51.64
        return muv

def mvir(rvir, z):
    """
    Calculate the virial mass given the virial radius and redshift.
    """
    rho_c = Planck18.critical_density(z).to(u.Msun / u.kpc**3).value
    # Bryan & Norman 1998
    # https://ned.ipac.caltech.edu/level5/Sept11/Norman/Norman3.html#:~:text=The%20value%20of%20c%20is,where%20x%20=
    delta_c = 18 * np.pi**2 + 82 * (Planck18.Om(z) - 1) - 39 * (Planck18.Om(z) - 1)**2
    mvir = (4/3) * np.pi * rvir**3 * delta_c * rho_c
    return np.log10(mvir)
# 
def rvir(mh, z):
    """
    Calculate the virial radius given the halo mass and redshift.
    """
    rho_c = Planck18.critical_density(z).to(u.Msun / u.kpc**3).value
    delta_c = 18 * np.pi**2 + 82 * (Planck18.Om(z) - 1) - 39 * (Planck18.Om(z) - 1)**2
    rvir = (3 * 10**mh / (4 * np.pi * delta_c * rho_c))**(1/3)
    return rvir

def v_ff(r, mh):
    km_per_kpc = 3.09e16  # km/kpc
    return np.sqrt(G.to('kpc3 / (Msun * s2)').value * 10**mh / r) * km_per_kpc

from halomod.halo_model import MassFunction

def ms_mh_flattening(mh, fstar_norm=1.0, alpha_star_low=0.5, M_knee=2.6e11):
    """
        Get scaling relations for SHMR based on Davies+in prep.
        Parameters
        ----------
        mh: float,
            halo mass at which we're evaluating the relation.
        Returns
        ----------
        ms_mean: floats; optional,
            a and b coefficient of the relation.
    """

    f_star_mean = fstar_norm
    f_star_mean /= (mh / M_knee) ** (-alpha_star_low) + (mh / M_knee) ** 0.61 #knee denominator
    f_star_mean *= (1e10 / M_knee) ** (-alpha_star_low) + (1e10 / M_knee) ** 0.61 #knee numerator
    return np.minimum(f_star_mean, Planck18.Ob0 / Planck18.Om0) * mh

def SFMS(Mstar, SFR_norm=1., z=9.25):
    """
        the functon returns SFR from Main sequence
    """
    b_SFR = -np.log10(SFR_norm) + np.log10(Planck18.H(z).to(u.yr**(-1)).value)

    return Mstar * 10 ** b_SFR

def gimme_dust(Muv, beta = None):
    # from Kar+25, based on the fit from some 1999 paper and beta-Muv relation from Bouwens+15
    if beta is None:
        beta = -0.17 * Muv - 5.40
    elif str(beta) == 'Hubble':
        beta = -0.2 * (Muv+19.5) - 2
    Auv = 4.43 + 1.99 * beta
    return np.clip(Auv,0,5)

param_MAP_new = [
    -0.280636368880184506E+01, # log10 fstar norm
    0.239311686215938429E+00,  # sigma star
    0.167616139258982610E+00,  # SFR norm
    0.470918493138135108E+00,  # alpha star low
    0.929791798953653048E-01,  # sigma SFR 0
    -0.188417610576141620E-01, # sigma SFR 1
    0.144476204688384300E+02,  # log10 M_knee
    0.244956626802721206E+00   # sigma UV
]

n_halos = 100000
redshift = 8.0
little_h = Planck18.h

hmf_ST = MassFunction(z=redshift, Mmin=10, Mmax=15, dlog10m=0.05, hmf_model='SMT')

mh_samples = np.random.choice(
        hmf_ST.m / little_h,
        p=hmf_ST.dndlnm * (little_h**3) / np.sum(hmf_ST.dndlnm * (little_h**3)),
        size=n_halos
)

expected_num_halos = trapezoid(hmf_ST.dndlnm * (little_h**3), x=np.log(hmf_ST.m / little_h))

m, dndm = hmf_ST.m/Planck18.h, hmf_ST.dndm*Planck18.h**3  # Msun, comoving Mpc^-3 Msun^-1
N_HALOS_MPC3 = trapezoid(hmf_ST.dndlnm, x=np.log(hmf_ST.m)) * (Planck18.h**3)  # comoving Mpc^-3
EFFECTIVE_VOLUME = n_halos / N_HALOS_MPC3
mh = np.random.choice(m, size=n_halos, p=dndm/np.sum(dndm))

# verify_hmf = True
verify_hmf = False
if verify_hmf:
    edges = np.logspace(10, 15, num=60)
    widths = np.diff(edges)
    dlnm = np.log(edges[1:]) - np.log(edges[:-1])
    centres = (edges[:-1] * np.exp(dlnm / 2)).astype("f4")
    hist_s, _ = np.histogram(mh_samples, edges)
    mf_s = hist_s / EFFECTIVE_VOLUME / dlnm

    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    axs.loglog(centres, mf_s, color="C1", label="binned reference sample")
    axs.loglog(
        hmf_ST.m / little_h,
        hmf_ST.dndlnm * (little_h**3),
        color="C0",
        linewidth=2,
        linestyle=":",
        label="reference HMF",
    )
    axs.set_xlabel(r'$M_h$ [M$_\odot$]', fontsize=font_size)
    axs.set_ylabel(r'dn/dln$M_h$ [Mpc$^{-3}$]', fontsize=font_size)
    axs.legend(fontsize=int(font_size*0.6))
    plt.show()
    quit()


ms_from_mh = ms_mh_flattening(
    mh,
    alpha_star_low=param_MAP_new[3],
    fstar_norm=10**param_MAP_new[0],
    M_knee=10**param_MAP_new[6]
)

ms_scatter = np.random.normal(0, param_MAP_new[1], size=n_halos)
stellar_mass = ms_from_mh * 10**(ms_scatter + 0.5 * param_MAP_new[1]**2 * np.log(10)**2)

sfr_from_ms = SFMS(ms_from_mh, SFR_norm=param_MAP_new[2], z=redshift)
sigma_sfr = np.maximum(
    param_MAP_new[4] + param_MAP_new[5] * np.log10(stellar_mass / 1e10),
    param_MAP_new[4]
)
sfr_scatter = np.random.normal(0, sigma_sfr, size=n_halos)
sfr = sfr_from_ms * 10**(sfr_scatter + 0.5 * sigma_sfr**2 * np.log(10)**2)

muv = get_muv(sfr, stellar_mass, z=redshift)
muv_scatter = np.random.normal(0, param_MAP_new[7], size=n_halos)
muv = muv + muv_scatter

Auv = gimme_dust(muv)
muv += Auv

# from Bouwens 2021 https://arxiv.org/pdf/2102.07775
def uvlf_params(z):
    muv_star = -21.03 - 0.04 * (z - 6)
    phi = 4e-4 * 10**(-0.33*(z - 6) - 0.024*(z - 6)**2)
    alpha = -1.94 - 0.11 * (z - 6)
    return phi, muv_star, alpha

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha + 1)*\
        np.exp(-10**(0.4*(muv_star - muv)))

muv_bins = np.linspace(-24, -17, 10)
muv_centers = 0.5 * (muv_bins[1:] + muv_bins[:-1])

uvlf_p = uvlf_params(redshift)
p_muv = schechter(muv_centers, *uvlf_p)

heights, _ = np.histogram(muv, bins=muv_bins, density=False)
p_muv_sim = heights / (muv_bins[1] - muv_bins[0]) / EFFECTIVE_VOLUME
p_muv_sim_err = np.sqrt(heights) / (muv_bins[1] - muv_bins[0]) / EFFECTIVE_VOLUME

fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
axs.errorbar(muv_centers, p_muv_sim, yerr=p_muv_sim_err, fmt='o', color='cyan', label='This Work')
axs.plot(muv_centers, p_muv, linestyle='-', color='orange', label='Bouwens+21')
axs.set_yscale('log')
axs.set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs.set_ylabel(r'$\phi$ [Mpc$^{-3}$ mag$^{-1}$]', fontsize=font_size)
plt.show()
quit()

# mh_example = 10**10  # Msun
# r_max = rvir(np.log10(mh_example), z=7)  # kpc
# r_range = np.linspace(-1, np.log10(r_max), 100)  # kpc

# m_range = mvir(r_range, z=7)
# vff_range = v_ff(r_range, m_range) # km/s

# plt.plot(m_range, vff_range, color='white')
# plt.yscale('log')
# plt.xlabel('Radius (kpc)', fontsize=font_size)
# plt.ylabel('log10 Virial Mass (Msun)', fontsize=font_size)
# plt.show()
# quit()

# p_r = r_range**2
# p_r /= np.sum(p_r)

# sampled_r = np.random.choice(r_range, size=100000, p=p_r)
# sampled_vff = v_ff(sampled_r, np.log10(mh_example)) # km/s

# fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# axs[0].hist(np.log10(sampled_r), bins=20, density=True, color='cyan', alpha=0.7)
# axs[0].set_yscale('log')
# axs[0].set_xlabel('Radius (kpc)', fontsize=font_size)
# axs[0].set_ylabel('Probability Density', fontsize=font_size)
# axs[0].set_title('Radial Distribution', fontsize=font_size)

# axs[1].hist(sampled_vff, bins=20, density=True, color='magenta', alpha=0.7)
# axs[1].set_xlabel('Free-fall Velocity (km/s)', fontsize=font_size)
# axs[1].set_ylabel('Probability Density', fontsize=font_size)
# axs[1].set_yscale('log')
# axs[1].set_title('Free-fall Velocity Distribution', fontsize=font_size)

# plt.show()

def vcirc(mh, z):
    units_factor = G * Planck18.H(z)
    return (10 * units_factor.to('km3 / (Msun * s3)').value * 10**mh)**(1/3)

mh_range = np.linspace(10, 12, 100)
z_list = [5.0, 7.0, 9.0, 11.0, 13.0]
color_list = ['red', 'orange', 'yellow', 'green', 'blue']
for z, color in zip(z_list, color_list):
    rmax_range = rvir(mh_range, z)
    vmin_range = v_ff(rmax_range, mh_range)
    plt.plot(mh_range, vmin_range, color=color, label=f'z={z}')
    # plt.fill_between(mh_range, vmin_range, 1.19*vmin_range, color=color, alpha=0.3)
    plt.fill_between(mh_range, vmin_range, 1.63*vmin_range, color=color, alpha=0.5)
    # plt.fill_between(mh_range, vmin_range, 2.11*vmin_range, color=color, alpha=0.5)
# plt.yscale('log')
plt.xlabel(r'$\log_{10} M_h$ [M$_\odot$]', fontsize=font_size)
plt.ylabel(r'$\Delta v$ [km s$^{-1}$]', fontsize=font_size)
plt.legend(fontsize=int(font_size*0.6))
plt.xlim(10, 12)
plt.show()