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

def get_stellar_mass(halo_masses, stellar_rng):
    sigma_star = 0.2393
    mp1 = 1e10
    mp2 = np.exp(14.44)
    M_turn = 10**(8.7)
    a_star = 0.4709
    a_star2 = -0.61
    f_star10 = np.exp(-2.81)
    omega_b = Planck18.Ob0
    omega_m = Planck18.Om0
    baryon_frac = omega_b/omega_m
    
    high_mass_turnover = ((mp2/mp1)**a_star + (mp2/mp1)**a_star2)/((halo_masses/mp2)**(-1*a_star)+(halo_masses/mp2)**(-1*a_star2))
    stoc_adjustment_term = 0.5*sigma_star**2
    low_mass_turnover = np.exp(-1*M_turn/halo_masses)*10**(stellar_rng*sigma_star - stoc_adjustment_term)
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
    stoc_adjustment_term = sigma_sfr * sigma_sfr / 2. # adjustment to the mean for lognormal scatter
    sfr_sample = sfr_mean * 10**(sfr_rng*sigma_sfr - stoc_adjustment_term)
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
    muv = -2.5 * np.log10(kuv * SFR) + 51.63
    stoc_adjustment_term = 0.5 * (SIGMA_UV)**2 * np.log(10)**2
    muv_sample = muv - (SIGMA_UV * np.random.normal(0, 1, size=len(muv))) -  stoc_adjustment_term
    return muv_sample

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

from halomod import TracerHaloModel

n_halos = 10000

hm_smt6 = TracerHaloModel(
    z=6.0,  # Redshift
    hmf_model="Tinker08",  # tinker 08 halo mass function
    cosmo_params={"Om0": Planck18.Om0, "H0": Planck18.H0.value},
)

m, dndm = hm_smt6.m, hm_smt6.dndm
dndm = dndm[m>1e10]
m = m[m>1e10]
p_m = dndm/np.sum(dndm)

from scipy.integrate import trapezoid

EFFECTIVE_VOLUME = n_halos/trapezoid(dndm, x=m)
print(f"Effective Volume: {EFFECTIVE_VOLUME:.2e} Mpc^3")
# quit()

mh = np.random.choice(m, size=n_halos, p=p_m)
# mh = 10**np.random.uniform(11, 13, size=n_halos)
sfr_rng = np.random.normal(0, 1, size=n_halos)
stellar_rng = np.random.normal(0, 1, size=n_halos)

# def get_stellar_mass(halo_masses, stellar_rng):
#     sigma_star = 0.5
#     mp1 = 1e10
#     mp2 = 2.8e11
#     M_turn = 10**(8.7)
#     a_star = 0.5
#     a_star2 = -0.61
#     f_star10 = 0.05
#     omega_b = Planck18.Ob0
#     omega_m = Planck18.Om0
#     baryon_frac = omega_b/omega_m
    
#     high_mass_turnover = ((mp2/mp1)**a_star + (mp2/mp1)**a_star2)/((halo_masses/mp2)**(-1*a_star)+(halo_masses/mp2)**(-1*a_star2))
#     stoc_adjustment_term = 0.5*sigma_star**2
#     low_mass_turnover = np.exp(-1*M_turn/halo_masses + stellar_rng*sigma_star - stoc_adjustment_term)
#     stellar_mass = f_star10 * baryon_frac * halo_masses * (high_mass_turnover * low_mass_turnover)
#     return stellar_mass

# def get_sfr(stellar_mass, sfr_rng, z):
#     sigma_sfr_lim = 0.19
#     sigma_sfr_idx = -0.12
#     t_h = 1/Planck18.H(z).to('s**-1').value
#     t_star = 0.5
#     sfr_mean = stellar_mass / (t_star * t_h)
#     sigma_sfr = sigma_sfr_idx * np.log10(stellar_mass/1e10) + sigma_sfr_lim
#     sigma_sfr[sigma_sfr < sigma_sfr_lim] = sigma_sfr_lim
#     stoc_adjustment_term = sigma_sfr * sigma_sfr / 2. # adjustment to the mean for lognormal scatter
#     sfr_sample = sfr_mean * np.exp(sfr_rng*sigma_sfr - stoc_adjustment_term)
#     return sfr_sample

# def get_muv(SFR, Mstar, z):
#     muv = -2.5 * (np.log10(SFR) + np.log10(3.1557e7) + np.log10(1.15e28)) + 51.64
#     return muv

mstar = get_stellar_mass(mh, stellar_rng)
sfr = get_sfr(mstar, sfr_rng, z=6.0)
muv = get_muv(sfr, mstar, z=6.0)

# from Bouwens 2021 https://arxiv.org/pdf/2102.07775
phi_5 = 0.79
muv_star_5 = -21.1
alpha_5 = -1.74

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**alpha*\
        np.exp(-10**(0.4*(muv_star - muv)))

muv_bins = np.linspace(-24, -17, 10)
muv_centers = 0.5 * (muv_bins[1:] + muv_bins[:-1])

p_muv = schechter(muv_centers, phi_5, muv_star_5, alpha_5) * 1e3 / EFFECTIVE_VOLUME
p_muv_err = np.sqrt(schechter(muv_centers, phi_5, muv_star_5, alpha_5) * 1e3) / EFFECTIVE_VOLUME

heights, _ = np.histogram(muv, bins=muv_bins)
heights = heights / (muv_bins[1] - muv_bins[0])
p_muv_sim = heights / EFFECTIVE_VOLUME
p_muv_sim_err = np.sqrt(heights) / EFFECTIVE_VOLUME
print(p_muv_sim, p_muv_sim_err)

plt.errorbar(muv_centers, p_muv_sim, yerr=p_muv_sim_err, fmt='o', color='cyan', label='Simulated z=6')
plt.errorbar(muv_centers, p_muv, yerr=p_muv_err, fmt='o', color='orange', label='Bouwens+21 z=5')
plt.yscale('log')
plt.xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
plt.ylabel(r'$\phi$ [Mpc$^{-3}$ mag$^{-1}$]', fontsize=font_size)
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