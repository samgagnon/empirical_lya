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

mh_example = 10**10  # Msun
r_max = rvir(np.log10(mh_example), z=7)  # kpc
r_range = np.linspace(-1, np.log10(r_max), 100)  # kpc

m_range = mvir(r_range, z=7)
vff_range = v_ff(r_range, m_range) # km/s

# plt.plot(m_range, vff_range, color='white')
# plt.yscale('log')
# plt.xlabel('Radius (kpc)', fontsize=font_size)
# plt.ylabel('log10 Virial Mass (Msun)', fontsize=font_size)
# plt.show()

p_r = r_range**2
p_r /= np.sum(p_r)

sampled_r = np.random.choice(r_range, size=100000, p=p_r)
sampled_vff = v_ff(sampled_r, np.log10(mh_example)) # km/s

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