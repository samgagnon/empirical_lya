"""
Compares scatter in the SGH model with the Mason et al. (2018) model
and the exponential functions of Tang et al. (2024).
"""

import os

import numpy as np
import py21cmfast as p21c

from tqdm import tqdm

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p, e, m_e

from scipy.special import gamma, erf

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

presentation = False  # Set to True for presentation style
# presentation = True
if presentation == True:
    plt.style.use('dark_background')
    color1 = 'cyan'
    color2 = 'lime'
    color3 = 'orange'
    textcolor = 'white'
else:
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'
    color4 = 'orange'
    textcolor = 'black'

# from Bouwens 2021 https://arxiv.org/pdf/2102.07775
phi_5 = 0.79
muv_star_5 = -21.1
alpha_5 = -1.74

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

muv_list = np.linspace(-24, -16, 50)
popt_mu = [11.50962787, -1.25471915, -2.12854869, -21.99916644]
popt_std = [-0.50714459, -20.92567604, 1.72699987, 0.72541845]
popt_alpha = [-2.13037242e+01, 1.83486155e+00, 2.49700612e+00, 8.04770033e-03]
mu_val = double_power(muv_list, *popt_mu)
std_val = sigmoid(muv_list, *popt_std)
alpha_val = lorentzian(muv_list, *popt_alpha)

delta = alpha_val / np.sqrt(1 + alpha_val**2)
mean_mh = mu_val + std_val * delta * np.sqrt(2/np.pi)

mean_dv_m18 = 10**(0.32*np.log10((10**mean_mh) / 1.55e12) + 2.48)
scatter_dv_m18 = 0.24

mean_dv_sgh = -27.92 * (muv_list + 18.5) + 197.19
scatter_dv_sgh = 89.17

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

ax.plot(muv_list, mean_dv_m18, color=color2, lw=2)
ax.fill_between(muv_list, mean_dv_m18/10**scatter_dv_m18, mean_dv_m18*10**scatter_dv_m18, 
                color=color2, alpha=0.3, label='Mason+18')

ax.plot(muv_list, mean_dv_sgh, color=color3, lw=2)
ax.fill_between(muv_list, mean_dv_sgh - scatter_dv_sgh, mean_dv_sgh + scatter_dv_sgh, 
                color=color3, alpha=0.3, label='This Work')

ax.set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
ax.set_ylabel(r'$\Delta v$ [km s$^{-1}$]', fontsize=font_size)

ax.set_xlim(-24, -16)
ax.set_ylim(0, 1000)
ax.legend()
plt.show()