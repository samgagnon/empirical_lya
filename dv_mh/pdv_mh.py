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
from astropy.constants import c, k_B, m_p, e, m_e

from scipy.special import gamma, erf
from scipy.optimize import curve_fit, differential_evolution

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

def pmh_from_muv(muv):
    """
    Get log10 halo mass from UV magnitude using the fitted skew normal parameters.
    This is the peak of the distribution.
    """
    # https://en.wikipedia.org/wiki/Skew_normal_distribution
    # https://stats.stackexchange.com/questions/316314/sampling-from-skew-normal-distribution
    popt_mu = [11.50962787, -1.25471915, -2.12854869, -21.99916644]
    popt_std = [-0.50714459, -20.92567604, 1.72699987, 0.72541845]
    popt_alpha = [-2.13037242e+01, 1.83486155e+00, 2.49700612e+00, 8.04770033e-03]
    mu_val = double_power(muv, *popt_mu)
    std_val = sigmoid(muv, *popt_std)
    alpha_val = lorentzian(muv, *popt_alpha)
    mh_space = np.linspace(6, 15, 1000)
    pdf = skewnormal_pdf(mh_space, mu_val, std_val, alpha_val)
    return mh_space, pdf

def pmh_conditional(muv, mh):
    """
    Get log10 halo mass from UV magnitude using the fitted skew normal parameters.
    This is the peak of the distribution.
    """
    # https://en.wikipedia.org/wiki/Skew_normal_distribution
    # https://stats.stackexchange.com/questions/316314/sampling-from-skew-normal-distribution
    popt_mu = [11.50962787, -1.25471915, -2.12854869, -21.99916644]
    popt_std = [-0.50714459, -20.92567604, 1.72699987, 0.72541845]
    popt_alpha = [-2.13037242e+01, 1.83486155e+00, 2.49700612e+00, 8.04770033e-03]
    mu_val = double_power(muv, *popt_mu)
    std_val = sigmoid(muv, *popt_std)
    alpha_val = lorentzian(muv, *popt_alpha)
    probability = skewnormal_pdf(mh, mu_val, std_val, alpha_val)
    return probability

# from Bouwens 2021 https://arxiv.org/pdf/2102.07775
phi_5 = 0.79
muv_star_5 = -21.1
alpha_5 = -1.74

def get_beta_bouwens14(muv):
    # https://arxiv.org/pdf/1306.2950
    return -2.05 + -0.2*(muv+19.5)

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha+1)*\
        np.exp(-10**(0.4*(muv_star - muv)))

# compute max_M_UV P (M_h|M_UV)
muv_range = np.linspace(-24, -16, 100)
mh_peaks, peak_probs = [], []
for muv in muv_range:
    mh_space, pdf = pmh_from_muv(np.array([muv]))
    peak_idx = np.argmax(pdf)
    mh_peaks.append(mh_space[peak_idx])
    peak_probs.append(pdf[peak_idx])

mh_peaks = np.array(mh_peaks)
peak_probs = np.array(peak_probs)

# fig, axs = plt.subplots(1, 2, figsize=(8,6))
# axs[0].plot(muv_range, mh_peaks, color='cyan', label=r'Peak $M_h|M_{UV}$')
# axs[0].set_xlabel(r'$M_{UV}$', fontsize=font_size)
# axs[0].set_ylabel(r'Peak $\log_{10} M_h$ [M$_\odot$]', fontsize=font_size)
# axs[0].set_title('Peak Halo Mass vs UV Magnitude', fontsize=font_size)

# axs[1].plot(mh_peaks, peak_probs, color='lime', label=r'Peak $P(M_h|M_{UV})$')
# axs[1].set_xlabel(r'$M_h$', fontsize=font_size)
# axs[1].set_ylabel(r'Peak Probability Density', fontsize=font_size)
# axs[1].set_title('Peak Probability Density vs UV Magnitude', fontsize=font_size)
# plt.show()
# quit()

NSAMPLES = 1000000

mh_range = np.linspace(8, 12, 10)

muv_space = np.linspace(-24, -16, NSAMPLES)
p_muv = schechter(muv_space, phi_5, muv_star_5, alpha_5)
n_gal = np.trapezoid(p_muv, x=muv_space)*1e-3 # galaxy number density in Mpc^-3
EFFECTIVE_VOLUME = NSAMPLES/n_gal  # Mpc3, for normalization

p_muv /= np.sum(p_muv)
muv_sample = np.random.choice(muv_space, size=NSAMPLES, p=p_muv)
dv_sample = np.random.normal(-27.92*(muv_sample+18.5)+197.19, 89.17, NSAMPLES)

mean_dv = []
sigma_dv = []
for mh in mh_range:
    p_accept_numerator = pmh_conditional(muv_sample, mh)
    p_accept_denominator = np.interp(mh, mh_peaks, peak_probs)
    p_accept = p_accept_numerator / p_accept_denominator
    u_random = np.random.uniform(0, 1, NSAMPLES)
    accepted_idxs = u_random < p_accept
    dv_accepted = dv_sample[accepted_idxs]
    mean_dv.append(np.mean(dv_accepted))
    sigma_dv.append(np.std(dv_accepted))

mean_dv = np.array(mean_dv)
sigma_dv = np.array(sigma_dv)

def vcirc(log10_mh):
    """
    Returns circular velocity in km/s as a function of MUV 
    at redshift 5.0.
    """
    return (log10_mh - 5.62)/3

plt.figure(figsize=(8,6))
plt.errorbar(mh_range, mean_dv, yerr=sigma_dv, fmt='o', color='cyan', label='This Work')
plt.plot(mh_range, 10**vcirc(mh_range), color='lime', linestyle='--', label='Expanding Shell Model')
plt.xlim(9, 12.5)
# plt.xlabel(r'$\log_{10} M_h$ [M$_\odot$]', fontsize=font_size)
# plt.ylabel(r'Ly$\alpha$ Velocity Offset $\Delta v$ [km s$^{-1}$]', fontsize=font_size)
# plt.title('Ly$\alpha$ Velocity Offset vs Halo Mass', fontsize=font_size)
plt.show()