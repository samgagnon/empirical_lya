import os

import numpy as np
import py21cmfast as p21c

from tqdm import tqdm

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p, e, m_e

from scipy.integrate import trapezoid
from scipy.special import gamma, erf
from scipy.optimize import differential_evolution

import matplotlib.pyplot as plt
rc = {"font.family" : "serif", 
    "mathtext.fontset" : "stix"}
plt.rcParams.update(rc) 
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 14})
presentation = False
# presentation = True
if presentation:
    plt.style.use('dark_background')
    cmap = 'Blues_r'
    color1 = 'white'
    color2 = 'cyan'
else:
    cmap = 'hot_r'
    color1 = 'black'
    color2 = 'black'

import matplotlib as mpl
label_size = 20
font_size = 30
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

# from Bouwens 2021 https://arxiv.org/pdf/2102.07775
phi_5 = 0.79
muv_star_5 = -21.1
alpha_5 = -1.74

lum_flux_factor = 4*np.pi*(Planck18.luminosity_distance(5.0).to('cm').value)**2

def get_beta_bouwens14(muv):
    # https://arxiv.org/pdf/1306.2950
    return -2.05 + -0.2*(muv+19.5)

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha+1)*\
        np.exp(-10**(0.4*(muv_star - muv)))

def normal_cdf(x, mu=0):
    """
    Cumulative distribution function for a normal distribution.
    """
    return 0.5 * (1 + erf((x - mu + mu/5) / (mu/5 * np.sqrt(2))))

def mh(muv, redshift):
    """
    Returns log10 Mh in solar masses as a function of MUV.
    """
    muv_inflection = -20.0 - 0.26*redshift
    gamma = 0.4*(muv >= muv_inflection) - 0.7
    return gamma * (muv - muv_inflection) + 11.75

def vcirc(muv, redshift):
    """
    Returns circular velocity in km/s as a function of MUV 
    at redshift 5.0.
    """
    redshift_factor = np.log10(Planck18.H(redshift).value) - np.log10(Planck18.H(5.0).value)
    log10_mh = mh(muv, redshift)
    return (log10_mh - 5.62 + redshift_factor)/3

def p_obs(lly, dv, lha, muv, theta, mode='wide'):
    """
    Probability of observing a galaxy with given Lya luminosity, H-alpha luminosity, and UV magnitude.
    """
    w1, w2, f1, f2, fh = theta
    # Convert luminosities to fluxes
    f_lya = lly / lum_flux_factor
    f_ha = lha / lum_flux_factor
    luv = 10**(0.4*(51.64 - muv))
    w_emerg = (1215.67/2.47e15)*(lly/luv)
    f_ha_lim = fh*2e-18  # H-alpha flux limit in erg/s/cm^2
    v_lim = 10**vcirc(muv, 5.0)
    if mode == 'wide':
        w_lim = 80*w1
        f_lya_lim = f1*2e-17
    elif mode == 'deep':
        w_lim = 25*w2
        f_lya_lim = f2*2e-18
    # https://arxiv.org/pdf/2202.06642
    # https://arxiv.org/pdf/2003.12083
    # muv_lim = -18.0
    muv_lim = -17.75

    p_v = normal_cdf(dv, (6/5)*v_lim)
    p_lya = normal_cdf(f_lya, f_lya_lim)
    p_ha = normal_cdf(f_ha, f_ha_lim)
    p_w = normal_cdf(w_emerg, w_lim)
    p_muv = 1 - normal_cdf(10**muv, 6*(10**muv_lim))
    
    return p_lya * p_ha * p_w * p_muv * p_v

def line(x, m, b):
    """
    Linear function.
    """
    return m * (x + 18.5) + b

# T = np.load('../data/pca/A.npy')
I = np.array([[1,0,0],[0,1,0],[0,0,1]])
A1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
A2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
A3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])
c1, c2, c3, c4 = 1, 1, 1/3, -1
T = c1 * I + c2 * A1 + c3 * A2 + c4 * A3

NSAMPLES = 100000
xc = np.load('../data/pca/xc.npy')
xstd = np.load('../data/pca/xstd.npy')
f = np.load('../data/pca/f.npy')
f_err = np.load('../data/pca/f_err.npy')
m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load('../data/pca/fit_params.npy')
theta = [w1, w2, f1, f2, fh]

redshift = 13.0
muv_sample = -18.5*np.ones(NSAMPLES)

# Fit the PCA coefficients to the observed fraction of LyA+Ha emitters
mu1 = line(muv_sample, m1, b1)
mu2 = line(muv_sample, m2, b2)
mu3 = line(muv_sample, m3, b3)

y1 = np.random.normal(mu1, std1, NSAMPLES)
y2 = np.random.normal(mu2, std2, NSAMPLES)
y3 = np.random.normal(mu3, std3, NSAMPLES)

# NOTE something is going wrong with the values here
Y = np.stack([y1, y2, y3], axis=-2)
X0 = T @ Y  # Transform to PCA basis

lly, dv, lha = X0[0], X0[1], X0[2]
lly = lly * xstd[0] + xc[0]
dv = dv * xstd[1] + xc[1]
lha = lha * xstd[2] + xc[2]

cov = np.cov([lly, dv, lha])
mean = np.array([np.mean(lly), np.mean(dv), np.mean(lha)])

def p_lya_conditional(lla_range, dv_min):
    mu_1 = mean[0]
    mu_2 = mean[1]
    cov_21 = cov[1,0]
    cov_11 = cov[0,0]
    cov_22 = cov[1,1]
    mu_21 = mu_2 + cov_21/cov_11 * (lla_range - mu_1)
    var_21 = cov_22 - cov_21**2/cov_11
    cdf_dv_cond = 0.5 * (1 - erf((dv_min - mu_21)/(np.sqrt(2*var_21))))
    p = np.exp(-(lla_range - mu_1)**2 / (2*cov_11)) * cdf_dv_cond
    p /= trapezoid(p, lla_range)
    return p.flatten()

def p_dv_conditional(dv_range, lla_min):
    mu_1 = mean[0]
    mu_2 = mean[1]
    cov_12 = cov[0,1]
    cov_11 = cov[0,0]
    cov_22 = cov[1,1]
    mu_12 = mu_1 + cov_12/cov_22 * (dv_range - mu_2)
    var_12 = cov_11 - cov_12**2/cov_22
    cdf_dv_cond = 0.5 * (1 - erf((lla_min - mu_12)/(np.sqrt(2*var_12))))
    p = np.exp(-(dv_range - mu_2)**2 / (2*cov_22)) * cdf_dv_cond
    p /= trapezoid(p, dv_range)
    return p.flatten()

lla_range = np.linspace(40, 44, 1000)
p_lla = np.exp(-(lla_range - mean[0])**2 / (2*cov[0,0]))
p_lla /= trapezoid(p_lla, lla_range)

lum_40A = (2.47e15/1215.67)*40*10**(0.4*(51.64 - (-18.5)))
log_lum_40A = np.log10(lum_40A)

v_lim_range = np.linspace(100, 500, 100)
p_40A = []
fig, axs = plt.subplots(2, 1, figsize=(6,10), constrained_layout=True, height_ratios=[2,1])
for dv_min in v_lim_range:
    p = p_lya_conditional(lla_range, dv_min)
    p_40A += [trapezoid(p[lla_range >= log_lum_40A], lla_range[lla_range >= log_lum_40A])]

p_40A = np.array(p_40A)
axs[0].plot(v_lim_range, p_40A, color=color2)
axs[0].set_xlabel(r'$\Delta v_{\rm min}$ [km s$^{-1}$]', fontsize=font_size)
axs[0].set_ylabel(r'$P(W_{\rm Ly\alpha}^{\rm emerg} > 40\;\AA|\Delta v>\Delta v_{\rm min})$', fontsize=font_size)
axs[0].set_xlim(100, 500)

dv_range = np.linspace(0, 500, 1000)
p = p_dv_conditional(dv_range, log_lum_40A)
axs[1].plot(dv_range, p, color=color2)
axs[1].set_xlabel(r'$\Delta v$ [km s$^{-1}$]', fontsize=font_size)
axs[1].set_ylabel(r'$P(\Delta v \mid W_{\rm Ly\alpha}^{\rm emerg} > 40\;\AA)$', fontsize=font_size)
axs[1].set_xlim(0, 500)
plt.show()

# lly, dv, lha = np.random.multivariate_normal(mean, cov, 100000).T
# fesc = 10**(lly - lha - np.log10(8.7))
# w_emerg = (1215.67/2.47e15)*(10**lly)*(10**(-0.4*(51.64 - (-18.5))))

# fig, axs = plt.subplots(1, 3, figsize=(18,6), constrained_layout=True)
# axs[0].hist(np.log10(w_emerg), density=True, bins=50, color=color2, alpha=0.7)
# axs[0].set_xlabel(r'$\log_{10}W^{{\rm Ly}\alpha}_{\rm emerg}$ [$\AA$]', fontsize=font_size)
# axs[1].hist(dv, density=True, bins=50, color=color2, alpha=0.7)
# axs[1].set_xlabel(r'$\Delta v$ [km s$^{-1}$]', fontsize=font_size)
# axs[2].hist(np.log10(fesc), density=True, bins=50, color=color2, alpha=0.7)
# axs[2].set_xlabel(r'$\log_{10}f_{\rm esc}^{\rm Ly\alpha}$', fontsize=font_size)
# plt.show()