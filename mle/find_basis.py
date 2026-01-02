"""
This script finds the PCA basis for the Lya luminosity function and the
H-alpha luminosity function from the Tang et al. (2024) data."

It does so by generating a data samples from a standard normal distribution, 
transforming them by a learned linear transformation, and then applying
the non-linear selection functions of MUSE+JWST to produce a mock observation.

The correlations within the mock observation is compared to the correlations
within the Tang et al. (2024) data. This is the loss function that is minimized
by the differential evolution algorithm to produce the PCA basis.
"""

import os

import numpy as np
import py21cmfast as p21c

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p, e, m_e

from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.special import gamma, erf
from scipy.optimize import differential_evolution, curve_fit

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
rc = {"font.family" : "serif", 
    "mathtext.fontset" : "stix"}
plt.rcParams.update(rc) 
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 14})
plt.style.use('dark_background')
import matplotlib as mpl
label_size = 20
font_size = 30
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

# from Bouwens 2021 https://arxiv.org/pdf/2102.07775
phi_5 = 0.79
muv_star_5 = -21.1
alpha_5 = -1.74

def line(x, m):
    """Linear function."""
    return m*x

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha+1)*\
        np.exp(-10**(0.4*(muv_star - muv)))

def p_muv(muv, phi, muv_star, alpha):
    return schechter(muv, phi, muv_star, alpha)/gamma(alpha_5+2)/(0.4*np.log(10))/phi_5

# measured lya properties from https://arxiv.org/pdf/2402.06070
MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, fescA, \
    fescA_err, fescB, fescB_err, ID = np.load('../data/tang24.npy').T

lum_lya = (ew_lya/1215.67) * 2.47e15 * 10**(0.4*(51.6 - MUV))
lum_lya_err = lum_lya*np.sqrt((ew_lya_err/ew_lya)**2 + (np.exp(-0.4*np.log(10)*MUV_err)/MUV)**2)

lum_ha = lum_lya / 11.4 / fescA
lum_ha_err = lum_ha*np.sqrt((lum_lya_err/lum_lya)**2 + (fescA_err/fescA)**2)

lum_flux_factor = 4*np.pi*(Planck18.luminosity_distance(5.0).to('cm').value)**2

distance_ratio = Planck18.luminosity_distance(5.0).to('cm').value / \
    Planck18.luminosity_distance(z).to('cm').value

wide = ID==0
_muv_wide = MUV[wide]
_muv_err_wide = MUV_err[wide]
_dv_lya_wide = dv_lya[wide]
_dv_lya_err_wide = dv_lya_err[wide]
_lum_lya_wide = lum_lya[wide]*distance_ratio[wide]
_lum_lya_err_wide = lum_lya_err[wide]
_lum_ha_wide = lum_ha[wide]*distance_ratio[wide]
_lum_ha_err_wide = lum_ha_err[wide]

deep = ID==1
_muv_deep = MUV[deep]
_muv_err_deep = MUV_err[deep]
_dv_lya_deep = dv_lya[deep]
_dv_lya_err_deep = dv_lya_err[deep]
_lum_lya_deep = lum_lya[deep]*distance_ratio[deep]
_lum_lya_err_deep = lum_lya_err[deep]
_lum_ha_deep = lum_ha[deep]*distance_ratio[deep]
_lum_ha_err_deep = lum_ha_err[deep]

muv_wide = np.zeros(len(_muv_wide)*1000)
lum_lya_wide = np.zeros_like(muv_wide)
dv_wide = np.zeros_like(muv_wide)
lum_ha_wide = np.zeros_like(muv_wide)

muv_deep = np.zeros(len(_muv_deep)*1000)
lum_lya_deep = np.zeros_like(muv_deep)
dv_deep = np.zeros_like(muv_deep)
lum_ha_deep = np.zeros_like(muv_deep)

for i, (muv, muve, lly, llye, dv, dve, lha, lhae) in enumerate(zip(_muv_wide, _muv_err_wide, _lum_lya_wide, _lum_lya_err_wide, \
                                                 _dv_lya_wide, _dv_lya_err_wide, _lum_ha_wide, _lum_ha_err_wide)):
    muv_wide[1000*i:1000*(i+1)] = np.random.normal(muv, muve, 1000)
    lum_lya_wide[1000*i:1000*(i+1)] = np.random.normal(lly, llye, 1000)
    dv_wide[1000*i:1000*(i+1)] = np.random.normal(dv, dve, 1000)
    lum_ha_wide[1000*i:1000*(i+1)] = np.random.normal(lha, lhae, 1000)
    lum_lya_wide[1000*i:1000*(i+1)][lum_lya_wide[1000*i:1000*(i+1)]<10] = np.mean(lum_lya_wide[1000*i:1000*(i+1)])  # replace with mean
    dv_wide[1000*i:1000*(i+1)][dv_wide[1000*i:1000*(i+1)]<10] = np.mean(dv_wide[1000*i:1000*(i+1)])  # replace with mean
    lum_ha_wide[1000*i:1000*(i+1)][lum_ha_wide[1000*i:1000*(i+1)]<10] = np.mean(lum_ha_wide[1000*i:1000*(i+1)])  # replace with mean

    
for i, (muv, muve, lly, llye, dv, dve, lha, lhae) in enumerate(zip(_muv_deep, _muv_err_deep, _lum_lya_deep, _lum_lya_err_deep, \
                                                 _dv_lya_deep, _dv_lya_err_deep, _lum_ha_deep, _lum_ha_err_deep)):
    muv_deep[1000*i:1000*(i+1)] = np.random.normal(muv, muve, 1000)
    lum_lya_deep[1000*i:1000*(i+1)] = np.random.normal(lly, llye, 1000)
    dv_deep[1000*i:1000*(i+1)] = np.random.normal(dv, dve, 1000)
    lum_ha_deep[1000*i:1000*(i+1)] = np.random.normal(lha, lhae, 1000)
    lum_lya_deep[1000*i:1000*(i+1)][lum_lya_deep[1000*i:1000*(i+1)]<10] = np.mean(lum_lya_deep[1000*i:1000*(i+1)])  # replace with mean
    dv_deep[1000*i:1000*(i+1)][dv_deep[1000*i:1000*(i+1)]<10] = np.mean(dv_deep[1000*i:1000*(i+1)])  # replace with mean
    lum_ha_deep[1000*i:1000*(i+1)][lum_ha_deep[1000*i:1000*(i+1)]<10] = np.mean(lum_ha_deep[1000*i:1000*(i+1)])  # replace with mean

def normal_cdf(x, mu=0):
    """
    Cumulative distribution function for a normal distribution.
    """
    return 0.5 * (1 + erf((x - mu + mu/5) / (mu/5 * np.sqrt(2))))

def mh(muv):
    """
    Returns log10 Mh in solar masses as a function of MUV.
    """
    redshift = 5.0
    muv_inflection = -20.0 - 0.26*redshift
    gamma = 0.4*(muv >= muv_inflection) - 0.7
    return gamma * (muv - muv_inflection) + 11.75

def vcirc(muv):
    """
    Returns circular velocity in km/s as a function of MUV 
    at redshift 5.0.
    """
    log10_mh = mh(muv)
    return (log10_mh - 5.62)/3

def p_obs(lly, dv, lha, muv, mode='wide'):
    """
    Probability of observing a galaxy with given Lya luminosity, H-alpha luminosity, and UV magnitude.
    """
    # Convert luminosities to fluxes
    f_lya = lly / lum_flux_factor
    f_ha = lha / lum_flux_factor
    luv = 10**(0.4*(51.64 - muv))
    w_emerg = (1215.67/2.47e15)*(lly/luv)
    f_ha_lim = 2e-18  # H-alpha flux limit in erg/s/cm^2
    v_lim = 10**vcirc(muv)
    if mode == 'wide':
        w_lim = 80
        f_lya_lim = 2e-17
    elif mode == 'deep':
        w_lim = 8
        f_lya_lim = 2e-18
    # https://arxiv.org/pdf/2202.06642
    # https://arxiv.org/pdf/2003.12083
    muv_lim = -17.75

    p_v = normal_cdf(dv, (6/5)*v_lim)
    p_lya = normal_cdf(f_lya, f_lya_lim)
    p_ha = normal_cdf(f_ha, f_ha_lim)
    p_w = normal_cdf(w_emerg, w_lim)
    p_muv = 1 - normal_cdf(10**muv, 6*(10**muv_lim))
    
    return p_lya * p_ha * p_w * p_muv * p_v

def p_f(_p_obs, f, fe):
    _f = np.sum(_p_obs) / len(_p_obs)
    _f_err = _f * np.sqrt((np.sqrt(len(_p_obs)) / len(_p_obs))**2 + (fe/f)**2)
    dist = np.exp(-0.5 * ((f - _f) / _f_err) ** 2)
    if np.isinf(dist):
        dist = 100
    # get Gaussian distance from expected observed fraction
    return dist


def multivar_normal_pdf(x, mu1, mu2, mu3, std1, std2, std3):
    """
    Multivariate normal probability density function.
    """
    coeff = 1 / (np.sqrt(2 * np.pi) ** 3 * std1 * std2 * std3)
    # coeff = 1
    exponent = -0.5 * ((x[0] - mu1) ** 2 / std1 ** 2 + 
                       (x[1] - mu2) ** 2 / std2 ** 2 + 
                       (x[2] - mu3) ** 2 / std3 ** 2)
    return coeff * np.exp(exponent)

XWIDE = np.array([np.log10(lum_lya_wide), dv_wide, np.log10(lum_ha_wide)])
XDEEP = np.array([np.log10(lum_lya_deep), dv_deep, np.log10(lum_ha_deep)])

os.makedirs('../data/pca', exist_ok=True)
XC = XDEEP.mean(axis=1, keepdims=True)
np.save('../data/pca/xc.npy', XC)
XSTD = XDEEP.std(axis=1, keepdims=True)
np.save('../data/pca/xstd.npy', XSTD)
XDEEP0 = (XDEEP - XC) / XSTD

# forward model sample from arbitrary basis and learn transformation which
# best fits the MUV-marginalized Tang data

def fit_transform():
    """
    Fit PCA to the data and return the transformation matrix.
    """
    muv_space = np.linspace(-20, -16, 1000)
    p_muv_space = p_muv(muv_space, phi_5, muv_star_5, alpha_5)
    p_muv_space /= np.sum(p_muv_space)  # normalize

    # fit straight line to Deltav vs Lya luminosity
    popt1, pcov1 = curve_fit(line, XDEEP0[0],  XDEEP0[1], p0=[0])
    # fit straight line to Lya vs H-alpha luminosity
    popt2, pcov2 = curve_fit(line, XDEEP0[0],  XDEEP0[2], p0=[0])
    # fit straight line to Deltav vs H-alpha luminosity
    popt3, pcov3 = curve_fit(line, XDEEP0[1], XDEEP0[2], p0=[0])

    I = np.array([[1,0,0],[0,1,0],[0,0,1]])
    A1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
    A2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
    A3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])

    def objective_function(params):
        """
        Objective function to minimize.
        """
        c1, c2, c3, c4 = params
        Y = np.random.normal(0, 1, (3, 1000))
        A = c1*I + c2*A1 + c3*A2 + c4*A3
        if np.linalg.det(A) == 0:
            return 1e10  # avoid singular matrix
        X = (A @ Y) * XSTD + XC
        # now apply selection effects
        lly, dv, lha = X
        # I also need to draw MUV from the relevant distribution
        muv_deep = np.random.choice(muv_space, size=1000, p=p_muv_space)
        _p_obs_deep = p_obs(10**lly, dv, 10**lha, muv_deep, mode='deep')

        lly = (lly - XC[0])/XSTD[0]
        dv = (dv - XC[1])/XSTD[1]
        lha = (lha - XC[2])/XSTD[2]

        lly_deep = lly[_p_obs_deep > 0.5]
        dv_deep = dv[_p_obs_deep > 0.5]
        lha_deep = lha[_p_obs_deep > 0.5]

        if len(lly_deep) < 100:
            return 1e10

        # fit line to deltav vs Lya luminosity
        popt1s, pcov1s = curve_fit(line, lly_deep, dv_deep, p0=[0])
        # fit line to Lya vs H-alpha luminosity
        popt2s, pcov2s = curve_fit(line, lly_deep, lha_deep, p0=[0])
        # fit line to Deltav vs H-alpha luminosity
        popt3s, pcov3s = curve_fit(line, dv_deep, lha_deep, p0=[0])
        err1 = popt1s*np.sqrt(pcov1s/popt1s**2 + pcov1/popt1**2)
        err2 = popt2s*np.sqrt(pcov2s/popt2s**2 + pcov2/popt2**2)
        err3 = popt3s*np.sqrt(pcov3s/popt3s**2 + pcov3/popt3**2)

        # calculate the log likelihood of the parameters
        logp_rho = -0.5*((popt1s-popt1)/err1)**2 - \
            0.5*((popt2s-popt2)/err2)**2 - 0.5*((popt3s-popt3)/err3)**2

        return -1*logp_rho.squeeze()
    
    bounds = [(-1, 1)] * 4  # coefficients for the linear combination of basis matrices
    result = differential_evolution(objective_function, bounds, maxiter=200, disp=True)
    print(f'Optimization result: {result}')
    c1, c2, c3, c4 = result.x
    print(f'Coefficients: {c1}, {c2}, {c3}, {c4}')
    np.save('../data/pca/coefficients.npy', result.x)
    A = c1*I + c2*A1 + c3*A2 + c4*A3
    np.save('../data/pca/A.npy', A)
    return A

A = fit_transform()
I = np.array([[1,0,0],[0,1,0],[0,0,1]])
A1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
A2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
A3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])
A = np.load('../data/pca/A.npy')

# NOTE we round the coefficients to simple fractions for easier interpretation
# c1, c2, c3, c4 = 1, 1, 1/3, -1/3
# A = c1*I + c2*A1 + c3*A2 + c4*A3

Y = np.random.normal(0, 1, (3, 1000))
X = (A @ Y) * XSTD + XC
lly, dv, lha = X
muv_space = np.linspace(-20, -16, 1000)
p_muv_space = p_muv(muv_space, phi_5, muv_star_5, alpha_5)
p_muv_space /= np.sum(p_muv_space)  # normalize
_muv_wide = np.random.choice(muv_space, size=1000, p=p_muv_space)
_muv_deep = np.random.choice(muv_space, size=1000, p=p_muv_space)
_p_obs_wide = p_obs(10**lly, dv, 10**lha, _muv_wide, mode='wide')
_p_obs_deep = p_obs(10**lly, dv, 10**lha, _muv_deep, mode='deep')

lly_wide = lly[_p_obs_wide > 0.5]
lly_deep = lly[_p_obs_deep > 0.5]
lha_wide = lha[_p_obs_wide > 0.5]
lha_deep = lha[_p_obs_deep > 0.5]
dv_wide = dv[_p_obs_wide > 0.5]
dv_deep = dv[_p_obs_deep > 0.5]

lly_space = np.linspace(38, 44, 100)
dv_space = np.linspace(1, 3, 100)
lha_space = np.linspace(38, 44, 100)

fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

axs[0].scatter(lly, dv, s=5, label='samples: all', color='cyan')
axs[0].scatter(lly_deep, dv_deep, s=5, label='samples: observed', color='lime')
axs[0].errorbar(np.log10(lum_lya), dv_lya, \
                xerr=lum_lya_err/lum_lya/np.log(10), \
                yerr=dv_lya_err, fmt='o', color='red', markersize=5, label='data')
axs[0].set_xlabel(r'$\log_{10}L_{\rm Ly\alpha}$ [erg/s]', fontsize=font_size)
axs[0].set_ylabel(r'$\Delta v$ [km/s]', fontsize=font_size)
axs[0].legend(fontsize=label_size)

axs[1].scatter(lly, lha, s=5, label='all', color='cyan')
axs[1].scatter(lly_deep, lha_deep, s=5, label='Deep sample', color='lime')
axs[1].errorbar(np.log10(lum_lya), np.log10(lum_ha), \
                xerr=lum_lya_err/lum_lya/np.log(10), \
                yerr=lum_ha_err/lum_ha/np.log(10), \
                fmt='o', color='red', markersize=5)
axs[1].set_xlabel(r'$\log_{10}L_{\rm Ly\alpha}$ [erg/s]', fontsize=font_size)
axs[1].set_ylabel(r'$\log_{10}L_{\rm H\alpha}$ [erg/s]', fontsize=font_size)

axs[2].scatter(dv, lha, s=5, label='all', color='cyan')
axs[2].scatter(dv_deep, lha_deep, s=5, label='Deep sample', color='lime')
axs[2].errorbar(dv_lya, np.log10(lum_ha), xerr=dv_lya_err, \
                yerr=lum_ha_err/lum_ha/np.log(10), \
                fmt='o', color='red', markersize=5)

axs[2].set_xlabel(r'$\Delta v$ [km/s]', fontsize=font_size)
axs[2].set_ylabel(r'$\log_{10}L_{\rm H\alpha}$ [erg/s]', fontsize=font_size)

plt.savefig('../out/basis.pdf', dpi=1000)