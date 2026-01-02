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

def get_beta_bouwens14(muv):
    # https://arxiv.org/pdf/1306.2950
    return -2.05 + -0.2*(muv+19.5)

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha+1)*\
        np.exp(-10**(0.4*(muv_star - muv)))

def get_a(m):
            return 0.65 + 0.1 * np.tanh(3 * (m + 20.75))

def get_wc(m):
    return 31 + 12 * np.tanh(4 * (m + 20.25))

def mason2018(Muv):
    """
    Samples EW and emission probability from the
    fit functions obtained by Mason et al. 2018.
    """
    A = get_a(Muv)
    rv_A = np.random.uniform(0, 1, len(Muv))
    emit_bool = rv_A < A
    Wc = get_wc(Muv[emit_bool])
    rv_W = np.random.uniform(0, 1, len(Wc))
    W = -1*Wc*np.log(rv_W)
    return W, emit_bool

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

def get_silverrush_laelf(z):
    if z==4.9:
        # SILVERRUSH XIV z=4.9 LAELF
        lum_silver = np.array([42.75, 42.85, 42.95, 43.05, 43.15, 43.25, 43.35, 43.45, 43.55, 43.65])
        logphi_silver = -1*np.array([2.91, 3.17, 3.42, 3.78, 3.88, 4.00, 4.75, 4.93, 5.23, 4.93])
        logphi_up_silver = 1e-2*np.array([5, 5, 6, 9, 10, 12, 29, 36, 52, 36])
        logphi_low_silver = 1e-2*np.array([5, 5, 6, 9, 10, 12, 34, 45, 77, 45])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==5.7:
        # SILVERRUSH XIV z=5.7 LAELF
        lum_silver = np.array([42.85, 42.95, 43.05, 43.15, 43.25, 43.35, 43.45, 43.55, 43.65, 43.75, 43.85, 43.95])
        logphi_silver = -1*np.array([3.05, 3.27, 3.56, 3.85, 4.15, 4.41, 4.72, 5.15, 5.43, 6.03, 6.33, 6.33])
        logphi_up_silver = 1e-2*np.array([4, 2, 2, 3, 4, 5, 7, 12, 17, 36, 52, 52])
        logphi_low_silver = 1e-2*np.array([4, 2, 2, 3, 4, 5, 7, 13, 18, 45, 77, 77])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==6.6:
        # SILVERRUSH XIV z=6.6 LAELF
        lum_silver = np.array([42.95, 43.05, 43.15, 43.25, 43.35, 43.45, 43.55, 43.65, 43.75, 43.95, 44.05])
        logphi_silver = -1*np.array([3.71, 4.11, 4.37, 4.65, 4.83, 5.28, 5.89, 5.9, 5.9, 6.38, 6.38])
        logphi_up_silver = 1e-2*np.array([9, 5, 6, 7, 8, 14, 29, 29, 29, 52, 52])
        logphi_low_silver = 1e-2*np.array([9, 5, 6, 7, 8, 15, 34, 34, 34, 77, 77])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==7.0:
        # wip
        # SILVERRUSH XIV z=7.0 LAELF
        lum_silver = np.array([43.25, 43.35])
        logphi_silver = -1*np.array([4.4, 4.95])
        logphi_up_silver = 1e-2*np.array([29, 52])
        logphi_low_silver = 1e-2*np.array([34, 77])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==7.3:
        # wip
        # SILVERRUSH XIV z=7.3 LAELF
        lum_silver = np.array([43.45])
        logphi_silver = -1*np.array([4.81])
        logphi_up_silver = 1e-2*np.array([36])
        logphi_low_silver = 1e-2*np.array([45])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    
lum, logphi, logphi_up, logphi_low = get_silverrush_laelf(4.9)
bin_edges = np.zeros(len(lum) + 1)
bin_edges[0] = lum[0] - 0.5*(lum[1] - lum[0])
bin_edges[1:-1] = 0.5*(lum[1:] + lum[:-1])
bin_edges[-1] = lum[-1] + 0.5*(lum[-1] - lum[-2])

muv_t24, muv_emu, muv_dex = np.array([[-19.5, -18.5, -17.5], \
                                      [10, 16, 27], [1.75, 1.51, 0.99]])
def mean_t24(muv):
    mean = np.zeros_like(muv)
    mean[muv<=-19.5] = muv_emu[0]
    mean[muv>-17.5] = muv_emu[2]
    mean[mean==0] = muv_emu[1]
    return mean

def sigma_t24(muv):
    sigma = np.zeros_like(muv)
    sigma[muv<=-19.5] = muv_dex[0]
    sigma[muv>-17.5] = muv_dex[2]
    sigma[sigma==0] = muv_dex[1]
    return sigma/np.log(10)

from scipy.optimize import curve_fit
emu_popt, _ = curve_fit(lambda x, a, b: a * x + b, muv_t24, muv_emu)
dex_popt, _ = curve_fit(lambda x, a, b: a * x + b, muv_t24, muv_dex)

NSAMPLES = 1000000

# Generate a sample of Muv values
# my EW PDF is highly, highly sensitive to the limiting MUV

# TOGGLE HERE

ewpdf = False  # Set to True to compute the EW PDF
muv_space = np.linspace(-24, -16, NSAMPLES)
# muv_space = np.linspace(-24, -18, NSAMPLES)
# ewpdf = True
# compare = False
p_muv = schechter(muv_space, phi_5, muv_star_5, alpha_5)
n_gal = np.trapezoid(p_muv, x=muv_space)*1e-3 # galaxy number density in Mpc^-3
EFFECTIVE_VOLUME = NSAMPLES/n_gal  # Mpc3, for normalization

p_muv /= np.sum(p_muv)
muv_sample = np.random.choice(muv_space, size=NSAMPLES, p=p_muv)

# Mason et al. (2018) model
w_m18, emit_bool_m18 = mason2018(muv_sample)
f10_m18 = np.sum(w_m18 > 10) / NSAMPLES
f25_m18 = np.sum(w_m18 > 25) / NSAMPLES
log10lya_m18 = np.log10(w_m18*(2.47e15/1215.67)*(1500/1215.67)**(get_beta_bouwens14(muv_sample[emit_bool_m18])+2)*\
    10**(0.4*(51.6-muv_sample[emit_bool_m18])))
heights_m18, bins_m18 = np.histogram(log10lya_m18, bins=bin_edges, density=False)
bin_widths = bins_m18[1:]-bins_m18[:-1]
height_err_m18 = np.sqrt(heights_m18) / bin_widths / EFFECTIVE_VOLUME
heights_m18 = heights_m18 / bin_widths / EFFECTIVE_VOLUME
logphi_m18 = np.log10(heights_m18)
logphi_up_m18 = np.abs(np.log10(height_err_m18 + heights_m18) - logphi_m18)
logphi_low_m18 = np.abs(logphi_m18 - np.log10(heights_m18 - height_err_m18))

# Tang et al. (2024) model
w_t24 = np.random.lognormal(mean=np.log(mean_t24(muv_sample)),
                            sigma=sigma_t24(muv_sample),
                            size=NSAMPLES)
f10_t24 = np.sum(w_t24 > 10) / NSAMPLES
f25_t24 = np.sum(w_t24 > 25) / NSAMPLES
log10lya_t24 = np.log10(w_t24*(2.47e15/1215.67)*(1215.67/1500)**(get_beta_bouwens14(muv_sample)+2)*\
    10**(0.4*(51.6-muv_sample)))
heights_t24, bins_t24 = np.histogram(log10lya_t24, bins=bin_edges, density=False)
bin_widths = bins_t24[1:]-bins_t24[:-1]
height_err_t24 = np.sqrt(heights_t24) / bin_widths / EFFECTIVE_VOLUME
heights_t24 = heights_t24 / bin_widths / EFFECTIVE_VOLUME
logphi_t24 = np.log10(heights_t24)
logphi_up_t24 = np.abs(np.log10(height_err_t24 + heights_t24) - logphi_t24)
logphi_low_t24 = np.abs(logphi_t24 - np.log10(heights_t24 - height_err_t24))

# Gagnon-Hartman et al. (2025) model

I = np.array([[1,0,0],[0,1,0],[0,0,1]])
A1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
A2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
A3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])
c1, c2, c3, c4 = 1, 1, 1/3, -1
A = c1 * I + c2 * A1 + c3 * A2 + c4 * A3
xc = np.load('../data/pca/xc.npy')
xstd = np.load('../data/pca/xstd.npy')

# m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load('../data/pca/diag_fit_params.npy')
# m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load('../data/pca/fobs_fit_params.npy')
# m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load('../data/pca/fesc_fit_params.npy')

# m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load('../data/pca/fit_params.npy')
# m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load('../data/pca/wide_fit_params.npy')
w_shg_list = []
for fit in ['', 'deep_', 'wide_']:
    m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load(f'../data/pca/{fit}fit_params.npy')

    u1, u2, u3 = np.random.normal(m1*(muv_sample + 18.5) + b1, std1, NSAMPLES), \
                np.random.normal(m2*(muv_sample + 18.5) + b2, std2, NSAMPLES), \
                np.random.normal(m3*(muv_sample + 18.5) + b3, std3, NSAMPLES)
    log10lya, dv, log10ha = (A @ np.array([u1, u2, u3]))* xstd + xc

    # apply dv selection
    v_lim = 10**vcirc(muv_sample)
    select = dv > v_lim
    log10lya = log10lya[select]
    muv_sample_ = muv_sample[select]

    w_sgh = (1215.67/2.47e15)*(10**log10lya)*10**(-0.4*(51.6-muv_sample_))*\
        (1215.67/1500)**(-1*get_beta_bouwens14(muv_sample_)-2)
    w_shg_list.append(w_sgh)

if ewpdf == True:
    if compare == True:
        # EW PDF
        h40, b40 = np.histogram(w_sgh[(w_sgh>40)*(w_sgh<200)], bins=100, density=True)
        b40_c = 0.5*(b40[1:] + b40[:-1])
        h1000, b1000 = np.histogram(w_sgh[(w_sgh>40)*(w_sgh<1000)], bins=100, density=True)
        b1000_c = 0.5*(b1000[1:] + b1000[:-1])
        p40, _ = curve_fit(lambda x, a, b: a * x + b, b40_c, np.log10(h40))
        p1000, _ = curve_fit(lambda x, a, b: a * x + b, b1000_c[h1000>0], np.log10(h1000[h1000>0]))
        log10w40_fit = p40[0] * b40_c + p40[1]
        log10w1000_fit = p1000[0] * b1000_c + p1000[1]

        def umeda_ewpdf(w, a, b):
            p_w = np.zeros_like(w)
            leg1 = np.exp(-1*w[w<=200] / 32.9)
            leg2 = np.exp(-1*w[w>200] / 76.3)
            p_w[w<=200] = a*leg1
            p_w[w>200] = (a*leg1[-1]/leg2[0])*leg2
            return p_w

        p_u, _ = curve_fit(umeda_ewpdf, b1000_c[h1000>0], h1000[h1000>0])

        p_w = umeda_ewpdf(b1000_c, p_u[0], p_u[1])

        # bins = np.linspace(40, 1000, 101)
        bins = np.logspace(np.log10(40), np.log10(1000), 101)
        p_w_res = umeda_ewpdf(0.5*(bins[1:]+bins[:-1]), p_u[0], p_u[1])

        fig, axs = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        axs[0].plot(b1000_c, p_w, color=color4, linewidth=3, label='Umeda+25')
        axs[0].hist(w_m18[(w_m18>40)*(w_m18<1000)], bins=bins, linewidth=2.0, density=True, histtype='step', color=color3, label='Mason+18')
        heights, bin_edges = np.histogram(w_m18[(w_m18>40)*(w_m18<1000)], bins=bins, density=True)
        
        # residuals
        axs[1].axhline(0, color='orange', linestyle='-', linewidth=3)
        axs[1].plot(0.5*(bins[1:]+bins[:-1]), heights/p_w_res - 1, color=color3, label='Mason+18')
        
        labels = ['Deep + Wide', 'Deep-only', 'Wide-only']
        linestyles = ['-', '--', ':']
        for i, w_sgh in enumerate(w_shg_list):
            axs[0].hist(w_sgh[(w_sgh>40)*(w_sgh<1000)], bins=bins, linewidth=2.0, density=True, histtype='step', color=color1, linestyle=linestyles[i], label=labels[i])
            heights, bin_edges = np.histogram(w_sgh[(w_sgh>40)*(w_sgh<1000)], bins=bins, density=True)
            axs[1].plot(0.5*(bins[1:]+bins[:-1]), heights/p_w_res - 1, color=color1, linestyle=linestyles[i], label=labels[i])

        axs[1].set_xlabel(r'$\rm W_{\rm emerg}^{\rm Ly\alpha}$ [$\AA$]', fontsize=font_size)
        axs[0].set_ylabel(r'$\rm P(W_{\rm emerg}^{\rm Ly\alpha})$', fontsize=font_size)
        axs[0].legend(fontsize=int(font_size/1.5), loc='upper right')
        axs[0].set_yscale('log')
        axs[0].set_xlim(50, 650)
        axs[0].set_ylim(1e-6, 1e-1)
        axs[1].set_ylim(-1, 1)
        axs[1].set_ylabel(r'Residuals', fontsize=font_size)
        axs[0].set_xscale('log')
        figdir = '/mnt/c/Users/sgagn/OneDrive/Documents/phd/lyman_alpha/figures/'
        plt.savefig(f'{figdir}/ew_residuals_pdf.pdf', bbox_inches='tight')
        plt.show()
        quit()
    else:
        
        bins = np.linspace(40, 1000, 101)
        
        h40, b40 = np.histogram(w_sgh[(w_sgh>40)*(w_sgh<200)], bins=100, density=True)
        b40_c = 0.5*(b40[1:] + b40[:-1])
        h1000, b1000 = np.histogram(w_sgh[(w_sgh>40)*(w_sgh<1000)], bins=100, density=True)
        b1000_c = 0.5*(b1000[1:] + b1000[:-1])
        p40, _ = curve_fit(lambda x, a, b: a * x + b, b40_c, np.log10(h40))
        p1000, _ = curve_fit(lambda x, a, b: a * x + b, b1000_c[h1000>0], np.log10(h1000[h1000>0]))
        log10w40_fit = p40[0] * b40_c + p40[1]
        log10w1000_fit = p1000[0] * b1000_c + p1000[1]

        def umeda_ewpdf(w, a, b):
            p_w = np.zeros_like(w)
            leg1 = np.exp(-1*w[w<=200] / 32.9)
            leg2 = np.exp(-1*w[w>200] / 76.3)
            p_w[w<=200] = a*leg1
            p_w[w>200] = (a*leg1[-1]/leg2[0])*leg2
            return p_w

        p_u, _ = curve_fit(umeda_ewpdf, b1000_c[h1000>0], h1000[h1000>0])

        p_w = umeda_ewpdf(b1000_c, p_u[0], p_u[1])

        # bins = np.logspace(np.log10(40), np.log10(1000), 101)
        p_w_res = umeda_ewpdf(0.5*(bins[1:]+bins[:-1]), p_u[0], p_u[1])

        fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
        
        ax.plot(b1000_c, p_w, color=color4, linewidth=3, label='Umeda+25')
        ax.hist(w_m18[(w_m18>40)*(w_m18<1000)], bins=bins, linewidth=2.0, density=True, histtype='step', color=color3, label='Mason+18')
        heights, bin_edges = np.histogram(w_m18[(w_m18>40)*(w_m18<1000)], bins=bins, density=True)
         
        w_sgh = w_shg_list[0]
        ax.hist(w_sgh[(w_sgh>40)*(w_sgh<1000)], bins=bins, linewidth=2.0, density=True, histtype='step', color=color1, linestyle='-', label='This Work')
        
        ax.set_xlabel(r'$\rm W_{\rm emerg}^{\rm Ly\alpha}$ [$\AA$]', fontsize=font_size)
        ax.set_ylabel(r'$\rm P(W_{\rm emerg}^{\rm Ly\alpha})$', fontsize=font_size)
        ax.legend(fontsize=int(font_size/1.5), loc='upper right')
        ax.set_yscale('log')
        ax.set_xlim(50, 650)
        ax.set_ylim(1e-6, 1e-1)
        # ax.set_xscale('log')
        figdir = '/mnt/c/Users/sgagn/OneDrive/Documents/phd/lyman_alpha/figures/'
        plt.savefig(f'{figdir}/ewpdf.pdf', bbox_inches='tight')
        plt.show()
        quit()

fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

heights_sgh, bins_sgh = np.histogram(log10lya, bins=bin_edges, density=False)
bin_widths = bins_sgh[1:]-bins_sgh[:-1]
height_err_sgh = np.sqrt(heights_sgh) / bin_widths / EFFECTIVE_VOLUME
heights_sgh = heights_sgh / bin_widths / EFFECTIVE_VOLUME
logphi_sgh = np.log10(heights_sgh)
logphi_up_sgh = np.abs(np.log10(heights_sgh + height_err_sgh) - logphi_sgh)
logphi_low_sgh = np.abs(logphi_sgh - np.log10(heights_sgh - height_err_sgh))

ax.errorbar(lum, logphi_sgh, yerr=[logphi_low_sgh, logphi_up_sgh],
            fmt='*', markeredgewidth=2, markersize=20, fillstyle='none', color=color1, label='This Work')

# log10lya = np.load('../data/log10lya_z5.7_16092025.npy')

# heights_sgh, bins_sgh = np.histogram(log10lya, bins=bin_edges, density=False)
# bin_widths = bins_sgh[1:]-bins_sgh[:-1]
# height_err_sgh = np.sqrt(heights_sgh) / bin_widths / 300**3
# heights_sgh = heights_sgh / bin_widths / 300**3
# logphi_sgh = np.log10(heights_sgh)
# logphi_up_sgh = np.abs(np.log10(heights_sgh + height_err_sgh) - logphi_sgh)
# logphi_low_sgh = np.abs(logphi_sgh - np.log10(heights_sgh - height_err_sgh))

# ax.errorbar(lum, logphi_sgh, yerr=[logphi_low_sgh, logphi_up_sgh],
#             fmt='^', markeredgewidth=2, markersize=20, fillstyle='none', color=color1, label='21cmFASTv4')

ax.errorbar(lum, logphi, yerr=[logphi_low, logphi_up], 
            fmt='o', markeredgewidth=2, markersize=20, fillstyle='none', color=color4, label='Umeda+25')
ax.errorbar(lum, logphi_m18, yerr=[logphi_low_m18, logphi_up_m18],
            fmt='*', markeredgewidth=2, markersize=20, fillstyle='none', color=color3, label='Mason+18')
# ax.errorbar(lum, logphi_t24, yerr=[logphi_low_t24, logphi_up_t24],
#             fmt='*', markeredgewidth=2, markersize=20, fillstyle='none', color=color2, label='Tang+24')
ax.set_xlabel(r'$\log_{10} L_{\rm Ly\alpha}$ [erg s$^{-1}$]', fontsize=font_size)
ax.set_ylabel(r'$\log_{10} \phi$ [Mpc$^{-3}$]', fontsize=font_size)
ax.set_ylim(-8, -2)
ax.legend(fontsize=int(font_size/1.5), loc='lower left')
figdir = '/mnt/c/Users/sgagn/OneDrive/Documents/phd/lyman_alpha/figures/'
plt.savefig(f'{figdir}/laelf.pdf', bbox_inches='tight')
plt.show()
