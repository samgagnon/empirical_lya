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

def vcirc(muv):
    """
    Returns circular velocity in km/s as a function of MUV 
    at redshift 5.0.
    """
    log10_mh = mh_from_muv(muv)
    return (log10_mh - 5.62)/3

def t_cgm(dv, muv):
    """
    CGM transmission as a function of velocity offset and MUV.
    """
    v_circ = vcirc(muv)
    tcgm = 0.5 * (1 - erf(1.25*((10**v_circ - dv)/(dv + 34))))
    return tcgm

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

muv_space = np.linspace(-24, -18.25, NSAMPLES)
p_muv = schechter(muv_space, phi_5, muv_star_5, alpha_5)
n_gal = np.trapezoid(p_muv, x=muv_space)*1e-3 # galaxy number density in Mpc^-3
EFFECTIVE_VOLUME = NSAMPLES/n_gal  # Mpc3, for normalization

p_muv /= np.sum(p_muv)
muv_sample = np.random.choice(muv_space, size=NSAMPLES, p=p_muv)

# Mason et al. (2018) model
w_m18, emit_bool_m18 = mason2018(muv_sample)
print(np.percentile(w_m18, 84), np.percentile(w_m18, 16), np.median(w_m18)/np.log(2))

# Gagnon-Hartman et al. (2025) model
loaddir = '../data/pca'
# loaddir = '../data/cgm'
I = np.array([[1,0,0],[0,1,0],[0,0,1]])
A1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
A2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
A3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])
c1, c2, c3, c4 = 1, 1, 1/3, -1
A = c1 * I + c2 * A1 + c3 * A2 + c4 * A3
xc = np.load(f'{loaddir}/xc.npy')
xstd = np.load(f'{loaddir}/xstd.npy')

m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load(f'{loaddir}/fit_params.npy')

# m_arr = np.array([m1, m2, m3])
# b_arr = np.array([b1, b2, b3])
# std_arr = np.array([std1, std2, std3])
# covariance_matrix = A@np.diag(std_arr**2)@A.T
# print(A@m_arr, A@b_arr)

# print(xc, xstd)

# print("Covariance matrix of the PCA components:")
# print(covariance_matrix)
# quit()

u1, u2, u3 = np.random.normal(m1*(muv_sample + 18.5) + b1, std1, NSAMPLES), \
            np.random.normal(m2*(muv_sample + 18.5) + b2, std2, NSAMPLES), \
            np.random.normal(m3*(muv_sample + 18.5) + b3, std3, NSAMPLES)
log10lya, dv, log10ha = (A @ np.array([u1, u2, u3]))* xstd + xc
w_sgh = 10**(log10lya) / ( (2.47e15/1215.67)*(1215.67/1500)**(get_beta_bouwens14(muv_sample)+2)*\
    10**(0.4*(51.6-muv_sample)) )

# solve for lognormal distribution which fits my model
# bins = np.linspace(0, 1000, 101)
# hs, bs = np.histogram(w_sgh, bins=bins, density=True)
# def objective(params):
#     mean, scale = params
#     samples = np.exp(np.random.normal(loc=np.log(mean), scale=scale, size=NSAMPLES))
#     h, b = np.histogram(samples, bins=bins, density=True)
#     return np.sum(hs*np.log((hs+1e-10)/(h+1e-10)))

# from scipy.optimize import differential_evolution
# result = differential_evolution(objective, bounds=[(20, 100), (0.1, 1.0)], maxiter=50, disp=True)
# mean_sgh, scale_sgh = result.x
# w_sgh_lognorm = np.exp(np.random.normal(loc=np.log(mean_sgh), scale=scale_sgh, size=NSAMPLES))

# print("SGH model lognormal fit parameters:")
# print("Mean:", mean_sgh)
# print("Scale (sigma):", scale_sgh)

# plt.hist(w_sgh, bins=bins, density=True, histtype='step', label='SGH model')
# plt.hist(w_sgh_lognorm, bins=bins, density=True, histtype='step', label='SGH lognormal fit')
# plt.legend()
# plt.show()
# quit()

# Apply CGM transmission
# tcgm = t_cgm(dv, muv_sample)
# log10lya += np.log10(tcgm)

# apply dv selection
v_lim = 10**vcirc(muv_sample)
select = dv > v_lim
log10lya = log10lya[select]
        
bins = np.linspace(40, 1000, 101)
# bins = np.linspace(0, 1000, 101)

# tang+24 lognorm fit
muv_c = np.array([-19.5, -18.5, -17.5])[:1]
emu_t24_c = np.array([10, 16, 27])[:1]
dex_t24_c = np.array([1.48, 1.75, 1.51])[:1]
weights = schechter(muv_c, phi_5, muv_star_5, alpha_5)
weights /= np.sum(weights)

emu_t24 = np.sum(emu_t24_c * weights)
dex_t24 = np.sum(dex_t24_c * weights)

w_t24 = np.exp(np.random.normal(loc=np.log(emu_t24), scale=dex_t24, size=NSAMPLES))

w_z14 = np.random.exponential(scale=7.3*(1 + 5)**1.7, size=NSAMPLES)

h40, b40 = np.histogram(w_sgh[(w_sgh>40)*(w_sgh<200)], bins=100, density=True)
b40_c = 0.5*(b40[1:] + b40[:-1])
h1000, b1000 = np.histogram(w_sgh[(w_sgh>40)*(w_sgh<1000)], bins=100, density=True)
b1000_c = 0.5*(b1000[1:] + b1000[:-1])
p40, _ = curve_fit(lambda x, a, b: a * x + b, b40_c, np.log10(h40))
p1000, _ = curve_fit(lambda x, a, b: a * x + b, b1000_c[h1000>0], np.log10(h1000[h1000>0]))
log10w40_fit = p40[0] * b40_c + p40[1]
log10w1000_fit = p1000[0] * b1000_c + p1000[1]

def umeda_ewpdf(w, a):
    p_w = np.zeros_like(w)
    leg1 = np.exp(-1*w[w<=200] / 32.9)
    leg2 = np.exp(-1*w[w>200] / 76.3)
    p_w[w<=200] = a*leg1
    p_w[w>200] = (a*leg1[-1]/leg2[0])*leg2
    return p_w

p_u, _ = curve_fit(umeda_ewpdf, b1000_c[h1000>0], h1000[h1000>0])

p_w = umeda_ewpdf(b1000_c, p_u[0])

p_w_res = umeda_ewpdf(0.5*(bins[1:]+bins[:-1]), p_u[0])

fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)

ax.plot(b1000_c, p_w, color=color4, linewidth=3, label='Umeda+25')
ax.hist(w_m18[(w_m18>40)*(w_m18<1000)], bins=bins, linewidth=2.0, density=True, histtype='step', color=color3, label='Mason+18')
ax.hist(w_t24[(w_t24>40)*(w_t24<1000)], bins=bins, linewidth=2.0, density=True, histtype='step', color=color2, label='Tang+24')
# ax.hist(w_z14[(w_z14>40)*(w_z14<1000)], bins=bins, linewidth=2.0, density=True, histtype='step', color='magenta', label='Zheng+14')

# heights, bin_edges = np.histogram(w_m18[(w_m18>40)*(w_m18<1000)], bins=bins, density=True)

ax.hist(w_sgh[(w_sgh>40)*(w_sgh<1000)], bins=bins, linewidth=2.0, density=True, histtype='step', color=color1, linestyle='-', label='This Work')

ax.set_xlabel(r'$\rm W_{\rm emerg}^{\rm Ly\alpha}$ [$\AA$]', fontsize=font_size)
ax.set_ylabel(r'$\rm P(W_{\rm emerg}^{\rm Ly\alpha})$', fontsize=font_size)
ax.legend(fontsize=int(font_size/1.5), loc='upper right')
ax.set_yscale('log')
ax.set_xlim(40, 650)
ax.set_ylim(1e-6, 1e-1)
figdir = '../out/'
plt.savefig(f'{figdir}/ewpdf.pdf', bbox_inches='tight')
plt.show()