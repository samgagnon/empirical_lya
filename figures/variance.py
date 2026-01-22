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
import matplotlib as mpl
label_size = 20
font_size = 30
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size
# helper for inset colorbars
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# presentation = True
presentation = False
if presentation:
    plt.style.use('dark_background')
    cmap = 'Blues_r'
else:
    cmap = 'hot_r'

# from Bouwens 2021 https://arxiv.org/pdf/2102.07775
phi_5 = 0.79
muv_star_5 = -21.1
alpha_5 = -1.74

lum_flux_factor = 4*np.pi*(Planck18.luminosity_distance(5.0).to('cm').value)**2

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

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha+1)*\
        np.exp(-10**(0.4*(muv_star - muv)))

def line(x, m, b):
    """
    Linear function.
    """
    return m * (x + 18.5) + b

# loaddir = '../data/pca'
loaddir = '../data/cgm'
# A = np.load(f'{loaddir}/A.npy')
I = np.array([[1,0,0],[0,1,0],[0,0,1]])
A1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
A2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
A3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])
c1, c2, c3, c4 = 1, 1, 1/3, -1
# c1, c2, c3, c4 = np.load(f'{loaddir}/coefficients.npy')
A = c1 * I + c2 * A1 + c3 * A2 + c4 * A3
xc = np.load(f'{loaddir}/xc.npy')
xstd = np.load(f'{loaddir}/xstd.npy')
m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load(f'{loaddir}/fit_params.npy')
# m1, m2, m3, b1, b2, b3, std1, std2, std3 = -0.05042401, -0.52413956, -0.36336859, \
#     -0.91255108, -0.73920641, -0.4175795, 0.85332351, 0.45477191, 0.31269265
theta = [w1, w2, f1, f2, fh]
NSAMPLES = 100000
muv_space = np.linspace(-24, -16, NSAMPLES)
# p_muv = schechter(muv_space, phi_5, muv_star_5, alpha_5)
# p_muv /= np.sum(p_muv)  # Normalize the probability distribution
# muv_sample = np.random.choice(muv_space, p=p_muv, size=NSAMPLES)
muv_sample = np.random.uniform(-24, -16, NSAMPLES)
muv_space = muv_sample
hist_res = 50
muv_side = np.linspace(-24, -16, hist_res)
lya_side = np.linspace(40, 45, hist_res)
lha_side = np.linspace(40, 45, hist_res)
dv_side = np.linspace(-250, 600, hist_res)

mu1 = line(muv_space, m1, b1)
mu2 = line(muv_space, m2, b2)
mu3 = line(muv_space, m3, b3)
y1 = np.random.normal(mu1, std1, NSAMPLES)
y2 = np.random.normal(mu2, std2, NSAMPLES)
y3 = np.random.normal(mu3, std3, NSAMPLES)
Y = np.vstack((y1, y2, y3))
X = (A @ Y) * xstd + xc

# Apply CGM transmission
tcgm = t_cgm(X[1], muv_sample)
X[0] += np.log10(tcgm)

# X[1] = 10**X[1]  # dv in km/s

fig, axs = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)

axs[0,0].hist2d(muv_space, X[0], bins=(muv_side, lya_side),
                            cmap=cmap, cmin=1, rasterized=True)
# axs[0,0].scatter(muv_space, X[0], s=1, color='cyan')
axs[0,0].set_ylabel(r'$\log_{10}L_{\rm Ly\alpha}$ [erg/s]', fontsize=font_size)
axs[0,0].set_xticklabels([])

axs[0,1].hist2d(X[1], X[0], bins=(dv_side, lya_side),
                            cmap=cmap, cmin=1, rasterized=True)
# axs[0,1].scatter(X[1], X[0], s=1, color='cyan')
axs[0,1].set_yticklabels([])
axs[0,1].set_xticklabels([])

h = axs[0,2].hist2d(X[2], X[0], bins=(lha_side, lya_side),
                            cmap=cmap, cmin=1, rasterized=False)
# ensure layout is drawn so axis positions are final
fig.canvas.draw()
# place a small colorbar inside the top-right axes using its bounding box
bbox = axs[0,2].get_position()
left = bbox.x0 + bbox.width*0.88
bottom = bbox.y0 + bbox.height*0.10
width = 0.015  # make colorbar much thinner
height = bbox.height*0.80
cax = fig.add_axes([left, bottom, width, height])
cbar = fig.colorbar(h[3], cax=cax, orientation='vertical')
cbar.set_label('probability density [arb. units]', fontsize=10, labelpad=4)
cax.yaxis.set_ticks_position('left')
# replace numeric tick labels with qualitative labels at min and max
try:
    vmin, vmax = cbar.mappable.get_clim()
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(['low', 'high'])
except Exception:
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['low', 'high'])
cbar.ax.tick_params(labelsize=9)

axs[0,2].set_yticklabels([])
axs[0,2].set_xticklabels([])

axs[1,0].hist2d(muv_space, X[1], bins=(muv_side, dv_side),
                            cmap=cmap, cmin=1, rasterized=True)
# axs[1,0].scatter(muv_space, X[1], s=1, color='cyan')
axs[1,0].set_ylabel(r'$\Delta v$ [km/s]', fontsize=font_size)
axs[1,0].set_xticklabels([])

axs[1,1].hist2d(X[1], X[1], bins=(dv_side, dv_side),
                            cmap=cmap, cmin=1, rasterized=True)
# axs[1,1].scatter(X[1], X[1], s=1, color='cyan')
axs[1,1].set_yticklabels([])
axs[1,1].set_xticklabels([])

axs[1,2].hist2d(X[2], X[1], bins=(lha_side, dv_side),
                            cmap=cmap, cmin=1, rasterized=True)
# axs[1,2].scatter(X[2], X[1], s=1, color='cyan')
axs[1,2].set_yticklabels([])
axs[1,2].set_xticklabels([])

axs[2,0].hist2d(muv_space, X[2], bins=(muv_side, lha_side),
                            cmap=cmap, cmin=1, rasterized=True)
# axs[2,0].scatter(muv_space, X[2], s=1, color='cyan')
axs[2,0].set_ylabel(r'$\log_{10}L_{\rm H\alpha}$ [erg/s]', fontsize=font_size)
axs[2,0].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)

axs[2,1].hist2d(X[1], X[2], bins=(dv_side, lha_side),
                            cmap=cmap, cmin=1, rasterized=True)
# axs[2,1].scatter(X[1], X[2], s=1, color='cyan')
axs[2,1].set_yticklabels([])
axs[2,1].set_xlabel(r'$\Delta v$ [km/s]', fontsize=font_size)

axs[2,2].hist2d(X[2], X[2], bins=(lha_side, lha_side),
                            cmap=cmap, cmin=1, rasterized=True)
# axs[2,2].scatter(X[2], X[2], s=1, color='cyan')
axs[2,2].set_yticklabels([])
axs[2,2].set_xlabel(r'$\log_{10}L_{\rm H\alpha}$ [erg/s]', fontsize=font_size)

figures_dir = '../out/'
plt.savefig(f'{figures_dir}/prop_var.pdf', bbox_inches='tight')
plt.show()