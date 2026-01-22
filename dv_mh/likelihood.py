"""
Uses the PCA basis and treats the evolution of mu, sigma 
for each basis vector as a linear function of Muv. Optimizes 
the parameters of each linear relation by fitting the observed 
fraction of LyA+Ha emitters quantity, as well as the means 
of each conditional distribution of LyA luminosity, 
velocity offset, H-alpha luminosity, and escape fraction.
"""

import os

import numpy as np

from scipy.integrate import trapezoid
from scipy.special import gamma, erf
from scipy.optimize import differential_evolution

from astropy.cosmology import Planck18

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

def p_muv(muv, phi, muv_star, alpha):
    return schechter(muv, phi, muv_star, alpha)/gamma(alpha_5+2)/(0.4*np.log(10))/phi_5

# measured lya properties from https://arxiv.org/pdf/2402.06070
MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, fescA, \
    fescA_err, fescB, fescB_err, ID = np.load('../data/tang24.npy').T

beta = get_beta_bouwens14(MUV)
lum_lya = (ew_lya/1215.67) * 2.47e15 * 10**(0.4*(51.6 - MUV)) * (1215.6/1500) ** (beta + 2)
lum_lya_err = lum_lya*np.sqrt((ew_lya_err/ew_lya)**2 + (np.exp(-0.4*np.log(10)*MUV_err)/MUV)**2)

lum_ha = lum_lya / 11.4 / fescA
lum_ha_err = lum_ha*np.sqrt((lum_lya_err/lum_lya)**2 + (fescA_err/fescA)**2)

lum_flux_factor = 4*np.pi*(Planck18.luminosity_distance(5.0).to('cm').value)**2

wide = ID==0
_muv_wide = MUV[wide]
_muv_err_wide = MUV_err[wide]
_dv_lya_wide = dv_lya[wide]
_dv_lya_err_wide = dv_lya_err[wide]
_lum_lya_wide = lum_lya[wide]
_lum_lya_err_wide = lum_lya_err[wide]
_lum_ha_wide = lum_ha[wide]
_lum_ha_err_wide = lum_ha_err[wide]
_fesc_wide = fescA[wide]
_fesc_err_wide = fescA_err[wide]
_ew_wide = ew_lya[wide]
_ew_err_wide = ew_lya_err[wide]

deep = ID==1
_muv_deep = MUV[deep]
_muv_err_deep = MUV_err[deep]
_dv_lya_deep = dv_lya[deep]
_dv_lya_err_deep = dv_lya_err[deep]
_lum_lya_deep = lum_lya[deep]
_lum_lya_err_deep = lum_lya_err[deep]
_lum_ha_deep = lum_ha[deep]
_lum_ha_err_deep = lum_ha_err[deep]
_fesc_deep = fescA[deep]
_fesc_err_deep = fescA_err[deep]
_ew_deep = ew_lya[deep]
_ew_err_deep = ew_lya_err[deep]

muv_wide = np.zeros(len(_muv_wide)*1000)
lum_lya_wide = np.zeros_like(muv_wide)
dv_wide = np.zeros_like(muv_wide)
lum_ha_wide = np.zeros_like(muv_wide)
fesc_wide = np.zeros_like(muv_wide)
ew_wide = np.zeros_like(muv_wide)

muv_deep = np.zeros(len(_muv_deep)*1000)
lum_lya_deep = np.zeros_like(muv_deep)
dv_deep = np.zeros_like(muv_deep)
lum_ha_deep = np.zeros_like(muv_deep)
fesc_deep = np.zeros_like(muv_deep)
ew_deep = np.zeros_like(muv_deep)

for i, (muv, muve, ew, ewe, lly, llye, dv, dve, lha, lhae, fe, fee) in enumerate(zip(_muv_wide, _muv_err_wide, \
                                                 _ew_wide, _ew_err_wide, _lum_lya_wide, _lum_lya_err_wide, \
                                                 _dv_lya_wide, _dv_lya_err_wide, _lum_ha_wide, _lum_ha_err_wide, \
                                                 _fesc_wide, _fesc_err_wide)):
    muv_wide[1000*i:1000*(i+1)] = np.random.normal(muv, muve, 1000)
    lum_lya_wide[1000*i:1000*(i+1)] = np.random.normal(lly, llye, 1000)
    dv_wide[1000*i:1000*(i+1)] = np.random.normal(dv, dve, 1000)
    lum_ha_wide[1000*i:1000*(i+1)] = np.random.normal(lha, lhae, 1000)
    fesc_wide[1000*i:1000*(i+1)] = np.random.normal(fe, fee, 1000)
    ew_wide[1000*i:1000*(i+1)] = np.random.normal(ew, ewe, 1000)
    ew_wide[1000*i:1000*(i+1)][ew_wide[1000*i:1000*(i+1)]<=0] = np.mean(ew_wide[1000*i:1000*(i+1)])  # replace with mean
    lum_lya_wide[1000*i:1000*(i+1)][lum_lya_wide[1000*i:1000*(i+1)]<10] = np.mean(lum_lya_wide[1000*i:1000*(i+1)])  # replace with mean
    dv_wide[1000*i:1000*(i+1)][dv_wide[1000*i:1000*(i+1)]<10] = np.mean(dv_wide[1000*i:1000*(i+1)])  # replace with mean
    lum_ha_wide[1000*i:1000*(i+1)][lum_ha_wide[1000*i:1000*(i+1)]<10] = np.mean(lum_ha_wide[1000*i:1000*(i+1)])  # replace with mean

    
for i, (muv, muve, ew, ewe, lly, llye, dv, dve, lha, lhae, fe, fee) in enumerate(zip(_muv_deep, _muv_err_deep, \
                                                 _ew_deep, _ew_err_deep, _lum_lya_deep, _lum_lya_err_deep, \
                                                 _dv_lya_deep, _dv_lya_err_deep, _lum_ha_deep, _lum_ha_err_deep,\
                                                 _fesc_deep, _fesc_err_deep)):
    muv_deep[1000*i:1000*(i+1)] = np.random.normal(muv, muve, 1000)
    lum_lya_deep[1000*i:1000*(i+1)] = np.random.normal(lly, llye, 1000)
    dv_deep[1000*i:1000*(i+1)] = np.random.normal(dv, dve, 1000)
    lum_ha_deep[1000*i:1000*(i+1)] = np.random.normal(lha, lhae, 1000)
    fesc_deep[1000*i:1000*(i+1)] = np.random.normal(fe, fee, 1000)
    ew_deep[1000*i:1000*(i+1)] = np.random.normal(ew, ewe, 1000)
    ew_deep[1000*i:1000*(i+1)][ew_deep[1000*i:1000*(i+1)]<=0] = np.mean(ew_deep[1000*i:1000*(i+1)])  # replace with mean
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

def p_obs(lly, lha, muv, theta, mode='wide'):
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
    if mode == 'wide':
        w_lim = 80*w1 # 80
        f_lya_lim = f1*2e-17 # 2e-17
    elif mode == 'deep':
        w_lim = 25*w2
        f_lya_lim = f2*2e-18 # 2e-18
    # https://arxiv.org/pdf/2202.06642
    # https://arxiv.org/pdf/2003.12083
    # muv_lim = -18.0
    muv_lim = -17.75 # performing the integral to find the min muv that 
    # results in a mean of -18.7 reveals a value of -17.68, however 
    # we are likely still biased lower by w and f selections, so we take -17.75

    p_lya = normal_cdf(f_lya, f_lya_lim)
    p_ha = normal_cdf(f_ha, f_ha_lim)
    p_w = normal_cdf(w_emerg, w_lim)
    p_muv = 1 - normal_cdf(10**muv, 6*(10**muv_lim))
    
    return p_lya * p_ha * p_w * p_muv

def line(x, m, b):
    """
    Linear function.
    """
    return m * (x + 18.5) + b

XWIDE = np.array([np.log10(lum_lya_wide), dv_wide, np.log10(lum_ha_wide)])
XDEEP = np.array([np.log10(lum_lya_deep), dv_deep, np.log10(lum_ha_deep)])

savedir = '../data/cgm/'
# Y = PCA T of X
os.makedirs(f'{savedir}', exist_ok=True)

# pca = PCA(n_components=3)  # Keeping all 3 dimensions

# transform the wide sample to the PCA basis
XALL = np.concatenate((XWIDE, XDEEP), axis=1)

XC = XALL.mean(axis=1, keepdims=True)
np.save(f'{savedir}/xc.npy', XC)
XSTD = XALL.std(axis=1, keepdims=True)
np.save(f'{savedir}/xstd.npy', XSTD)
XALL0 = (XALL - XC) / XSTD

I = np.array([[1,0,0],[0,1,0],[0,0,1]])
A1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
A2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
A3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])
c1, c2, c3, c4 = 1, 1, 1/3, -1
# c1, c2, c3, c4 = np.load(f'{savedir}/coefficients.npy')
T = c1 * I + c2 * A1 + c3 * A2 + c4 * A3

YALL = np.linalg.inv(T) @ XALL0

XWIDE0 = XWIDE - XC
XDEEP0 = XDEEP - XC

YWIDE = np.linalg.inv(T) @ XWIDE0
YDEEP = np.linalg.inv(T) @ XDEEP0

def fit():
    """
    Fit the PCA coefficients to the observed fraction of LyA+Ha emitters.
    """

    NBINS = 20
    xc = np.load(f'{savedir}/xc.npy')
    xstd = np.load(f'{savedir}/xstd.npy')

    muv_centers = np.linspace(-20, -17, NBINS)
    f = np.zeros((NBINS, 2))
    f_err = np.zeros((NBINS, 2))
    ew_mean = np.zeros((NBINS, 2))
    ew_std = np.zeros((NBINS, 2))
    lly_mean = np.zeros((NBINS, 2))
    dv_mean = np.zeros((NBINS, 2))
    lha_mean = np.zeros((NBINS, 2))
    lly_std = np.zeros((NBINS, 2))
    dv_std = np.zeros((NBINS, 2))
    lha_std = np.zeros((NBINS, 2))
    fesc_mean = np.zeros((NBINS, 2))
    fesc_std = np.zeros((NBINS, 2))
    for i, muv in enumerate(muv_centers):

        muv_range = np.linspace(muv-0.5, muv+0.5, 1000)
        GALDENS = trapezoid(schechter(muv_range, phi_5, muv_star_5, alpha_5), muv_range)

        VOL_WIDE = 146.082  # 10^3 Mpc^3
        n_gal = GALDENS * VOL_WIDE
        n_gal_rel_err = np.sqrt(n_gal)/n_gal
        n_muv_wide = np.sum(np.abs(muv_wide - muv) < 0.5)/1000
        n_muv_wide_rel_err = np.sqrt(n_muv_wide)/n_muv_wide

        f[i,0] = n_muv_wide / n_gal
        f_err[i,0] = f[i,0] * np.sqrt(n_gal_rel_err**2 + n_muv_wide_rel_err**2)
        lly_mean[i,0] = np.mean(XWIDE[0, np.abs(muv_wide - muv) < 0.5])
        dv_mean[i,0] = np.mean(XWIDE[1, np.abs(muv_wide - muv) < 0.5])
        lha_mean[i,0] = np.mean(XWIDE[2, np.abs(muv_wide - muv) < 0.5])
        lly_std[i,0] = np.std(XWIDE[0, np.abs(muv_wide - muv) < 0.5])
        dv_std[i,0] = np.std(XWIDE[1, np.abs(muv_wide - muv) < 0.5])
        lha_std[i,0] = np.std(XWIDE[2, np.abs(muv_wide - muv) < 0.5])
        fesc_mean[i, 0] = np.mean(fesc_wide[np.abs(muv_wide - muv) < 0.5])
        fesc_std[i ,0] = np.std(fesc_wide[np.abs(muv_wide - muv) < 0.5])
        ew_mean[i, 0] = np.mean(ew_wide[np.abs(muv_wide - muv) < 0.5])
        ew_std[i, 0] = np.std(ew_wide[np.abs(muv_wide - muv) < 0.5])
        
        VOL_DEEP = 7.877 # 10^3 Mpc^3
        # NOTE factor of three accounts for the weighted mean of 
        # UDF10 (1.15 arcmin2 31 hr) and MOSAIC (9 arcmin2 10 hr)
        n_gal = GALDENS * VOL_DEEP
        n_gal_rel_err = np.sqrt(n_gal)/n_gal
        n_muv_deep = np.sum(np.abs(muv_deep - muv) < 0.5)/1000
        n_muv_deep_rel_err = np.sqrt(n_muv_deep)/n_muv_deep

        f[i,1] = n_muv_deep / n_gal
        f_err[i,1] = f[i,1] * np.sqrt(n_gal_rel_err**2 + n_muv_deep_rel_err**2)
        lly_mean[i,1] = np.mean(XDEEP[0, np.abs(muv_deep - muv) < 0.5])
        dv_mean[i,1] = np.mean(XDEEP[1, np.abs(muv_deep - muv) < 0.5])
        lha_mean[i,1] = np.mean(XDEEP[2, np.abs(muv_deep - muv) < 0.5])
        lly_std[i,1] = np.std(XDEEP[0, np.abs(muv_deep - muv) < 0.5])
        dv_std[i,1] = np.std(XDEEP[1, np.abs(muv_deep - muv) < 0.5])
        lha_std[i,1] = np.std(XDEEP[2, np.abs(muv_deep - muv) < 0.5])
        fesc_mean[i, 1] = np.mean(fesc_deep[np.abs(muv_deep - muv) < 0.5])
        fesc_std[i ,1] = np.std(fesc_deep[np.abs(muv_deep - muv) < 0.5])
        ew_mean[i, 1] = np.mean(ew_deep[np.abs(muv_deep - muv) < 0.5])
        ew_std[i, 1] = np.std(ew_deep[np.abs(muv_deep - muv) < 0.5])

    np.save(f'{savedir}/f.npy', f)
    np.save(f'{savedir}/f_err.npy', f_err)
    np.save(f'{savedir}/lly_mean.npy', lly_mean)
    np.save(f'{savedir}/dv_mean.npy', dv_mean)
    np.save(f'{savedir}/lha_mean.npy', lha_mean)
    np.save(f'{savedir}/lly_std.npy', lly_std)
    np.save(f'{savedir}/dv_std.npy', dv_std)
    np.save(f'{savedir}/lha_std.npy', lha_std)

    def objective(params):
        m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = params
        theta = [w1, w2, f1, f2, fh]

        # Fit the PCA coefficients to the observed fraction of LyA+Ha emitters
        mu1 = line(muv_centers, m1, b1)
        mu2 = line(muv_centers, m2, b2)
        mu3 = line(muv_centers, m3, b3)

        y1 = np.random.normal(mu1, std1, (1000, NBINS))
        y2 = np.random.normal(mu2, std2, (1000, NBINS))
        y3 = np.random.normal(mu3, std3, (1000, NBINS))

        # NOTE something is going wrong with the values here
        Y = np.stack([y1, y2, y3], axis=-2)
        X0 = T @ Y  # Transform to PCA basis

        lly, dv, lha = X0[:,0,:], X0[:,1,:], X0[:,2,:]
        lly = lly * xstd[0] + xc[0]
        dv = dv * xstd[1] + xc[1]
        lha = lha * xstd[2] + xc[2]

        # Apply CGM transmission
        tcgm = t_cgm(dv.flatten(), np.tile(muv_centers, 1000)).reshape((1000, NBINS))
        lly += np.log10(tcgm)

        logp = 0

        use_fobs = True  # Whether to use the observed fraction in the likelihood
        # use_fobs = False  # For testing without observed fraction

        # calculate the distance from the expected observed fraction
        _p_obs_wide = p_obs(10**lly, 10**lha, muv_centers.reshape((1, 20)), theta, mode='wide')
        _p_obs_deep = p_obs(10**lly, 10**lha, muv_centers.reshape((1, 20)), theta, mode='deep')

        if use_fobs:

            f_wide = np.sum(_p_obs_wide, axis=0) / 1000
            f_deep = np.sum(_p_obs_deep, axis=0) / 1000

            p_wide = -0.5*((f_wide - f[:,0])/f_err[:,0])**2
            p_deep = -0.5*((f_deep - f[:,1])/f_err[:,1])**2

            logp += np.sum(p_wide) + np.sum(p_deep)


        use_fesc = True  # Whether to use the escape fraction in the likelihood
        # use_fesc = False  # For testing without escape fraction
        if use_fesc:

            _p_obs_wide = _p_obs_wide / np.sum(_p_obs_wide, axis=0, keepdims=True)
            _p_obs_wide[np.isnan(_p_obs_wide)] = 0

            beta = get_beta_bouwens14(muv_centers.reshape((1, 20)))
            ew = (1215.67/2.47e15)*(10**lly)*10**(-0.4 * (51.6 - muv_centers.reshape((1, 20)))) \
                 * (1215.6/1500) ** (-1*beta - 2)

            lly_wide_mean = np.sum(lly*_p_obs_wide, axis=0)
            lha_wide_mean = np.sum(lha*_p_obs_wide, axis=0)
            dv_wide_mean = np.sum(dv*_p_obs_wide, axis=0)
            ew_wide_mean = np.sum(ew*_p_obs_wide, axis=0)

            _p_obs_deep = _p_obs_deep / np.sum(_p_obs_deep, axis=0, keepdims=True)
            _p_obs_deep[np.isnan(_p_obs_deep)] = 0
            lly_deep_mean = np.sum(lly*_p_obs_deep, axis=0)
            lha_deep_mean = np.sum(lha*_p_obs_deep, axis=0)
            dv_deep_mean = np.sum(dv*_p_obs_deep, axis=0)
            ew_deep_mean = np.sum(ew*_p_obs_deep, axis=0)

            fesc_wide_mean = (10**lly_wide_mean)/(11.4*10**lha_wide_mean)
            p_wide = -0.5 * ((ew_wide_mean - ew_mean[:,0])/ew_std[:,0])**2 -\
                0.5 * ((dv_wide_mean - dv_mean[:,0]) / dv_std[:,0])**2 - \
                0.5 * ((fesc_wide_mean - fesc_mean[:,0]) / fesc_std[:,0])**2
            
            fesc_deep_mean = (10**lly_deep_mean)/(11.4*10**lha_deep_mean)
            p_deep = -0.5 * ((ew_deep_mean - ew_mean[:,1]) / ew_std[:,1])**2 - \
                0.5 * ((dv_deep_mean - dv_mean[:,1]) / dv_std[:,1])**2 - \
                0.5 * ((fesc_deep_mean - fesc_mean[:,1]) / fesc_std[:,1])**2

            logp += np.sum(p_wide) + np.sum(p_deep)

        if -1*logp < 50:
            np.save(f'{savedir}/fit_params.npy', params)

        return -1*logp

    bounds = [(-1, 1)]*3 + [(-3, 3)]*3 + [(0.01, 1)]*3 + [(0.5, 1.5)]*5

    # result = differential_evolution(objective, x0=x0, bounds=bounds, maxiter=100, \
    #                                  popsize=5, disp=True)
    result = differential_evolution(objective, bounds=bounds, maxiter=500, mutation=(0.1, 1.9), \
                                     popsize=20, disp=True, recombination=0.5)
    all_params = result.x
    np.save(f'{savedir}/fit_params.npy', all_params)
    print("Fitted parameters:", result.x)
    return result.x

fit_params = fit()

NBINS = 20
muv_centers = np.linspace(-20, -17, NBINS)
xc = np.load(f'{savedir}/xc.npy')
xstd = np.load(f'{savedir}/xstd.npy')
f = np.load(f'{savedir}/f.npy')
f_err = np.load(f'{savedir}/f_err.npy')
m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load(f'{savedir}/fit_params.npy')
theta = [w1, w2, f1, f2, fh]

# Fit the PCA coefficients to the observed fraction of LyA+Ha emitters
mu1 = line(muv_centers, m1, b1)
mu2 = line(muv_centers, m2, b2)
mu3 = line(muv_centers, m3, b3)

y1 = np.random.normal(mu1, std1, (1000, NBINS))
y2 = np.random.normal(mu2, std2, (1000, NBINS))
y3 = np.random.normal(mu3, std3, (1000, NBINS))

# NOTE something is going wrong with the values here
Y = np.stack([y1, y2, y3], axis=-2)
X0 = T @ Y  # Transform to PCA basis

lly, _, lha = X0[:,0,:], X0[:,1,:], X0[:,2,:]
lly = lly * xstd[0] + xc[0]
lha = lha * xstd[2] + xc[2]

# Apply CGM transmission
tcgm = t_cgm(dv.flatten(), np.tile(muv_centers, 1000)).reshape((1000, NBINS))
lly += np.log10(tcgm)

_p_obs_wide = p_obs(10**lly, 10**lha, muv_centers.reshape((1, 20)), theta, mode='wide')
_p_obs_deep = p_obs(10**lly, 10**lha, muv_centers.reshape((1, 20)), theta, mode='deep')

f_wide = np.sum(_p_obs_wide, axis=0) / 1000
f_deep = np.sum(_p_obs_deep, axis=0) / 1000

fig, axs = plt.subplots(figsize=(6, 6), constrained_layout=True)

axs.plot(muv_centers, f_wide, color='red', linestyle=':', alpha=0.5, label='This Work')
axs.plot(muv_centers, f_deep, color='orange', linestyle=':', alpha=0.5)
axs.errorbar(muv_centers, f[:,0], yerr=f_err[:,0], fmt='o', color='red', markersize=5, label='MUSE-Wide')
axs.errorbar(muv_centers, f[:,1], yerr=f_err[:,1], fmt='o', color='orange', markersize=5, label='MUSE-Deep')
axs.set_ylabel(r'$f_{\rm obs}$', fontsize=font_size)
axs.set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs.set_yscale('log')
axs.legend(fontsize=int(font_size/1.5), loc='lower left')

figures_dir = '../out'
plt.savefig(f'{figures_dir}/fobs.pdf')
