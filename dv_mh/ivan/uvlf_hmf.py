"""Python file computes p(Muv|Mh) for some astrophysical parameters"""

import numpy as np
import numba
from scipy.interpolate import RegularGridInterpolator
import argparse
import hmf
import math

from astropy import units as u
from astropy.cosmology import Planck18
from numba import njit, prange

INV_SQRT2PI = 0.3989422804014327
cosmo = Planck18

#hmf_loc = hmf.MassFunction(z=11)
def ms_mh_flattening(mh, cosmo, fstar_norm = 1.0, alpha_star_low = 0.5, M_knee=2.6e11):
    """
        Get scaling relations for SHMR based on Davies+in prep.
        Parameters
        ----------
        mh: float,
            halo mass at which we're evaluating the relation.
        Returns
        ----------
        ms_mean: floats; optional,
            a and b coefficient of the relation.
    """

    f_star_mean = fstar_norm
    f_star_mean /= (mh / M_knee) ** (-alpha_star_low) + (mh / M_knee) ** 0.61 #knee denominator
    f_star_mean *= (1e10 / M_knee) ** (-alpha_star_low) + (1e10 / M_knee) ** 0.61 #knee numerator
    return np.minimum(f_star_mean, cosmo.Ob0 / cosmo.Om0) * mh

def ms_mh(ms, fstar_norm=1, alpha_star_low=0.5, M_knee=2.6e11):
    """
        Get inverse of the SHMR relation
        Parameters
        ----------
        ms: float,
            stellar mass at which we're evaluating the relation.
        Returns
        ----------
        mh_mean: floats; optional,
            mh of the relation
    """
    mhs = np.logspace(5,18,1000)
    mss = ms_mh_flattening(mhs, cosmo = cosmo, fstar_norm=fstar_norm, alpha_star_low=alpha_star_low, M_knee=M_knee)
    return 10**np.interp(np.log10(ms), np.log10(mss), np.log10(mhs))

def linear_model_kuv(X, sigma_kuv):
    a,b,c = (0.05041177782984782, -0.029117831879005154, -0.04726733615202826)
    M, z = X
    sigmas = a * (M-9) + b * (z-6) - c * (z-6) * (M-9) + sigma_kuv
    sigmas = np.clip(sigmas, 0.0, 0.5)
    return sigmas

def SFMS(Mstar, SFR_norm = 1., z=9.25):
    """
        the functon returns SFR from Main sequence
    """
    b_SFR = -np.log10(SFR_norm) + np.log10(cosmo.H(z).to(u.yr ** (-1)).value)

    return Mstar * 10 ** b_SFR #* SFR_norm

def sigma_SFR_variable(Mstar, norm=0.18740570999999995, a_sig_SFR=-0.11654893):
    """
        Variable scatter of SFR-Mstar relation.
        It's based on FirstLight database.
    Parameters
    ----------
    Mstar: stellar mass at which the relation is taken

    Returns
    -------
    sigma: sigma of the relation
    """
    # a_sig_SFR = -0.11654893
    #b_sig_SFR = 1.35289501
    #     sigma = a_sig_SFR * np.log10(Mstar) + b_sig_SFR

    Mstar = np.asarray(Mstar)  # Convert input to a numpy array if not already

    sigma = a_sig_SFR * np.log10(Mstar/1e10) + norm
    sigma[Mstar > 10 ** 10] = norm

    return sigma

def kUV(SFR):
    """
        Simplest transformation between SFR and Luv
    """
    return SFR / (1.15 * 1e-28)


def Muv_Luv(Luv):
    """
        Luv to Muv
    """
    return -2.5 * np.log10(Luv) + 51.6



@njit(parallel=True, fastmath=True)
def _gauss_muv_sfr(muv_grid, muuv_of_sfr, sigma_uv_of_sfr):
    Nmuv  = muv_grid.size
    Nsfr  = muuv_of_sfr.size
    outT  = np.empty((Nmuv, Nsfr), dtype=np.float64)  # row-major writes
    for k in prange(Nsfr):
        mu   = muuv_of_sfr[k]
        sig  = sigma_uv_of_sfr[k]
        invs = 1.0 / sig
        norm = INV_SQRT2PI * invs
        c    = -0.5 * invs * invs
        for j in range(Nmuv):
            dx = muv_grid[j] - mu
            outT[j, k] = norm * math.exp(c * dx * dx)
    return outT.T  # (Nsfr, Nmuv)

@njit(parallel=True, fastmath=True)
def _gauss_sfr_mstar(sfr_samples, sfr_target_of_ms, sigma_sfr_of_sfr):
    Nsfr      = sfr_samples.size
    Nmstar    = sfr_target_of_ms.size
    out = np.empty((Nmstar, Nsfr), dtype=np.float64)
    for i in prange(Nmstar):
        mu = sfr_target_of_ms[i]
        for j in range(Nsfr):
            sig  = sigma_sfr_of_sfr[j]
            invs = 1.0 / sig
            norm = INV_SQRT2PI * invs
            c    = -0.5 * invs * invs
            dx   = (sfr_samples[j] - mu)
            out[i, j] = norm * math.exp(c * dx * dx)
    return out  # (Nmstar, Nsfr)

@njit(parallel=True, fastmath=True)
def _gauss_mstar_mh(mstar_samples, mstar_tgt_of_mh, sigma_SHMR):
    Nmstar = mstar_samples.size
    Nmh    = mstar_tgt_of_mh.size
    invs = 1.0 / sigma_SHMR
    norm = INV_SQRT2PI * invs
    c    = -0.5 * invs * invs
    out  = np.empty((Nmh, Nmstar), dtype=np.float64)
    for i in prange(Nmh):
        mu = mstar_tgt_of_mh[i]
        for j in range(Nmstar):
            dx = mstar_samples[j] - mu
            out[i, j] = norm * math.exp(c * dx * dx)
    return out  # (Nmh, Nmstar)

# ---------- your setup(), but leaner & faster ----------
def setup_sample_probabilities_fast(
    muv_grid, sigma_UV, muuv_of_sfr_grid, sfr_grid,
    mstar_grid, sigma_sfr_grid, mh_grid, sigma_SHMR,
    dndlnm_grid, *, Nsfr, Nmstar, Nmh, seed=0, use_float32=False
):
    rng = np.random.default_rng(seed)
    sfr_range   = (-5.0, 5.0)
    mstar_range = ( 2.0,12.0)
    mh_range    = ( 5.0,15.0)

    sfr_samples   = rng.uniform(*sfr_range,   Nsfr).astype(np.float64)
    mstar_samples = rng.uniform(*mstar_range, Nmstar).astype(np.float64)
    mh_samples    = rng.uniform(*mh_range,    Nmh).astype(np.float64)

    scale_sfr   = (sfr_range[1]   - sfr_range[0])   / float(Nsfr)
    scale_mstar = (mstar_range[1] - mstar_range[0]) / float(Nmstar)
    scale_mh    = (mh_range[1]    - mh_range[0])    / float(Nmh)
    total_scale = scale_sfr * scale_mstar * scale_mh

    # Interpolations
    muuv_of_sfr      = np.interp(sfr_samples, sfr_grid,      muuv_of_sfr_grid).astype(np.float64)
    sigma_sfr_of_sfr = np.interp(sfr_samples, sfr_grid,      sigma_sfr_grid   ).astype(np.float64)
    sfr_target_of_ms = np.interp(mstar_samples, mstar_grid,  sfr_grid         ).astype(np.float64)
    dndlnm_on_mh     = np.interp(mh_samples,   mh_grid,      dndlnm_grid      ).astype(np.float64)
    mstar_tgt_of_mh  = np.interp(mh_samples,   mh_grid,      mstar_grid       ).astype(np.float64)

    # sigma_UV can be scalar or array on sfr_grid; handle both:
    if np.ndim(sigma_UV) == 0:
        sigma_uv_of_sfr = np.full_like(sfr_samples, float(sigma_UV), dtype=np.float64)
    else:
        sigma_uv_of_sfr = np.interp(sfr_samples, sfr_grid, sigma_UV).astype(np.float64)

    # Build Gaussian tables with Numba
    p_muv_sfr   = _gauss_muv_sfr(muv_grid, muuv_of_sfr, sigma_uv_of_sfr)      # (Nsfr,   Nmuv)
    p_sfr_mstar = _gauss_sfr_mstar(sfr_samples, sfr_target_of_ms, sigma_sfr_of_sfr)  # (Nmstar, Nsfr)
    p_mstar_mh  = _gauss_mstar_mh(mstar_samples, mstar_tgt_of_mh, sigma_SHMR) # (Nmh,    Nmstar)

    if use_float32:
        p_muv_sfr   = np.ascontiguousarray(p_muv_sfr,   dtype=np.float32)
        p_sfr_mstar = np.ascontiguousarray(p_sfr_mstar, dtype=np.float32)
        p_mstar_mh  = np.ascontiguousarray(p_mstar_mh,  dtype=np.float32)
        dndlnm_on_mh= np.ascontiguousarray(dndlnm_on_mh,dtype=np.float32)

    return p_muv_sfr, p_sfr_mstar, p_mstar_mh, dndlnm_on_mh, total_scale, mh_samples



def pMuv_Mh_einsum(
    muv_grid, sigma_UV, muuv_of_sfr_grid, sfr_grid,
    mstar_grid, sigma_sfr_grid, mh_grid, sigma_SHMR, dndlnm_grid,
    *, Nsfr, Nmstar, Nmh, seed=0, use_float32=False
):
    p_muv_sfr, p_sfr_mstar, p_mstar_mh, dndlnm_on_mh, total_scale, mh_samples = setup_sample_probabilities_fast(
        muv_grid, sigma_UV, muuv_of_sfr_grid, sfr_grid,
        mstar_grid, sigma_sfr_grid, mh_grid, sigma_SHMR, dndlnm_grid,
        Nsfr=Nsfr, Nmstar=Nmstar, Nmh=Nmh, seed=seed, use_float32=use_float32
    )
    out = np.einsum('ma,ar,ru -> mu',
        p_mstar_mh,p_sfr_mstar, p_muv_sfr, optimize = 'greedy' )
    # Contract: dnd * B(mh,m*) * H(m*,sfr) * G(sfr,muv)  -> (Nmuv,)

    # Note: by index naming, result shape is 's' == Nmuv.
    return (out * total_scale).astype(np.float64), mh_samples

def interp_kuv(SFR, Mstar, z, bounds_error=False, fill_value=1.15e-28,
               interpolation_table_loc = '../../data/interpolation_table.npy'):
    """
    Trilinear interpolation on a regular grid.
    """
    table = np.load(interpolation_table_loc)
    SFR_grid = np.logspace(-5, 5, 100)
    Ms_grid = np.logspace(3, 13, 100)
    z_grid = np.linspace(5, 15, 10)

    interp = RegularGridInterpolator(
        (z_grid, Ms_grid, SFR_grid),
        table,
        method="linear",
        bounds_error=bounds_error,
        fill_value=fill_value,
    )
    return interp((z, Mstar, SFR))

def pUV_calc_numba(
        Muv,
        masses_hmf,
        dndm,
        f_star_norm=1.0,
        alpha_star=0.5,
        sigma_SHMR=0.3,
        sigma_SFMS_norm=0.0,
        t_star=0.5,
        a_sig_SFR=-0.11654893,
        z=11,
        M_knee=2.6e11,
        sigma_kuv = 0.1,
        mass_dependent_sigma_uv=False,
        seed=0,
        **kw,
):
    msss = ms_mh_flattening(10 ** masses_hmf, cosmo, alpha_star_low=alpha_star,
                            fstar_norm=f_star_norm, M_knee=M_knee)
    sfrs = SFMS(msss, SFR_norm=t_star, z=z)

    F_UV = interp_kuv(sfrs, msss, z) * sfrs
    muvs = Muv_Luv(F_UV)

    if mass_dependent_sigma_uv:
        sigma_kuv_var = linear_model_kuv((msss, z), sigma_kuv)
    else:
        sigma_kuv_var = sigma_kuv * np.ones(np.shape(msss))
    sigma_SFMS_var = sigma_SFR_variable(msss, norm=sigma_SFMS_norm,
                                        a_sig_SFR=a_sig_SFR)
    puvlf = pMuv_Mh_einsum(
        Muv,
        sigma_kuv_var,              # scalar (dispersion of Muv|SFR)
        muvs,      # mu_UV(SFR) tabulated on sfr_grid
        np.log10(sfrs),              # SFR grid (log10)
        np.log10(msss),            # M* grid (log10) for interpolation of mu_SFR(M*)
        sigma_SFMS_var,        # sigma_SFR(SFR) tabulated on sfr_grid
        masses_hmf,               # Mh grid (log10)
        sigma_SHMR,            # scalar (dispersion of log10 M* | Mh)
        dndm,           # d n / d ln M on mh_grid
        Nsfr=10_000,
        Nmstar=10_000,
        Nmh=10_000,
        seed=seed,
    )
    #The output of this is the ((p(Muv), Muv),Mh)
    #as such p(Muv) has the shape of (len(Mh), len(Mh))
    #In order to evaluate it, pick a halo mass (note that halo masses are random samples so they are not sorted)
    #then you can plot for this halo mass p(Muv).
    #On top of that this function can be used to make p(Muv|Mh) some Muv that is given as a median from the Muv-Mh relation
    return puvlf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #I'll implement a version where you can submit your own parameters in command line
    #For now I'll just make an example that uses my default
    import matplotlib.pyplot as plt
    from scipy.integrate import trapezoid

    z = 10
    param_MAP_new = [
        -0.280636368880184506E+01,
        0.239311686215938429E+00,
        0.167616139258982610E+00,
        0.470918493138135108E+00,
        0.929791798953653048E-01,
        -0.188417610576141620E-01,
        0.144476204688384300E+02,
        0.244956626802721206E+00
    ]
    hmf_loc_10 = hmf.MassFunction(z=10, Mmin=5, Mmax=15, dlog10m=0.05)  # , hmf_model='SMT')
    muv_bins = np.linspace(-26, -10, 150)
    dndlog10m = hmf_loc_10.dndlog10m * cosmo.h**3 * np.exp(- 5e8 / (hmf_loc_10.m / cosmo.h) )

    g = pUV_calc_numba(
        muv_bins,
        np.log10(hmf_loc_10.m/ cosmo.h),
        dndlog10m,
        f_star_norm=10 ** param_MAP_new[0],
        alpha_star=param_MAP_new[3],
        sigma_SHMR=param_MAP_new[1],
        sigma_SFMS_norm=param_MAP_new[4],
        t_star=param_MAP_new[2],
        a_sig_SFR=param_MAP_new[5],
        z=10,
        M_knee=10**param_MAP_new[6],
        sigma_kuv=param_MAP_new[7],
        mass_dependent_sigma_uv=False, #safer choice
    )

    log10mh_linspace  = g[1]
    # check normalization
    # print('Checking normalization...')
    # norm = trapezoid(g[0], x=muv_bins, axis=1)
    p_muv_given_mh = g[0] #/ norm[:, np.newaxis]
    dndlog10m = np.interp(log10mh_linspace, np.log10(hmf_loc_10.m/ cosmo.h), dndlog10m)
    # plt.contourf(muv_bins, g[1], np.log10(g[0] + 1e-20), levels=20)
    # plt.colorbar(label='log10 P(Muv|Mh)')
    # plt.xlabel('Muv')
    # plt.ylabel('log10 Mh [Msun]')
    # plt.show()
    # quit()

    dndMuv = trapezoid(p_muv_given_mh*dndlog10m[:,np.newaxis], x=log10mh_linspace, axis=0)
    plt.plot(muv_bins, np.log10(dndMuv), '.')
    plt.xlabel('Luv [erg/s/Hz]')
    plt.ylabel('phi [cMpc^-3 / (erg/s/Hz)]')
    plt.show()
    quit()