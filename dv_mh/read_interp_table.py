import numpy as np
from scipy.interpolate import RegularGridInterpolator

def interp_kuv(SFR, Mstar, z, bounds_error=True, fill_value=None, interpolation_table_loc = '/groups/astro/ivannik/projects/UVLF_clust/interpolation_table.npy'):
    """
    Trilinear interpolation on a regular grid.
    """
    table = np.load(interpolation_table_loc)
    SFR_grid = np.logspace(-5,5,100)
    Ms_grid = np.logspace(3,13,100)
    z_grid = np.linspace(5,15,10)
    
    interp = RegularGridInterpolator(
        (z_grid, Ms_grid, SFR_grid),
        table,
        method="linear",
        bounds_error=bounds_error,
        fill_value=fill_value,
    )
    return interp((z,Mstar,SFR))


Most recent parameterization
F_STAR10                    = 10**(-0.281E+01) *cosmo.Om0 / cosmo.Ob0
SIGMA_STAR                  = 0.2393E+00
t_STAR                      = 0.1676E+00
ALPHA_STAR                  = 0.4709E+00
SIGMA_SFMS_0                = 0.9297E-01
SIGMA_SFR_INDEX             = -0.1884E-01
UPPER_STELLAR_TURNOVER_MASS = 0.1444E+02
SIGMA_UV                    = 0.244947465229813069E+00

Ask polish mit:
clustering lifetime estimate?