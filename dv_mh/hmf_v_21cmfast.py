import os

import numpy as np
import py21cmfast as p21c

from tqdm import tqdm

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p, e, m_e, G

from scipy.integrate import trapezoid
from scipy.special import gamma, erf
from scipy.optimize import curve_fit, differential_evolution
from scipy.interpolate import RegularGridInterpolator

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

from halomod.halo_model import MassFunction

n_halos = 1000000
redshift = 5.0

hmf_loc_10_ST = MassFunction(z=redshift, Mmin=10, Mmax=15, dlog10m=0.05, hmf_model='SMT')

m, dndm = hmf_loc_10_ST.m/Planck18.h, hmf_loc_10_ST.dndm*Planck18.h**3  # Msun, comoving Mpc^-3 Msun^-1
N_HALOS_MPC3 = trapezoid(hmf_loc_10_ST.dndm, x=hmf_loc_10_ST.m) * (Planck18.h**3)  # comoving Mpc^-3
EFFECTIVE_VOLUME = n_halos / N_HALOS_MPC3
mh = np.random.choice(m, size=n_halos, p=dndm/np.sum(dndm))

# We are using a relatively small box for this test
inputs = p21c.InputParameters.from_template(
    'latest-dhalos',
    random_seed=42,
).evolve_input_structs(
    SAMPLER_MIN_MASS=1e9,
    BOX_LEN=100,
    DIM=200,
    HII_DIM=50,
    USE_TS_FLUCT=False,
    INHOMO_RECO=False,
    HALOMASS_CORRECTION=1.,
    USE_EXP_FILTER=False,
    USE_UPPER_STELLAR_TURNOVER=False,
    CELL_RECOMB=False,
    R_BUBBLE_MAX=15.
).clone(
    node_redshifts=(8,10)
)

#set up some histogram parameters for plotting hmfs
edges = np.logspace(7, 13, num=64)
widths = np.diff(edges)
dlnm = np.log(edges[1:]) - np.log(edges[:-1])
centres = (edges[:-1] * np.exp(dlnm / 2)).astype("f4")
volume = inputs.simulation_options.BOX_LEN**3
little_h = inputs.cosmo_params.cosmo.H0.to("km s-1 Mpc-1") / 100

mf_pkg_st = MassFunction(
    z=inputs.node_redshifts[-1],
    Mmin=7,
    Mmax=15,
    cosmo_model=inputs.cosmo_params.cosmo,
    hmf_model="ST",
    hmf_params={"a": 0.73, "p": 0.175, "A": 0.353},
    transfer_model="EH",
)
mf_pkg_st2 = MassFunction(
    z=inputs.node_redshifts[-2],
    Mmin=7,
    Mmax=15,
    cosmo_model=inputs.cosmo_params.cosmo,
    hmf_model="ST",
    hmf_params={"a": 0.73, "p": 0.175, "A": 0.353},
    transfer_model="EH",
)

# create the initial conditions
init_box = p21c.compute_initial_conditions(
    inputs=inputs,
)

halolist_init = p21c.determine_halo_catalog(
    redshift=inputs.node_redshifts[-1],
    initial_conditions=init_box,
    inputs=inputs
)

# sample from reference
mh_samples = np.random.choice(
        mf_pkg_st.m / little_h,
        p=mf_pkg_st.dndlnm * (little_h**3) / np.sum(mf_pkg_st.dndlnm * (little_h**3)),
        size=n_halos
)
from scipy.integrate import trapezoid
expected_num_halos = trapezoid(mf_pkg_st.dndlnm * (little_h.value**3), x=np.log(mf_pkg_st.m / little_h.value))
EFFECTIVE_VOLUME = n_halos / expected_num_halos
hist_s, _ = np.histogram(mh_samples, edges)
mf_s = hist_s / EFFECTIVE_VOLUME / dlnm

# get the mass function
masses = halolist_init.get('halo_masses')
hist, _ = np.histogram(masses, edges)
mf = hist / volume / dlnm
plt.loglog(
    mf_pkg_st.m / little_h,
    mf_pkg_st.dndlnm * (little_h**3),
    color="C0",
    linewidth=2,
    linestyle=":",
    label="reference HMF",
)
plt.loglog(centres, mf, color="C0", label="binned sample")
plt.loglog(centres, mf_s, color="C1", label="binned reference sample")
plt.loglog(centres, 1 / volume / dlnm, "k:", label="one halo in box")

plt.xlim([1e9, 5e12])
plt.ylim([1e-7, 1e2])
plt.ylabel("dNdlnM (Mpc-3)")
plt.xlabel("M (Msun)")
plt.legend()
plt.show()