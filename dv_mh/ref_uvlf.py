import numpy as np

def get_ref_uvlf(redshift):
    # BOUWENS 2021
    b21_mag = [[-22.52, -22.02, -21.52, -21.02, -20.52, -20.02, -19.52, -18.77, -17.77, -16.77],
            [-22.19, -21.69, -21.19, -20.68, -20.19, -19.69, -19.19, -18.69, -17.94, -16.94],
            [-21.85, -21.35, -20.85, -20.10, -19.35, -18.6, -17.6]]
    b21_phi = [[2e-6, 1.4e-5, 5.1e-5, 1.69e-4, 3.17e-4, 7.24e-4, 1.124e-3, 2.82e-3, 8.36e-3, 1.71e-2],
            [1e-6, 4.1e-5, 4.7e-5, 1.98e-4, 2.83e-4, 5.89e-4, 1.172e-3, 1.433e-3, 5.76e-3, 8.32e-3],
            [3e-6, 1.2e-5, 4.1e-5, 1.2e-4, 6.57e-4, 1.1e-3, 3.02e-3]]
    b21_phi_err = [[2e-6, 5e-6, 1.1e-5, 2.4e-5, 4.1e-5, 8.7e-5, 1.57e-4, 4.4e-4, 1.66e-3, 5.26e-3],
                [2e-6, 1.1e-5, 1.5e-5, 3.6e-5, 6.6e-5, 1.26e-4, 3.36e-4, 4.19e-4, 1.44e-3, 2.9e-3],
                [2e-6, 4e-6, 1.1e-5, 4e-5, 2.33e-4, 3.4e-4, 1.14e-3]]

    b21_6 = np.array(b21_phi[0])
    b21_7 = np.array(b21_phi[1])
    b21_8 = np.array(b21_phi[2])

    b21_6_err = np.array(b21_phi_err[0])
    b21_7_err = np.array(b21_phi_err[1])
    b21_8_err = np.array(b21_phi_err[2])

    logphi_b21_6 = np.log10(b21_6)
    logphi_b21_7 = np.log10(b21_7)
    logphi_b21_8 = np.log10(b21_8)

    logphi_err_b21_6_up = np.log10(b21_6 + b21_6_err) - logphi_b21_6
    logphi_err_b21_7_up = np.log10(b21_7 + b21_7_err) - logphi_b21_7
    logphi_err_b21_8_up = np.log10(b21_8 + b21_8_err) - logphi_b21_8

    logphi_err_b21_6_low = logphi_b21_6 - np.log10(b21_6 - b21_6_err)
    logphi_err_b21_7_low = logphi_b21_7 - np.log10(b21_7 - b21_7_err)
    logphi_err_b21_8_low = logphi_b21_8 - np.log10(b21_8 - b21_8_err)

    logphi_err_b21_6_low[np.isinf(logphi_err_b21_6_low)] = np.abs(logphi_b21_6[np.isinf(logphi_err_b21_6_low)])
    logphi_err_b21_7_low[np.isinf(logphi_err_b21_7_low)] = np.abs(logphi_b21_7[np.isinf(logphi_err_b21_7_low)])
    logphi_err_b21_8_low[np.isinf(logphi_err_b21_8_low)] = np.abs(logphi_b21_8[np.isinf(logphi_err_b21_8_low)])

    logphi_err_b21_6_low[np.isnan(logphi_err_b21_6_low)] = np.abs(logphi_b21_6[np.isnan(logphi_err_b21_6_low)])
    logphi_err_b21_7_low[np.isnan(logphi_err_b21_7_low)] = np.abs(logphi_b21_7[np.isnan(logphi_err_b21_7_low)])
    logphi_err_b21_8_low[np.isnan(logphi_err_b21_8_low)] = np.abs(logphi_b21_8[np.isnan(logphi_err_b21_8_low)])

    logphi_b21 = [logphi_b21_6, logphi_b21_7, logphi_b21_8]
    logphi_err_b21_up = [logphi_err_b21_6_up, logphi_err_b21_7_up, logphi_err_b21_8_up]
    logphi_err_b21_low = [logphi_err_b21_6_low, logphi_err_b21_7_low, logphi_err_b21_8_low]

    if redshift == 6.0:
        return b21_mag[0], logphi_b21[0], logphi_err_b21_up[0], logphi_err_b21_low[0]
    elif redshift == 7.0:
        return b21_mag[1], logphi_b21[1], logphi_err_b21_up[1], logphi_err_b21_low[1]
    elif redshift == 8.0:
        return b21_mag[2], logphi_b21[2], logphi_err_b21_up[2], logphi_err_b21_low[2]
    else:
        raise ValueError('Redshift not in B21 data.')