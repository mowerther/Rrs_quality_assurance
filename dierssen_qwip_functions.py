import numpy as np

# Functions to calculate the Quality Water Index Polynomial (QWIP) using the Apparent Visible Wavelength (AVW)
# Dierssen et al., 2022: https://doi.org/10.3389/frsen.2022.869611
# Translated from MATLAB - Mortimer Werther 2023 (EAWAG, Switzerland)

def find_closest_wavelength(wavelengths, target):

    """
    Find the closest wavelength to the target.

    :param wavelengths: list of available wavelengths of the multispectral sensor (ints)
    :param target: target wavelength, an integer
    :return: closest wavelength in the list
    """
    closest_wavelength = min(wavelengths, key=lambda x:abs(x-target))
    return closest_wavelength

def extract_columns(df, spectral_resolution, wavelengths=None):
    """
    Extract specific columns from an input DataFrame (df).

    :param df: df containing the data
    :param spectral_resolution: 'hyper' or 'multi' to specify the spectral resolution of the data
    :param wavelengths (optional): list of available wavelengths of the multispectral sensor (ints)
    :return: A tuple containing the extracted columns and their respective np arrays
    """
    if spectral_resolution == 'hyper':
        rrs_vis = df.loc[:, '400':'700']
        #rrs_492 = df.loc[:, '492']
        #rrs_665 = df.loc[:, '665']

    elif spectral_resolution == 'multi':
        if wavelengths is None:
            raise ValueError("When 'spectral_resolution' is 'multi', 'wavelengths' must be provided.")
        wls = [str(wavelength) for wavelength in wavelengths]
        rrs_vis = df.loc[:, wls]
        # Find the index of the band closest to 492 and 665 nm or what is closest to 492/665, e.g. for MODIS 488
        #rrs_492 = df.loc[:, str(find_closest_wavelength(wavelengths, 492))]
        #rrs_665 = df.loc[:, str(find_closest_wavelength(wavelengths, 665))]
    
    return rrs_vis#, rrs_492.to_numpy(), rrs_665.to_numpy()


def calculate_avw(rrs_vis, spectral_resolution, wavelengths=None, sensor_coeffs=None):
    """
    Calculate the apparent visible wavelenght (AVW) values.

    :param rrs_vis: df containing Rrs values
    :param spectral_resolution: 'hyper' or 'multi' to specify the spectral resolution of the data
    :param wavelengths (optional): list of available wavelengths of the multispectral sensor (ints)
    :param sensor_coeffs (optional): list of multispectral sensor coefficients.
    :return: np array containing avw values.
    """

    if spectral_resolution == 'hyper': 
        wave = np.arange(400, 701)
    elif spectral_resolution == 'multi':
        if wavelengths is None:
            raise ValueError("When 'spectral_resolution' is 'multi', 'wavelengths' must be provided.")
        wave = wavelengths
    else:
        raise ValueError("Invalid spectral resolution: must be either 'multi' or 'hyper'")

    rrs = rrs_vis.to_numpy()
    avw = np.sum(rrs, axis=1) / np.sum(rrs / wave, axis=1)
    avw = avw.reshape(-1, 1)
    if spectral_resolution == 'hyper':
        return avw
    elif spectral_resolution == 'multi':
        if sensor_coeffs is None:
            raise ValueError("When 'spectral_resolution' is 'multi', 'sensor_coeffs' must be provided.")
        # "calibrate" AVW to a hyperspectral-equivalent value (See original code by Ryan Vandermeulen)
        avw = (sensor_coeffs[0]*avw**5 + sensor_coeffs[1]*avw**4 + sensor_coeffs[2]*avw**3 + sensor_coeffs[3]*avw**2 + sensor_coeffs[4]*avw**1 + sensor_coeffs[5])
        avw.reshape(-1,1)
        return avw
    else:
        raise ValueError("Invalid spectral resolution: must be either 'multi' or 'hyper'")


def calculate_ndi(rrs_vis, wavelengths):
    """
    Calculate the Normalized Difference Index (NDI) values.

    :param rrs: np array containing Rrs values
    :param wave: np arraycontaining wavelengths from 400 to 700 nm
    :return: A np array containing NDI values and indices of 492 and 665 nm.
    """

    wavelengths = np.array(wavelengths)
    
    # Find closest indices 
    index_492 = np.argmin(np.abs(wavelengths - 492))
    index_665 = np.argmin(np.abs(wavelengths - 665))

    # Calculate normalized difference index (ndi) / quality control index (qci) - term is used interchangably in the code and manuscript of the authors
    ndi = (rrs_vis[:, index_665] - rrs_vis[:, index_492]) / (rrs_vis[:, index_665] + rrs_vis[:, index_492])
    
    return ndi, index_492, index_665


def calculate_qwip_score(ndi, avw):
    """
    Calculate the QWIP score.

    :param ndi: np array containing NDI values
    :param avw: np array containing AVW values
    :return: A np array containing QWIP scores
    
    """
    # see paper, original coefficients that should be kept
    p = [-8.399884740300151e-09, 1.715532100780679e-05, -1.301670056641901e-02,
         4.357837742180596e+00, -5.449532021524279e+02]
    ndi_pred = (p[0]*avw**4 + p[1]*avw**3 + p[2]*avw**2 + p[3]*avw**1 + p[4])
    qwip_score = ndi_pred.squeeze() - ndi

    # the polynomial fit for plotting
    avw_poly = np.arange(400, 631)
    fit1 = np.polyval(p, avw_poly)  
    
    return qwip_score, fit1


def flag_qwip_score(qwip_score, threshold=0.2):
    """
    Flag QWIP scores based on a threshold.

    :param qwip_score: np array containing QWIP scores
    :param threshold: Threshold value for flagging (default: 0.2)
    :return: A boolean np array indicating flagged scores
    
    """
    
    abs_qwip_score = np.abs(qwip_score)
    qwip_flag = abs_qwip_score >= threshold
    
    return qwip_flag


def calculate_classification_indices(df, wavelengths):
    """
    Calculate classification indices.

    :param df: df containing the data
    :param wavelengths: list of available wavelengths of the multispectral sensor (ints)
    :return: A tuple containing boolean numpy arrays for each classification index

    """
    
    # Find closest wavelengths in df
    rrs_492_wavelength = find_closest_wavelength(wavelengths, 492)
    rrs_560_wavelength = find_closest_wavelength(wavelengths, 560)
    rrs_665_wavelength = find_closest_wavelength(wavelengths, 665)

    rrs_492 = df.loc[:, str(rrs_492_wavelength)].to_numpy()
    rrs_560 = df.loc[:, str(rrs_560_wavelength)].to_numpy()
    rrs_665 = df.loc[:, str(rrs_665_wavelength)].to_numpy()
    
    step1 = (rrs_665 > rrs_560)
    # see paper
    step2 = (rrs_665 > 0.025)
    step3 = (rrs_560 < rrs_492)

    ind_600a = (step1 | step2)
    ind_500a = (~step1 & ~step2) & ~step3
    ind_400a = (~step1 & ~step2) & step3

    return ind_600a, ind_500a, ind_400a