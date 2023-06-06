import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Functions to calculate the Quality Water Index Polynomial (QWIP) using the Apparent Visible Wavelength (AVW)
# Dierssen et al., 2022: https://doi.org/10.3389/frsen.2022.869611
# Translated from MATLAB - Mortimer Werther 2023 (EAWAG, Switzerland)

# DataFrame (df) columns represent hyperspectral wavelengths (e.g. '400'), rows the Rrs values. No other columns and formats are currently handled.
# The QWIP-specific columns are added as new columns.
# Load the df before running the code, e.g. df = pd.read_csv('your_df.csv')

def extract_columns(df):
    """
    Extract specific columns from an input DataFrame (df).

    :param df: df containing the data
    :return: A tuple containing the extracted columns and their respective np arrays
    """
    
    rrs_vis = df.loc[:, '400':'700']
    rrs_492 = df.loc[:, '492']
    rrs_665 = df.loc[:, '665']
    
    return rrs_vis, rrs_492.to_numpy(), rrs_665.to_numpy()


def calculate_avw(rrs_vis):
    """
    Calculate the AVW values.

    :param rrs_vis: df containing Rrs values from 400 to 700 nm
    :return: np array containing AVW values
    """
    
    wave = np.arange(400, 701)
    rrs = rrs_vis.to_numpy()

    avw = np.sum(rrs, axis=1) / np.sum(rrs / wave, axis=1)
    avw = avw.reshape(-1, 1)

    return avw


def calculate_ndi(rrs_vis, wave):
    """
    Calculate the Normalized Difference Index (NDI) values.

    :param rrs: np array containing Rrs values
    :param wave: np arraycontaining wavelengths from 400 to 700 nm
    :return: A np array containing NDI values
    """
    
    index_492 = np.argmin(np.abs(wave - 492))
    index_665 = np.argmin(np.abs(wave - 665))
    ndi = (rrs_vis[:, index_665] - rrs_vis[:, index_492]) / (rrs_vis[:, index_665] + rrs_vis[:, index_492])
    
    return ndi, index_492, index_665


def calculate_qwip_score(ndi, avw):
    """
    Calculate the QWIP score.

    :param ndi: np array containing NDI values
    :param avw: np array containing AVW values
    :return: A np array containing QWIP scores
    
    """
    # see paper
    p = [-8.399884740300151e-09, 1.715532100780679e-05, -1.301670056641901e-02,
         4.357837742180596e+00, -5.449532021524279e+02]
    ndi_pred = (p[0]*avw**4 + p[1]*avw**3 + p[2]*avw**2 + p[3]*avw**1 + p[4])
    qwip_score = ndi - ndi_pred.squeeze()
    
    return qwip_score


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


def calculate_classification_indices(rrs_492, rrs_560, rrs_665):
    """
    Calculate classification indices.

    :param rrs_492: np array containing Rrs values at 492 nm
    :param rrs_560: np array containing Rrs values at 560 nm
    :param rrs_665: np array containing Rrs values at 665 nm
    :return: A tuple containing boolean numpy arrays for each classification index
    
    """
    
    step1 = (rrs_665 > rrs_560)
    # see paper
    step2 = (rrs_665 > 0.025)
    step3 = (rrs_560 < rrs_492)

    ind_600a = (step1 | step2)
    ind_500a = (~step1 & ~step2) & ~step3
    ind_400a = (~step1 & ~step2) & step3

    return ind_600a, ind_500a, ind_400a

## Run the functions using the df

rrs_vis, rrs_492, rrs_665 = extract_columns(df)
avw = calculate_avw(rrs_vis)
wave = np.arange(400, 701) #i.e. 400 - 700 nm
ndi, index_492, index_665 = calculate_ndi(rrs_vis.to_numpy(), wave)
qwip_score = calculate_qwip_score(ndi, avw)
qwip_flag = flag_qwip_score(qwip_score)
rrs_560 = rrs_vis.loc[:, '560'].to_numpy()
ind_600a, ind_500a, ind_400a = calculate_classification_indices(rrs_492, rrs_560, rrs_665)

# add QWIP values to df
df['ndi'] = ndi
df['index_492'] = index_492
df['index_665'] = index_665
df['qwip_score'] = qwip_score
df['qwip_flag'] = qwip_flag
df['ind_400a'] = ind_400a
df['ind_500a'] = ind_500a
df['ind_600a'] = ind_600a

# Generate figure to show QCI index relative to AVW
fit1a = fit1 + 0.1
fit1b = fit1 - 0.1
fit2a = fit1 + 0.2
fit2b = fit1 - 0.2
fit3a = fit1 + 0.3
fit3b = fit1 - 0.3
fit4a = fit1 + 0.4
fit4b = fit1 - 0.4

## Plots 
fig, ax = plt.subplots()

g1 = ax.plot(avw[ind_500a], NDI[ind_500a], 'og', markersize=1, label='Index 500', alpha=0.5)
g2 = ax.plot(avw[ind_400a], NDI[ind_400a], 'ob', markersize=1, label='Index 400', alpha=0.5)
g3 = ax.plot(avw[ind_600a], NDI[ind_600a], 'or', markersize=1, label='Index 600', alpha=0.5)
ax.plot(avw_poly, fit1, '-k', linewidth=2)
qwip_lines = []
qwip_lines.append(ax.plot(avw_poly, fit1a, '--g', linewidth=2, label='QWIP ± 0.1')[0])
qwip_lines.append(ax.plot(avw_poly, fit1b, '--g', linewidth=2)[0])
qwip_lines.append(ax.plot(avw_poly, fit2a, '--', linewidth=2, color=[0.9290, 0.6940, 0.1250], label='QWIP ± 0.2')[0])
qwip_lines.append(ax.plot(avw_poly, fit2b, '--', linewidth=2, color=[0.9290, 0.6940, 0.1250])[0])
qwip_lines.append(ax.plot(avw_poly, fit3a, '--', linewidth=2, color=[0.8500, 0.3250, 0.0980], label='QWIP ± 0.3')[0])
qwip_lines.append(ax.plot(avw_poly, fit3b, '--', linewidth=2, color=[0.8500, 0.3250, 0.0980])[0])
qwip_lines.append(ax.plot(avw_poly, fit4a, '-r', linewidth=2, label='QWIP ± 0.4')[0])
qwip_lines.append(ax.plot(avw_poly, fit4b, '-r', linewidth=2)[0])

ax.set_xlabel('AVW (nm)', fontsize=16)
ax.set_ylabel(f'NDI ({wave[index_492]},{wave[index_665]})', fontsize=16)
ax.set_ylim([-2.5, 2])
ax.set_xlim([440, 630])

# Combined legend
legend1 = ax.legend(handles=[g2[0],g1[0], g3[0]], loc='lower right', fontsize=12)
legend2 = ax.legend(handles=qwip_lines[::2], loc='upper left', fontsize=12)
ax.add_artist(legend1)

#plt.savefig('qwip_wispstation_lxp.pdf')

plt.show()
