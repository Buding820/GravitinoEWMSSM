#!/usr/bin/env python3

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import LogFormatterSciNotation
from cProfile import label
import numpy as np
from matplotlib.ticker import LinearLocator, FixedLocator, AutoMinorLocator, MaxNLocator
import pandas as pd
import os
import sys
from shutil import which
import matplotlib.pyplot as plt
import json
import math


from matplotlib import rc, rcParams
from matplotlib.image import NonUniformImage
from matplotlib import cm, ticker
from matplotlib.font_manager import FontProperties
import matplotlib
config = {
    "font.family": ["serif", "Times New Roman"],
    "font.size": 20,
    "mathtext.fontset": 'stix',
    "font.serif": ['Computer Modern'],
    "text.latex.preamble": r"\usepackage{amsmath}"
}
rcParams.update(config)
plt.rcParams['axes.formatter.min_exponent'] = 2

df_fit = pd.read_csv("Data/SR-limit.csv")

def S95(bkg):
    return 2.0 * (math.sqrt(bkg + 1) + 1.0)

Lumi    = 3.e6 

brW2lv  = 0.1071 + 0.1063 
brZ2ll  = 0.033632 + 0.033662 
brH2bb  = 0.575 
brH2tt  = 0.063

N_SM = {
    "SR-HH":        9153.0,
    "SR-ZH":        2185.8,
    "SR-ZZ":        75.8,
    "SR-WH":        4028.2,
    "SR-WZ":        480.2
}
s95 = {
    "SR-HH":        260.06,
    "SR-ZH":        131.790,
    "SR-ZZ":        25.1563,
    "SR-WH":        175.995,
    "SR-WZ":        60.729
}


BR = {
    "SR-HH":        2.0 * brH2bb * brH2tt,
    "SR-ZH":        0.5 * brH2bb * brZ2ll,
    "SR-ZZ":        brZ2ll * brZ2ll,
    "SR-WH":        brW2lv * brH2bb,
    "SR-WZ":        brW2lv * brZ2ll
}

df_sma     = pd.read_csv("Data/XSect_SMa.csv")

df_sma['eff_HH']         = df_sma['SR-HH'] / ( df_sma['XSect'] * Lumi * 2.0 * brH2bb * brH2tt ) 
df_sma['eff_ZH']         = df_sma['SR-ZH'] / ( df_sma['XSect'] * Lumi * 0.5 * brH2bb * brZ2ll )
df_sma['eff_ZZ']         = df_sma['SR-ZZ'] / ( df_sma['XSect'] * Lumi * brZ2ll * brZ2ll )
df_sma.to_csv("Data/XSect_SMa.csv", index=None)

df_smb     = pd.read_csv("Data/XSect_SMb.csv")

df_smb['eff_WH']         = df_smb['SR-WH'] / (df_smb['XSect'] * Lumi * brW2lv * brH2bb)
df_smb['eff_WZ']         = df_smb['SR-WZ'] / (df_smb['XSect'] * Lumi * brW2lv * brZ2ll)
df_smb.to_csv("Data/XSect_SMb.csv", index=None)

from scipy.interpolate import interp1d
XSect_SMa   = interp1d(df_sma['mHiggsino'], np.log(df_sma['XSect']), kind="cubic", fill_value="extrapolate")
eff_HH      = interp1d(df_sma['mHiggsino'], df_sma['eff_HH'], kind="cubic", fill_value="extrapolate")
eff_ZH      = interp1d(df_sma['mHiggsino'], df_sma['eff_ZH'], kind="cubic", fill_value="extrapolate")
eff_ZZ      = interp1d(df_sma['mHiggsino'], df_sma['eff_ZZ'], kind="cubic", fill_value="extrapolate")

XSect_SMb   = interp1d(df_smb['mHiggsino'], np.log(df_smb['XSect']), kind="cubic", fill_value="extrapolate")
eff_WH      = interp1d(df_smb['mHiggsino'], df_smb['eff_WH'], kind="cubic", fill_value="extrapolate")
eff_WZ      = interp1d(df_smb['mHiggsino'], df_smb['eff_WZ'], kind="cubic", fill_value="extrapolate")

fig = plt.figure(figsize=[10, 8])
ax1 = fig.add_axes([0.169, 0.13, 0.672, 0.84])
axc = fig.add_axes([0.851, 0.15, 0.02, 0.80])
# ax2 = fig.add_axes([0.58, 0.16, 0.41, 0.83])

# ax1.plot(df_sma['mHiggsino'], df_sma['SR-WZ_95up'], '-', c='r')
# ax1.plot(df_sma['mHiggsino'], df_sma['XSect'], '-', c='r', zorder=20)
# ax1.plot(np.linspace(0, 1000, 100), np.exp(XSect_SMa(np.linspace(0, 1000, 100))), '-', c='b')

# ax1.set_yscale('log')
df = pd.read_csv("tot_data_1107.csv", index_col="ID")
data2 = pd.read_csv("param.csv", index_col="index")
df = df.sort_values(['LogL'])


# Higgsino-Dominated N1  

# Case SM-A 
dc1 = df[(df["N13"] ** 2 + df["N14"] ** 2 > df["N11"] ** 2) & (df["N13"] ** 2 + df["N14"] ** 2 > df["N12"] ** 2) & (df['WN2'] > df['WN2Gamma'] + df['WN2H'] + df['WN2Z']) ]
dc1 = dc1.sort_values(['LogL'])

dc1['BRN1Hg'] = dc1['WN1H'] / (dc1['WN1Gamma'] + dc1['WN1H'] + dc1['WN1Z'])
dc1['BRN1Zg'] = dc1['WN1Z'] / (dc1['WN1Gamma'] + dc1['WN1H'] + dc1['WN1Z'])

dc1['S_HH']         = np.exp(XSect_SMa(np.abs(dc1['mN1']))) * Lumi * dc1['BRN1Hg'] * dc1['BRN1Hg'] * BR['SR-HH'] * eff_HH(np.abs(dc1['mN1']))
dc1['S_ZH']         = np.exp(XSect_SMa(np.abs(dc1['mN1']))) * Lumi * dc1['BRN1Hg'] * dc1['BRN1Zg'] * BR['SR-ZH'] * eff_ZH(np.abs(dc1['mN1']))
dc1['S_ZZ']         = np.exp(XSect_SMa(np.abs(dc1['mN1']))) * Lumi * dc1['BRN1Zg'] * dc1['BRN1Zg'] * BR['SR-ZZ'] * eff_ZZ(np.abs(dc1['mN1']))

dc2     = dc1[dc1['S_HH'] > s95['SR-HH']]
dc3     = dc1[dc1['S_ZH'] > s95['SR-ZH']]
dc4     = dc1[dc1['S_ZZ'] > s95['SR-ZZ']]
dz1     = dc1[(dc1['S_HH'] <= s95['SR-HH']) & (dc1['S_ZH'] <= s95['SR-ZH']) & (dc1['S_ZZ'] <= s95['SR-ZZ']) ]

# Case SM-B
dd1 = df[(df["N13"] ** 2 + df["N14"] ** 2 > df["N11"] ** 2) & (df["N13"] ** 2 + df["N14"] ** 2 > df["N12"] ** 2) & (df['WN2'] <= df['WN2Gamma'] + df['WN2H'] + df['WN2Z']) ]
dd1 = dd1.sort_values(['LogL'])

dd1['BRN1Hg'] = dd1['WN1H'] / (dd1['WN1Gamma'] + dd1['WN1H'] + dd1['WN1Z'])
dd1['BRN1Zg'] = dd1['WN1Z'] / (dd1['WN1Gamma'] + dd1['WN1H'] + dd1['WN1Z'])

dd1['S_WH']    = np.exp(XSect_SMb(np.abs(dd1['mN1']))) * Lumi * dd1['BRN1Hg'] * BR['SR-WH']   * eff_WH(np.abs(dd1['mN1']))
dd1['S_WZ']    = np.exp(XSect_SMb(np.abs(dd1['mN1']))) * Lumi * dd1['BRN1Zg'] * BR['SR-WZ']   * eff_WZ(np.abs(dd1['mN1']))

dd2     = dd1[dd1['S_WH'] > s95['SR-ZZ']]
dd4     = dd1[dd1['S_WZ'] > s95['SR-ZZ']]
dz2     = dd1[(dd1['S_WH'] <= s95['SR-ZZ']) & (dd1['S_WZ'] <= s95['SR-ZZ']) ]

# sc = ax1.scatter(dz2['M1'], dz2['Mu'] , marker='.', s=1.0, c=dz2['LogL'], cmap="Spectral_r", vmax=3.8, vmin=-3.8, zorder=100)


# Wino-Dominated N1 
# Case SM-A 
de1 = df[(df["N12"] ** 2 > df["N13"] ** 2 + df["N14"] ** 2) & (df['N12'] ** 2 > df['N11'] ** 2) & (df['WN2'] > df['WN2Gamma'] + df['WN2H'] + df['WN2Z']) ]
de1 = de1.sort_values(['LogL'])

de1['BRN1Hg'] = de1['WN1H'] / (de1['WN1Gamma'] + de1['WN1H'] + de1['WN1Z'])
de1['BRN1Zg'] = de1['WN1Z'] / (de1['WN1Gamma'] + de1['WN1H'] + de1['WN1Z'])

de1['S_HH']         = 3.5 * np.exp(XSect_SMb(np.abs(de1['mN1']))) * Lumi * de1['BRN1Hg'] * de1['BRN1Hg'] * BR['SR-HH'] * eff_HH(np.abs(de1['mN1']))
de1['S_ZH']         = 3.5 * np.exp(XSect_SMb(np.abs(de1['mN1']))) * Lumi * de1['BRN1Hg'] * de1['BRN1Zg'] * BR['SR-ZH'] * eff_ZH(np.abs(de1['mN1']))
de1['S_ZZ']         = 3.5 * np.exp(XSect_SMb(np.abs(de1['mN1']))) * Lumi * de1['BRN1Zg'] * de1['BRN1Zg'] * BR['SR-ZZ'] * eff_ZZ(np.abs(de1['mN1']))

de2     = de1[de1['S_HH'] > s95['SR-HH']]
de3     = de1[de1['S_ZH'] > s95['SR-ZH']]
de4     = de1[de1['S_ZZ'] > s95['SR-ZZ']]
dz3     = de1[(de1['S_HH'] <= s95['SR-HH']) & (de1['S_ZH'] <= s95['SR-ZH']) & (de1['S_ZZ'] <= s95['SR-ZZ']) ]



# Case SM-B 
dg1 = df[(df["N12"] ** 2 > df["N13"] ** 2 + df["N14"] ** 2) & (df['N12'] ** 2 > df['N11'] ** 2) & (df['WN2'] <= df['WN2Gamma'] + df['WN2H'] + df['WN2Z']) ]
dg1 = dg1.sort_values(['LogL'])

dg1['BRN1Hg'] = dg1['WN1H'] / (dg1['WN1Gamma'] + dg1['WN1H'] + dg1['WN1Z'])
dg1['BRN1Zg'] = dg1['WN1Z'] / (dg1['WN1Gamma'] + dg1['WN1H'] + dg1['WN1Z'])

dg1['S_WH']    = 3.5 * np.exp(XSect_SMb(np.abs(dg1['mN1']))) * Lumi * dg1['BRN1Hg'] * BR['SR-WH']   * eff_WH(np.abs(dg1['mN1']))
dg1['S_WZ']    = 3.5 * np.exp(XSect_SMb(np.abs(dg1['mN1']))) * Lumi * dg1['BRN1Zg'] * BR['SR-WZ']   * eff_WZ(np.abs(dg1['mN1']))

dg2     = dg1[dg1['S_WH'] > s95['SR-ZZ']]
dg4     = dg1[dg1['S_WZ'] > s95['SR-ZZ']]
dz4     = dg1[(dg1['S_WH'] <= s95['SR-ZZ']) & (dg1['S_WZ'] <= s95['SR-ZZ']) ]

# Bino-Dominated Case 
# Assumed no limits at All 
dz5 = df[(df["N11"] ** 2 > df["N13"] ** 2 + df["N14"] ** 2) & (df['N11'] ** 2 > df['N12'] ** 2)]
dz5 = dz5.sort_values(['LogL'])

# =============== Plot M1-Mu plane ================ # 

# ax1.scatter(dc2['M1'], dc2['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
# ax1.scatter(dc3['M1'], dc3['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
# ax1.scatter(dc4['M1'], dc4['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
# ax1.scatter(dd2['M1'], dd2['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
# ax1.scatter(dd4['M1'], dd4['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
# ax1.scatter(dg2['M1'], dg2['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
# ax1.scatter(dg4['M1'], dg4['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
# ax1.scatter(de2['M1'], de2['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
# ax1.scatter(de3['M1'], de3['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
# ax1.scatter(de4['M1'], de4['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)

# sc = ax1.scatter(dz1['M1'], dz1['Mu'] , marker='.', s=4.0, c=dz1['LogL'], cmap="Spectral_r", vmax=3.8, vmin=-3.8)
# ax1.scatter(dz2['M1'], dz2['Mu'] , marker='.', s=4.0, c=dz2['LogL'], cmap="Spectral_r", vmax=3.8, vmin=-3.8)
# ax1.scatter(dz3['M1'], dz3['Mu'] , marker='.', s=4.0, c=dz3['LogL'], cmap="Spectral_r", vmax=3.8, vmin=-3.8)
# ax1.scatter(dz4['M1'], dz4['Mu'] , marker='.', s=4.0, c=dz4['LogL'], cmap="Spectral_r", vmax=3.8, vmin=-3.8)
# ax1.scatter(dz5['M1'], dz5['Mu'] , marker='.', s=4.0, c=dz5['LogL'], cmap="Spectral_r", vmax=3.8, vmin=-3.8)

# ax1.plot([-1000, 1000], [-1000, 1000], ':', linewidth=1.4, c='grey')
# ax1.plot([-1000, 1000], [1000, -1000], ':', linewidth=1.4, c='grey')

# # sc = ax1.scatter(df['M1'], df['Mu'] , marker='.', s=4.0, c=df['LogL'], cmap="Spectral_r", vmax=3.8, vmin=-3.8)

# ax1.set_xlim(-1000., 1000)
# ax1.set_ylim(-1000., 1000)
# ax1.yaxis.set_minor_locator(AutoMinorLocator())
# ax1.xaxis.set_minor_locator(AutoMinorLocator())


# ax1.tick_params(
#     which='both',
#     direction="in",
#     labelsize=18,
#     left=True,
#     right=True,
#     bottom=True,
#     top=True
# )
# ax1.tick_params(which="major", length=10, width=1.2)
# ax1.tick_params(which="minor", length=4, width=1.2)
# ax1.set_xticks(np.linspace(-1000, 1000, 5))
# ax1.set_yticks(np.linspace(-1000, 1000, 5))

# axc.tick_params(
#     which='both',
#     direction="in",
#     labelsize=18,
#     left=False,
#     right=True,
#     bottom=False,
#     top=False
# )
# axc.tick_params(which="major", length=7, width=1.2)
# axc.tick_params(which="minor", length=4, width=1.2)
# ax1.set_xlabel("$M_{1}$ [GeV]", fontsize=30, loc="right")
# ax1.set_ylabel("$\mu$ [GeV]", fontsize=30, loc="top")

# plt.colorbar(sc, axc)
# axc.yaxis.set_minor_locator(AutoMinorLocator())
# axc.set_ylabel(r"$\ln\mathcal{L}^{\rm searches}$", fontsize=30)


# plt.savefig("m1-mu-HLLHC.png", dpi=150)

# =============== Plot M2-Mu plane ================ # 

ax1.scatter(dc2['M2'], dc2['Mu'] , marker='.', s=1.0, c="lightgray")
ax1.scatter(dc3['M2'], dc3['Mu'] , marker='.', s=1.0, c="lightgray")
ax1.scatter(dc4['M2'], dc4['Mu'] , marker='.', s=1.0, c="lightgray")
ax1.scatter(dd2['M2'], dd2['Mu'] , marker='.', s=1.0, c="lightgray")
ax1.scatter(dd4['M2'], dd4['Mu'] , marker='.', s=1.0, c="lightgray")
ax1.scatter(dg2['M2'], dg2['Mu'] , marker='.', s=1.0, c="lightgray")
ax1.scatter(dg4['M2'], dg4['Mu'] , marker='.', s=1.0, c="lightgray")
ax1.scatter(de2['M2'], de2['Mu'] , marker='.', s=1.0, c="lightgray")
ax1.scatter(de3['M2'], de3['Mu'] , marker='.', s=1.0, c="lightgray")
ax1.scatter(de4['M2'], de4['Mu'] , marker='.', s=1.0, c="lightgray")

sc = ax1.scatter(dz1['M2'], dz1['Mu'] , marker='.', s=4.0, c=dz1['LogL'], cmap="Spectral_r", vmax=3.8, vmin=-3.8)
ax1.scatter(dz2['M2'], dz2['Mu'] , marker='.', s=4.0, c=dz2['LogL'], cmap="Spectral_r", vmax=3.8, vmin=-3.8)
ax1.scatter(dz3['M2'], dz3['Mu'] , marker='.', s=4.0, c=dz3['LogL'], cmap="Spectral_r", vmax=3.8, vmin=-3.8)
ax1.scatter(dz4['M2'], dz4['Mu'] , marker='.', s=4.0, c=dz4['LogL'], cmap="Spectral_r", vmax=3.8, vmin=-3.8)
ax1.scatter(dz5['M2'], dz5['Mu'] , marker='.', s=4.0, c=dz5['LogL'], cmap="Spectral_r", vmax=3.8, vmin=-3.8)

# sc = ax1.scatter(df['M2'], df['Mu'] , marker='.', s=4.0, c=df['LogL'], cmap="Spectral_r", vmax=3.8, vmin=-3.8)

ax1.plot([-1000, 1000], [-1000, 1000], ':', linewidth=1.4, c='grey')
ax1.plot([-1000, 1000], [1000, -1000], ':', linewidth=1.4, c='grey')

ax1.set_xlim(0., 1000)
ax1.set_ylim(-1000., 1000)
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.xaxis.set_minor_locator(AutoMinorLocator())


ax1.tick_params(
    which='both',
    direction="in",
    labelsize=18,
    left=True,
    right=True,
    bottom=True,
    top=True
)
ax1.tick_params(which="major", length=10, width=1.2)
ax1.tick_params(which="minor", length=4, width=1.2)
ax1.set_xticks(np.linspace(0, 1000, 5))
ax1.set_yticks(np.linspace(-1000, 1000, 5))

axc.tick_params(
    which='both',
    direction="in",
    labelsize=18,
    left=False,
    right=True,
    bottom=False,
    top=False
)
axc.tick_params(which="major", length=7, width=1.2)
axc.tick_params(which="minor", length=4, width=1.2)
ax1.set_xlabel("$M_{2}$ [GeV]", fontsize=30, loc="right")
ax1.set_ylabel("$\mu$ [GeV]", fontsize=30, loc="top")

plt.colorbar(sc, axc)
axc.yaxis.set_minor_locator(AutoMinorLocator())
axc.set_ylabel(r"$\ln\mathcal{L}^{\rm searches}$", fontsize=30)

plt.savefig("m2-mu-HLLHC.png", dpi=150)
# plt.savefig("m2-mu.png", dpi=150)

# =================== Plot mN1-Decay Width ====================

# ax1.set_xlim(0., 1000)
# # ax1.set_ylim(1.0e-21, 1.0e-14)
# axc.set_yscale("log")

# ax1.yaxis.set_minor_locator(AutoMinorLocator())
# ax1.xaxis.set_minor_locator(AutoMinorLocator())


# ax1.tick_params(
#     which='both',
#     direction="in",
#     labelsize=18,
#     left=True,
#     right=True,
#     bottom=True,
#     top=True
# )
# ax1.tick_params(which="major", length=10, width=1.2)
# ax1.tick_params(which="minor", length=4, width=1.2)
# ax1.set_xticks(np.linspace(0, 1000, 5))
# # ax1.set_yticks(np.linspace(-1000, 1000, 5))

# axc.tick_params(
#     which='both',
#     direction="in",
#     labelsize=18,
#     left=False,
#     right=True,
#     bottom=False,
#     top=False
# )
# axc.tick_params(which="major", length=7, width=1.2)
# axc.tick_params(which="minor", length=4, width=1.2)
# ax1.set_xlabel(r"$m_{\tilde{\chi}_1^0}$ [GeV]", fontsize=30, loc="right")
# ax1.set_ylabel(r"$\Gamma_{\tilde{\chi}_2^0}$ [GeV]", fontsize=30, loc="top")

# sc = ax1.scatter(np.abs(df['mN1']), df['WN2'] , marker='.', s=8.0, c=np.abs(df['mN2']) - np.abs(df['mN1']), cmap="tab20c", norm=matplotlib.colors.LogNorm(vmin=1.e-4, vmax=10))
# plt.colorbar(sc, axc)
# axc.set_ylabel(r"$m_{\tilde{\chi}_2^0} - m_{\tilde{\chi}_1^0}$ [GeV]", fontsize=30, loc="top")
# ax1.set_yscale('log')

plt.show()
# plt.savefig("LLP.png", dpi=300)




