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
    "SR-WH-High":   5580.7,
    "SR-WH-Low":    295821.0,
    "SR-WZ":        480.2
}

BR = {
    "SR-HH":        2.0 * brH2bb * brH2tt,
    "SR-ZH":        0.5 * brH2bb * brZ2ll,
    "SR-ZZ":        brZ2ll * brZ2ll,
    "SR-WH-High":   brW2lv * brH2bb,
    "SR-WH-Low":    brW2lv * brH2bb,
    "SR-WZ":        brW2lv * brZ2ll
}

df_sma     = pd.read_csv("Data/XSect_SMa.csv")

# xsect_sma['eff_HH']         = xsect_sma['SR-HH'] / ( xsect_sma['XSect'] * Lumi * 2.0 * brH2bb * brH2tt ) 
# xsect_sma['eff_ZH']         = xsect_sma['SR-ZH'] / ( xsect_sma['XSect'] * Lumi * 0.5 * brH2bb * brZ2ll )
# xsect_sma['eff_ZZ']         = xsect_sma['SR-ZZ'] / ( xsect_sma['XSect'] * Lumi * brZ2ll * brZ2ll )

# xsect_sma.to_csv("Data/XSect_SMa.csv", index=None)

df_smb     = pd.read_csv("Data/XSect_SMb.csv")

# xsect_smb['eff_WH-High']    = S95(N_SM['SR-WH-High']) / (xsect_smb['SR-WH-High_95up'] * Lumi * brW2lv * brH2bb)
# xsect_smb['eff_WH-Low']     = S95(N_SM['SR-WH-Low']) / (xsect_smb['SR-WH-Low_95up'] * Lumi * brW2lv * brH2bb)
# xsect_smb['eff_WZ']         = S95(N_SM['SR-WZ']) / (xsect_smb['SR-WZ_95up'] * Lumi * brW2lv * brZ2ll)



# df_smb.to_csv("Data/XSect_SMb.csv", index=None)

from scipy.interpolate import interp1d
XSect_SMa   = interp1d(df_sma['mHiggsino'], np.log(df_sma['XSect']), kind="cubic", fill_value="extrapolate")
eff_HH      = interp1d(df_sma['mHiggsino'], df_sma['eff_HH'], kind="cubic", fill_value="extrapolate")
eff_ZH      = interp1d(df_sma['mHiggsino'], df_sma['eff_ZH'], kind="cubic", fill_value="extrapolate")
eff_ZZ      = interp1d(df_sma['mHiggsino'], df_sma['eff_ZZ'], kind="cubic", fill_value="extrapolate")

XSect_SMb   = interp1d(df_smb['mHiggsino'], np.log(df_smb['XSect']), kind="cubic", fill_value="extrapolate")
eff_WHH     = interp1d(df_smb['mHiggsino'], df_smb['eff_WH-High'], kind="cubic", fill_value="extrapolate")
eff_WHL     = interp1d(df_smb['mHiggsino'], df_smb['eff_WH-Low'], kind="cubic", fill_value="extrapolate")
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

dc2     = dc1[dc1['S_HH'] > S95(N_SM['SR-HH'])]
dc3     = dc1[dc1['S_ZH'] > S95(N_SM['SR-ZH'])]
dc4     = dc1[dc1['S_ZZ'] > S95(N_SM['SR-ZZ'])]
dz1     = dc1[(dc1['S_HH'] <= S95(N_SM['SR-HH'])) & (dc1['S_ZH'] <= S95(N_SM['SR-ZH'])) & (dc1['S_ZZ'] <= S95(N_SM['SR-ZZ'])) ]

# Case SM-B
dd1 = df[(df["N13"] ** 2 + df["N14"] ** 2 > df["N11"] ** 2) & (df["N13"] ** 2 + df["N14"] ** 2 > df["N12"] ** 2) & (df['WN2'] <= df['WN2Gamma'] + df['WN2H'] + df['WN2Z']) ]
dd1 = dd1.sort_values(['LogL'])

dd1['BRN1Hg'] = dd1['WN1H'] / (dd1['WN1Gamma'] + dd1['WN1H'] + dd1['WN1Z'])
dd1['BRN1Zg'] = dd1['WN1Z'] / (dd1['WN1Gamma'] + dd1['WN1H'] + dd1['WN1Z'])

dd1['S_WH-High']    = np.exp(XSect_SMb(np.abs(dd1['mN1']))) * Lumi * dd1['BRN1Hg'] * BR['SR-WH-High']   * eff_WHH(np.abs(dd1['mN1']))
dd1['S_WH-Low']     = np.exp(XSect_SMb(np.abs(dd1['mN1']))) * Lumi * dd1['BRN1Hg'] * BR['SR-WH-Low']    * eff_WHL(np.abs(dd1['mN1']))
dd1['S_WZ']         = np.exp(XSect_SMb(np.abs(dd1['mN1']))) * Lumi * dd1['BRN1Zg'] * BR['SR-WZ']        * eff_WZ(np.abs(dd1['mN1']))

dd2     = dd1[dd1['S_WH-High'] > S95(N_SM['SR-ZZ'])]
dd3     = dd1[dd1['S_WH-Low'] > S95(N_SM['SR-ZZ'])]
dd4     = dd1[dd1['S_WZ'] > S95(N_SM['SR-ZZ'])]
dz2     = dd1[(dd1['S_WH-High'] <= S95(N_SM['SR-ZZ'])) & (dd1['S_WH-Low'] <= S95(N_SM['SR-ZZ'])) & (dd1['S_WZ'] <= S95(N_SM['SR-ZZ'])) ]

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

de2     = de1[de1['S_HH'] > S95(N_SM['SR-HH'])]
de3     = de1[de1['S_ZH'] > S95(N_SM['SR-ZH'])]
de4     = de1[de1['S_ZZ'] > S95(N_SM['SR-ZZ'])]
dz3     = de1[(de1['S_HH'] <= S95(N_SM['SR-HH'])) & (de1['S_ZH'] <= S95(N_SM['SR-ZH'])) & (de1['S_ZZ'] <= S95(N_SM['SR-ZZ'])) ]



# Case SM-B 
dg1 = df[(df["N12"] ** 2 > df["N13"] ** 2 + df["N14"] ** 2) & (df['N12'] ** 2 > df['N11'] ** 2) & (df['WN2'] <= df['WN2Gamma'] + df['WN2H'] + df['WN2Z']) ]
dg1 = dg1.sort_values(['LogL'])

dg1['BRN1Hg'] = dg1['WN1H'] / (dg1['WN1Gamma'] + dg1['WN1H'] + dg1['WN1Z'])
dg1['BRN1Zg'] = dg1['WN1Z'] / (dg1['WN1Gamma'] + dg1['WN1H'] + dg1['WN1Z'])

dg1['S_WH-High']    = 3.5 * np.exp(XSect_SMb(np.abs(dg1['mN1']))) * Lumi * dg1['BRN1Hg'] * BR['SR-WH-High']   * eff_WHH(np.abs(dg1['mN1']))
dg1['S_WH-Low']     = 3.5 * np.exp(XSect_SMb(np.abs(dg1['mN1']))) * Lumi * dg1['BRN1Hg'] * BR['SR-WH-Low']    * eff_WHL(np.abs(dg1['mN1']))
dg1['S_WZ']         = 3.5 * np.exp(XSect_SMb(np.abs(dg1['mN1']))) * Lumi * dg1['BRN1Zg'] * BR['SR-WZ']        * eff_WZ(np.abs(dg1['mN1']))

dg2     = dg1[dg1['S_WH-High'] > S95(N_SM['SR-ZZ'])]
dg3     = dg1[dg1['S_WH-Low'] > S95(N_SM['SR-ZZ'])]
dg4     = dg1[dg1['S_WZ'] > S95(N_SM['SR-ZZ'])]
dz4     = dg1[(dg1['S_WH-High'] <= S95(N_SM['SR-ZZ'])) & (dg1['S_WH-Low'] <= S95(N_SM['SR-ZZ'])) & (dg1['S_WZ'] <= S95(N_SM['SR-ZZ'])) ]

# Bino-Dominated Case 
# Assumed no limits at All 
dz5 = df[(df["N11"] ** 2 > df["N13"] ** 2 + df["N14"] ** 2) & (df['N11'] ** 2 > df['N12'] ** 2)]
dz5 = dz5.sort_values(['LogL'])

# =============== Plot M1-Mu plane ================ # 

# ax1.scatter(dc2['M1'], dc2['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
# ax1.scatter(dc3['M1'], dc3['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
# ax1.scatter(dc4['M1'], dc4['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
# ax1.scatter(dd2['M1'], dd2['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
# ax1.scatter(dd3['M1'], dd3['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
# ax1.scatter(dd4['M1'], dd4['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
# ax1.scatter(dg2['M1'], dg2['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
# ax1.scatter(dg3['M1'], dg3['Mu'] , marker='.', s=1.0, c="lightgray", alpha=0.9)
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
ax1.scatter(dd3['M2'], dd3['Mu'] , marker='.', s=1.0, c="lightgray")
ax1.scatter(dd4['M2'], dd4['Mu'] , marker='.', s=1.0, c="lightgray")
ax1.scatter(dg2['M2'], dg2['Mu'] , marker='.', s=1.0, c="lightgray")
ax1.scatter(dg3['M2'], dg3['Mu'] , marker='.', s=1.0, c="lightgray")
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



# plt.show()




