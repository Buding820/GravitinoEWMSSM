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
from scipy.interpolate import interp1d

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


df_sma     = pd.read_csv("Data/XSect_SMa.csv")
df_smb     = pd.read_csv("Data/XSect_SMb.csv")
df_fit     = pd.read_csv("Data/SR-limit.csv")

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

sr_lim = df_fit.loc[df_fit['SR'] == "SR-WZ"]['exp']
print(sr_lim)

def upper_lim(sr, kk):
    return df_fit.loc[df_fit['SR'] == sr][kk]

fig = plt.figure(figsize=(10, 8))
ax  = fig.add_axes([0.14, 0.13, 0.82, 0.84])

mH = df_sma['mHiggsino']
xs = df_sma['XSect']
# print(eft)

# ax.plot(mH, xs, '-', color="r")

eft      = np.array(df_sma['eff_HH']) 
xs_upp0  = float(upper_lim("SR-HH", "exp"))        / ( Lumi * BR['SR-HH'] * eft)
xs_upp1p = float(upper_lim("SR-HH", "exp+1sigma")) / ( Lumi * BR['SR-HH'] * eft )
xs_upp1m = float(upper_lim("SR-HH", "exp-1sigma")) / ( Lumi * BR['SR-HH'] * eft )
xs_upp2p = float(upper_lim("SR-HH", "exp+2sigma")) / ( Lumi * BR['SR-HH'] * eft )
xs_upp2m = float(upper_lim("SR-HH", "exp-2sigma")) / ( Lumi * BR['SR-HH'] * eft )

dfxs = interp1d(np.linspace(0., 1., mH.shape[0]), np.log(xs), kind='cubic')
dfexp = interp1d(np.linspace(0., 1., mH.shape[0]), np.log(xs_upp0), kind="cubic")

mashgrid = pd.DataFrame(
    index=np.linspace(0., 1., 50),
    columns=np.linspace(0., 1., 50)
).unstack().reset_index().rename(columns={'level_0': 'yy', 'level_1': 'xx', 0: 'z'})
mashgrid['z'] =  np.exp(dfxs(mashgrid['xx'])) * (1 - mashgrid['yy'])**2 /  np.exp(dfexp(mashgrid['xx'])) 
print(mashgrid)

from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
plottri_zz = Triangulation(mashgrid['xx'], mashgrid['yy'])
refiner_zz = UniformTriRefiner(plottri_zz)
tri_refine_zz, zz_refine = refiner_zz.refine_field(mashgrid['z'], subdiv=3)
# fig = plt.figure(figsize=(10,8))
# ax = fig.add_axes([0.16, 0.16, 0.82, 0.82])
ax.tricontourf(tri_refine_zz, zz_refine, 
               levels=[
                    float(upper_lim("SR-HH", "exp-1sigma")) /float(upper_lim("SR-HH", "exp")), 
                    float(upper_lim("SR-HH", "exp+1sigma")) /float(upper_lim("SR-HH", "exp"))
                    ], 
               colors=[ "#0096ff",],
               alpha=0.4,
               transform=ax.transAxes)
ax.tricontour(tri_refine_zz, zz_refine, 
               levels=[1.0], linewidths=2.4, 
               colors=["#0096ff"],
               transform=ax.transAxes)


eft      = np.array(df_sma['eff_ZH']) 
xs_upp0  = float(upper_lim("SR-ZH", "exp"))        / ( Lumi * BR['SR-ZH'] * eft)
xs_upp1p = float(upper_lim("SR-ZH", "exp+1sigma")) / ( Lumi * BR['SR-ZH'] * eft )
xs_upp1m = float(upper_lim("SR-ZH", "exp-1sigma")) / ( Lumi * BR['SR-ZH'] * eft )
xs_upp2p = float(upper_lim("SR-ZH", "exp+2sigma")) / ( Lumi * BR['SR-ZH'] * eft )
xs_upp2m = float(upper_lim("SR-ZH", "exp-2sigma")) / ( Lumi * BR['SR-ZH'] * eft )

# dfxs = interp1d(np.linspace(0., 1., mH.shape[0]), np.log(xs), kind='cubic')
dfexp = interp1d(np.linspace(0., 1., mH.shape[0]), np.log(xs_upp0), kind="cubic")

mashgrid = pd.DataFrame(
    index=np.linspace(0., 1., 50),
    columns=np.linspace(0., 1., 50)
).unstack().reset_index().rename(columns={'level_0': 'yy', 'level_1': 'xx', 0: 'z'})
mashgrid['z'] =  np.exp(dfxs(mashgrid['xx'])) * 4.0 * mashgrid['yy'] * (1 - mashgrid['yy']) /  np.exp(dfexp(mashgrid['xx'])) 
print(mashgrid)

# from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
plottri_zz = Triangulation(mashgrid['xx'], mashgrid['yy'])
refiner_zz = UniformTriRefiner(plottri_zz)
tri_refine_zz, zz_refine = refiner_zz.refine_field(mashgrid['z'], subdiv=3)
# fig = plt.figure(figsize=(10,8))
# ax = fig.add_axes([0.16, 0.16, 0.82, 0.82])
ax.tricontourf(tri_refine_zz, zz_refine, 
               levels=[
                    float(upper_lim("SR-HH", "exp-1sigma")) /float(upper_lim("SR-HH", "exp")), 
                    float(upper_lim("SR-HH", "exp+1sigma")) /float(upper_lim("SR-HH", "exp"))
                    ], 
               colors=["#008f00"],
               alpha=0.4,
               transform=ax.transAxes)
ax.tricontour(tri_refine_zz, zz_refine, 
               levels=[1.0], linewidths=2.4, 
               colors=["#008f00"],
               transform=ax.transAxes)


eft      = np.array(df_sma['eff_ZZ']) 
xs_upp0  = float(upper_lim("SR-ZZ", "exp"))        / ( Lumi * BR['SR-ZZ'] * eft)
xs_upp1p = float(upper_lim("SR-ZZ", "exp+1sigma")) / ( Lumi * BR['SR-ZZ'] * eft )
xs_upp1m = float(upper_lim("SR-ZZ", "exp-1sigma")) / ( Lumi * BR['SR-ZZ'] * eft )
xs_upp2p = float(upper_lim("SR-ZZ", "exp+2sigma")) / ( Lumi * BR['SR-ZZ'] * eft )
xs_upp2m = float(upper_lim("SR-ZZ", "exp-2sigma")) / ( Lumi * BR['SR-ZZ'] * eft )

dfexp = interp1d(np.linspace(0., 1., mH.shape[0]), np.log(xs_upp0), kind="cubic")

mashgrid = pd.DataFrame(
    index=np.linspace(0., 1., 50),
    columns=np.linspace(0., 1., 50)
).unstack().reset_index().rename(columns={'level_0': 'yy', 'level_1': 'xx', 0: 'z'})
mashgrid['z'] =  np.exp(dfxs(mashgrid['xx'])) * mashgrid['yy'] * mashgrid['yy'] /  np.exp(dfexp(mashgrid['xx'])) 
print(mashgrid)

# from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
plottri_zz = Triangulation(mashgrid['xx'], mashgrid['yy'])
refiner_zz = UniformTriRefiner(plottri_zz)
tri_refine_zz, zz_refine = refiner_zz.refine_field(mashgrid['z'], subdiv=3)
# fig = plt.figure(figsize=(10,8))
# ax = fig.add_axes([0.16, 0.16, 0.82, 0.82])
ax.tricontourf(tri_refine_zz, zz_refine, 
               levels=[
                    float(upper_lim("SR-HH", "exp-1sigma")) /float(upper_lim("SR-HH", "exp")), 
                    float(upper_lim("SR-HH", "exp+1sigma")) /float(upper_lim("SR-HH", "exp"))
                    ], 
               colors=["#ff9300"],
               alpha=0.4,
               transform=ax.transAxes)
ax.tricontour(tri_refine_zz, zz_refine, 
               levels=[1.0], linewidths=2.4, 
               colors=["#ff9300"],
               transform=ax.transAxes)

ax.plot([0.58, 0.65], [0.26, 0.26], '-', color="#0096ff", transform=ax.transAxes)
ax.plot([0.58, 0.65], [0.20, 0.20], '-', color="#008f00", transform=ax.transAxes)
ax.plot([0.58, 0.65], [0.14, 0.14], '-', color="#ff9300", transform=ax.transAxes)
ax.fill_between([0.58, 0.65], [0.28, 0.28], [0.24, 0.24], fc="#0096ff", alpha=0.4, transform=ax.transAxes)
ax.fill_between([0.58, 0.65], [0.22, 0.22], [0.18, 0.18], fc="#008f00", alpha=0.4, transform=ax.transAxes)
ax.fill_between([0.58, 0.65], [0.12, 0.12], [0.16, 0.16], fc="#ff9300", alpha=0.4, transform=ax.transAxes)


ax.text(0.67, 0.26, r"HH: Exp. Limit & $\pm 1\sigma$", transform=ax.transAxes, ha="left", va='center')
ax.text(0.67, 0.20, r"ZH: Exp. Limit & $\pm 1\sigma$", transform=ax.transAxes, ha="left", va='center')
ax.text(0.67, 0.14, r"ZZ: Exp. Limit & $\pm 1\sigma$", transform=ax.transAxes, ha="left", va='center')

ax.set_xlabel(r"$m_{\tilde{H}}~[{\rm GeV}]$", fontsize=30, loc='right')
ax.set_ylabel(r"${\rm BR}(\tilde{\chi}_1^0 \to Z \tilde{G})$", fontsize=30, loc='top')

ax.set_xlim(150, 1000)
ax.set_ylim(0, 1)

ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())


ax.tick_params(
    which='both',
    direction="in",
    labelsize=18,
    left=True,
    right=True,
    bottom=True,
    top=True
)
ax.tick_params(which="major", length=10, width=1.2)
ax.tick_params(which="minor", length=4, width=1.2)

# plt.show()
plt.savefig("sma_lim.png", dpi=300)



