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
    "SR-WH":        4028.2,
    "SR-WZ":        480.2
}

BR = {
    "SR-HH":        2.0 * brH2bb * brH2tt,
    "SR-ZH":        0.5 * brH2bb * brZ2ll,
    "SR-ZZ":        brZ2ll * brZ2ll,
    "SR-WH":        brW2lv * brH2bb,
    "SR-WZ":        brW2lv * brZ2ll
}
# df_smb['eff_WH']    = df_smb['SR-WH'] / (df_smb['XSect'] * Lumi * brW2lv * brH2bb)
# print(df_smb)
# df_smb.to_csv("Data/XSect_SMb.csv")


sr_lim = df_fit.loc[df_fit['SR'] == "SR-WZ"]['exp']
# print(sr_lim)

def upper_lim(sr, kk):
    return df_fit.loc[df_fit['SR'] == sr][kk]

fig = plt.figure(figsize=(10, 8))
ax  = fig.add_axes([0.14, 0.13, 0.82, 0.84])

mH = df_smb['mHiggsino']
xs = df_smb['XSect']
# print(eft)


# N_upp = float(upper_lim("SR-WH-Low", "exp"))

# eft = np.array(df_smb['eff_WH-Low']) 
# xs_upp0 =  float(upper_lim("SR-WH-Low", "exp")) / ( Lumi * BR['SR-WH-Low'] * eft)
# xs_upp1p = float(upper_lim("SR-WH-Low", "exp+1sigma")) / ( Lumi * BR['SR-WH-Low'] * eft )
# xs_upp1m = float(upper_lim("SR-WH-Low", "exp-1sigma")) / ( Lumi * BR['SR-WH-Low'] * eft )
# xs_upp2p = float(upper_lim("SR-WH-Low", "exp+2sigma")) / ( Lumi * BR['SR-WH-Low'] * eft )
# xs_upp2m = float(upper_lim("SR-WH-Low", "exp-2sigma")) / ( Lumi * BR['SR-WH-Low'] * eft )
# # print(xs_upp)
cteq = pd.read_csv("Data/pdf_cteq66.csv")
mstw = pd.read_csv("Data/pdf_MSTW2008nlo90cl.csv")
xx = cteq['mass']


xscsump =  cteq['N1C1p'] + cteq['N1C1m'] 
xscsumm =  mstw['N1C1p'] + mstw['N1C1m'] 
yyitp = np.log(xscsump)
yyitm = np.log(xscsumm)
xscp = interp1d(xx, yyitp, kind="cubic")
xscm = interp1d(xx, yyitm, kind="cubic")
xms = np.linspace(100, 1000, 200)
ax.fill_between(xms, np.exp(xscp(xms)), np.exp(xscm(xms)), color='#055C9D', alpha=1.0, zorder=10 )
ax.plot(xms, np.exp((xscp(xms)+xscm(xms))/2.0), '-', c='#055C9D', linewidth=1.3, zorder=10)

ax.plot([0.45, 0.52], [0.95, 0.95], "-", c='#055C9D', linewidth=2.4, transform=ax.transAxes )
ax.text(0.54, 0.95, "Theory", ha='left', va='center', transform=ax.transAxes )


# ax.plot(mH, xs, '-', c="#065c9d", zorder=10)
# ax.plot(mH, xs_upp0, ':', color="r")
# ax.fill_between(mH, xs_upp2p, xs_upp2m, fc="yellow", alpha=0.5)
# ax.fill_between(mH, xs_upp1p, xs_upp1m, fc="green")

eft = np.array(df_smb['eff_WH']) 
xs_upp0 =  float(upper_lim("SR-WH", "exp")) / ( Lumi * BR['SR-WH'] * eft)
xs_upp1p = float(upper_lim("SR-WH", "exp+1sigma")) / ( Lumi * BR['SR-WH'] * eft )
xs_upp1m = float(upper_lim("SR-WH", "exp-1sigma")) / ( Lumi * BR['SR-WH'] * eft )
xs_upp2p = float(upper_lim("SR-WH", "exp+2sigma")) / ( Lumi * BR['SR-WH'] * eft )
xs_upp2m = float(upper_lim("SR-WH", "exp-2sigma")) / ( Lumi * BR['SR-WH'] * eft )

ax.plot(mH, xs_upp0, '--', color="#C2255C", lw=2.4, zorder=11)
ax.fill_between(mH, xs_upp2p, xs_upp2m, fc="#fffc79")
ax.fill_between(mH, xs_upp1p, xs_upp1m, fc="#8efa00")
ax.plot(mH, xs_upp2m, '-', color="#FFE066", lw=0.8, zorder=10)
ax.plot(mH, xs_upp2p, '-', color="#FFE066", lw=0.8, zorder=10)
ax.plot(mH, xs_upp1m, '-', color="#00f900", lw=0.8, zorder=10)
ax.plot(mH, xs_upp1p, '-', color="#00f900", lw=0.8, zorder=10)

ax.plot([0.45, 0.52], [0.75, 0.75], '--', color="#C2255C", lw=2.4, transform=ax.transAxes )
ax.fill_between([0.45, 0.52], [0.72, 0.72], [0.78, 0.78], fc="#fffc79", transform=ax.transAxes )
ax.fill_between([0.45, 0.52], [0.735, 0.735], [0.765, 0.765], fc="#8efa00", transform=ax.transAxes )
ax.text(0.54, 0.75, "WH: Exp. Limit & $\pm 1\sigma, \pm 2\sigma$", ha='left', va='center', transform=ax.transAxes )


eft = np.array(df_smb['eff_WZ']) 
xs_upp0 =  float(upper_lim("SR-WZ", "exp")) / ( Lumi * BR['SR-WZ'] * eft)
xs_upp1p = float(upper_lim("SR-WZ", "exp+1sigma")) / ( Lumi * BR['SR-WZ'] * eft )
xs_upp1m = float(upper_lim("SR-WZ", "exp-1sigma")) / ( Lumi * BR['SR-WZ'] * eft )
xs_upp2p = float(upper_lim("SR-WZ", "exp+2sigma")) / ( Lumi * BR['SR-WZ'] * eft )
xs_upp2m = float(upper_lim("SR-WZ", "exp-2sigma")) / ( Lumi * BR['SR-WZ'] * eft )

ax.plot(mH, xs_upp0, '--', color="#5F3DC4", lw=2.4, zorder=10)
ax.fill_between(mH, xs_upp2p, xs_upp2m, fc="#ffcc33")
ax.fill_between(mH, xs_upp1p, xs_upp1m, fc="#4bd9da")

ax.plot([0.45, 0.52], [0.85, 0.85], '--', color="#5F3DC4", lw=2.4, transform=ax.transAxes )
ax.fill_between([0.45, 0.52], [0.82, 0.82], [0.88, 0.88], fc="#ffcc33", transform=ax.transAxes )
ax.fill_between([0.45, 0.52], [0.835, 0.835], [0.865, 0.865], fc="#4bd9da", transform=ax.transAxes )
ax.text(0.54, 0.85, "WZ: Exp. Limit & $\pm 1\sigma, \pm 2\sigma$", ha='left', va='center', transform=ax.transAxes )

ax.set_xlabel(r"$m_{\tilde{H}}~[{\rm GeV}]$", fontsize=30, loc='right')
ax.set_ylabel(r"95% C.L. upper limit on $\sigma(pp\to \tilde{H}\tilde{H})$ [pb]", fontsize=24, loc="top")

ax.set_xlim(150., 1000)
# ax1.set_ylim(-1000., 1000)
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
ax.set_yscale("log")
# ax.legend()


# plt.show()
plt.savefig("smb_exp.png", dpi=300)

