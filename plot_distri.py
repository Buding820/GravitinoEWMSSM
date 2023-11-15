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

pwd = os.path.abspath(os.path.dirname(__file__))

def plot(cs, info):
    Lumi = 3000. 
    for var, inf in info.items():
        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_axes([0.15, 0.16, 0.81, 0.81])

        xrgs = inf['xrgs']
        nbin = inf['nbin']

        for kk, bkg in cs['Background'].items():
            pdf = []
            nn = 0.
            print(bkg.keys())
            for item in bkg['dfs']:
                df = pd.read_csv(os.path.join(cs['path']['pwd'], item['df']))
                df['wgh'] =  (item['XSect'] * Lumi) / df.shape[0]
                nn += item['XSect'] * Lumi
                pdf.append(df)
            
            pdf = pd.concat(pdf)
            hist, bins = np.histogram(pdf[var], range=xrgs, bins=nbin, weights=pdf['wgh'] / nn)
            xx = np.array([bins[:-1], bins[1:]]).ravel(order="F")
            yy = np.array([hist, hist]).ravel(order="F")

            ax.plot(xx, yy, "-", linewidth=2.0, label=r"{}".format(bkg['Label']), c=bkg['color'])

        print("Background Loaded")
        ds = pd.read_csv(os.path.join(cs['path']['pwd'], cs['Signal']['df']))
        ds['wgh'] = 1.0 / ds.shape[0]

        hist, bins = np.histogram(ds[var], range=xrgs, bins=nbin, weights=ds['wgh'])
        xx = np.array([bins[:-1], bins[1:]]).ravel(order="F")
        yy = np.array([hist, hist]).ravel(order="F")

        ax.plot(xx, yy, "-", c='r', linewidth=3.0, label=r"{}".format(cs['Signal']["Label"]))

        ax.set_xlim(xrgs)
        ax.set_ylim(inf['yrgs'])
        ax.set_xlabel(r"${}$".format(inf['xlabel']), fontsize=36, loc='right')
        ax.set_ylabel(r"${}$".format(inf['ylabel']), fontsize=36, loc='top')
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(
            which='both',
            direction="in",
            labelsize=24,
            left=True,
            right=True,
            bottom=True,
            top=True
        )
        ax.tick_params(which="major", length=13, width=1.6)
        ax.tick_params(which="minor", length=8, width=1.6)

        # ax.set_yscale("logit")
        if inf['legend']:
            ax.legend(fontsize=30, ncols=2)
        plt.savefig("ZH_{}.png".format(var), dpi=150)
        # plt.show()
        # fig.clf()


if __name__ == "__main__":
    switch_HH = False
    switch_ZH = False
    switch_ZZ = True
    switch_WH = False
    switch_WZ = False
    if switch_HH:
        with open(os.path.join(pwd, "Data/SRHH.json")) as f1:
            ds = json.loads(f1.read())
            info = {
                "ET_miss":{
                    "xlabel":   "E_{\\rm T}^{\\rm miss}~[{\\rm GeV}]",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0., 500),
                    "yrgs": (0., 0.3),
                    "nbin": 50,
                    "legend":   False
                },
                "m_bb":{
                    "xlabel":   "m_{bb}~[{\\rm GeV}]",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0., 500),
                    "yrgs": (0., 0.25),
                    "nbin": 50,
                    "legend":   True
                },
                "deltaR_bb":{
                    "xlabel":   "\Delta R_{bb}",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0.4, 5.),
                    "yrgs": (0., 0.15),
                    "nbin": 50,
                    "legend":   False
                },
                "mT2_min":{
                    "xlabel":   "m_{\\rm T2}^{\\rm min}~[{\\rm GeV}]",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0., 800),
                    "yrgs": (0., 0.15),
                    "nbin": 50,
                    "legend":   False
                }
            } 
            plot(ds, info)

    if switch_ZH:
        with open(os.path.join(pwd, "Data/SRZH.json")) as f1:
            ds = json.loads(f1.read())
            info = {
                "ET_miss":{
                    "xlabel":   "E_{\\rm T}^{\\rm miss}~[{\\rm GeV}]",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0., 500),
                    "yrgs": (0., 0.3),
                    "nbin": 50,
                    "legend":   False
                },
                "m_ll":{
                    "xlabel":   "m_{\ell\ell}~[{\\rm GeV}]",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0., 500),
                    "yrgs": (0., 0.25),
                    "nbin": 50,
                    "legend":   True
                },
                "deltaR_ll":{
                    "xlabel":   "\Delta R_{\ell\ell}",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0.4, 5.),
                    "yrgs": (0., 0.15),
                    "nbin": 50,
                    "legend":   False
                },
                "mT2_min":{
                    "xlabel":   "m_{\\rm T2}^{\\rm min}~[{\\rm GeV}]",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0., 800),
                    "yrgs": (0., 0.15),
                    "nbin": 50,
                    "legend":   False
                }
            } 
            plot(ds, info)

    if switch_ZZ:
        with open(os.path.join(pwd, "Data/SRZZ.json")) as f1:
            ds = json.loads(f1.read())
            info = {
                "ET_miss":{
                    "xlabel":   "E_{\\rm T}^{\\rm miss}~[{\\rm GeV}]",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0., 300),
                    "yrgs": (0., 0.15),
                    "nbin": 50,
                    "legend":   True
                },
                "mll1":{
                    "xlabel":   "m_{\ell\ell}^{\\rm 1st}~[{\\rm GeV}]",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0., 150),
                    "yrgs": (0., 0.25),
                    "nbin": 50,
                    "legend":   False
                },
                "mll2":{
                    "xlabel":   "m_{\ell\ell}^{\\rm 2nd}~[\\rm GeV]",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0.4, 200),
                    "yrgs": (0., 0.15),
                    "nbin": 50,
                    "legend":   False
                }
            } 
            plot(ds, info)

    if switch_WH:
        with open(os.path.join(pwd, "Data/SRWH.json")) as f1:
            ds = json.loads(f1.read())
            info = {
                "ET_miss":{
                    "xlabel":   "E_{\\rm T}^{\\rm miss}~[{\\rm GeV}]",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0., 400),
                    "yrgs": (0., 0.15),
                    "nbin": 50,
                    "legend":   True
                },
                "m_bb":{
                    "xlabel":   "m_{\ell\ell}~[{\\rm GeV}]",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0., 300),
                    "yrgs": (0., 0.2),
                    "nbin": 50,
                    "legend":   False
                },
                "mCT":{
                    "xlabel":   "m_{\\rm CT}~{[\\rm GeV]}",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0.4, 400.),
                    "yrgs": (0., 0.1),
                    "nbin": 50,
                    "legend":   False
                },
                "mT":{
                    "xlabel":   "m_{\\rm T}~[{\\rm GeV}]",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0., 400),
                    "yrgs": (0., 0.2),
                    "nbin": 50,
                    "legend":   False
                }
            } 
            plot(ds, info)

    if switch_WZ:
        with open(os.path.join(pwd, "Data/SRWZ.json")) as f1:
            ds = json.loads(f1.read())
            info = {
                "ET_miss":{
                    "xlabel":   "E_{\\rm T}^{\\rm miss}~[{\\rm GeV}]",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0., 400),
                    "yrgs": (0., 0.3),
                    "nbin": 50,
                    "legend":   True
                },
                "m_ll":{
                    "xlabel":   "m_{\ell\ell}~[{\\rm GeV}]",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0., 200),
                    "yrgs": (0., 0.25),
                    "nbin": 50,
                    "legend":   False
                },
                "deltaR_ll":{
                    "xlabel":   "\Delta R_{\ell\ell}",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0., 5.),
                    "yrgs": (0., 0.15),
                    "nbin": 50,
                    "legend":   False
                },
                "mT_min":{
                    "xlabel":   "m_{\\rm T}^{\\rm min}~[{\\rm GeV}]",
                    "ylabel":   "{\\rm Events~fraction}",
                    "xrgs": (0., 250),
                    "yrgs": (0., 0.15),
                    "nbin": 50,
                    "legend":   False
                }
            } 
            plot(ds, info)