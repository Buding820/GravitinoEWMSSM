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
        plt.cla()
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

            ax.plot(xx, yy, "-", linewidth=1.4, label=r"{}".format(bkg['Label']), c=bkg['color'])

        ds = pd.read_csv(os.path.join(cs['path']['pwd'], cs['Signal']['df']))
        ds['wgh'] = 1.0 / ds.shape[0]

        hist, bins = np.histogram(ds[var], range=xrgs, bins=nbin, weights=ds['wgh'])
        xx = np.array([bins[:-1], bins[1:]]).ravel(order="F")
        yy = np.array([hist, hist]).ravel(order="F")

        ax.plot(xx, yy, "-", c='r', linewidth=2.4, label=r"{}".format(cs['Signal']["Label"]))

        ax.set_xlim(xrgs)
        ax.set_ylim(inf['yrgs'])
        ax.set_xlabel(r"${}$".format(inf['xlabel']), fontsize=30, loc='right')
        ax.set_ylabel(r"${}$".format(inf['ylabel']), fontsize=30, loc='top')
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

        # ax.set_yscale("logit")
        if inf['legond']:
            ax.legend(ncols=2, fontsize=24)
        # plt.show()
        plt.savefig("ZH_{}.png".format(var), dpi=150)

if __name__ == "__main__":
    # with open(os.path.join(pwd, "Data/SRHH.json")) as f1:
    #     ds = json.loads(f1.read())
    #     info = {
    #         "ET_miss":{
    #             "xlabel":   "E_{\\rm T}^{\\rm miss}~[{\\rm GeV}]",
    #             "ylabel":   "{\\rm Events~fraction}",
    #             "xrgs": (0., 500),
    #             "yrgs": (0., 0.3),
    #             "nbin": 50,
    #             "legond":   False
    #         },
    #         "m_bb":{
    #             "xlabel":   "m_{bb}~[{\\rm GeV}]",
    #             "ylabel":   "{\\rm Events~fraction}",
    #             "xrgs": (0., 500),
    #             "yrgs": (0., 0.25),
    #             "nbin": 50,
    #             "legond":   True
    #         },
    #         "deltaR_bb":{
    #             "xlabel":   "\Delta R_{bb}",
    #             "ylabel":   "{\\rm Events~fraction}",
    #             "xrgs": (0.4, 5.),
    #             "yrgs": (0., 0.15),
    #             "nbin": 50,
    #             "legond":   False
    #         },
    #         "mT2_min":{
    #             "xlabel":   "m_{\\rm T2}^{\\rm min}~[{\\rm GeV}]",
    #             "ylabel":   "{\\rm Events~fraction}",
    #             "xrgs": (0., 800),
    #             "yrgs": (0., 0.15),
    #             "nbin": 50,
    #             "legond":   False
    #         }
    #     } 
    #     plot(ds, info)

    with open(os.path.join(pwd, "Data/SRHH.json")) as f1:
        ds = json.loads(f1.read())
        info = {
            "ET_miss":{
                "xlabel":   "E_{\\rm T}^{\\rm miss}~[{\\rm GeV}]",
                "ylabel":   "{\\rm Events~fraction}",
                "xrgs": (0., 500),
                "yrgs": (0., 0.3),
                "nbin": 50,
                "legond":   False
            },
            "m_bb":{
                "xlabel":   "m_{bb}~[{\\rm GeV}]",
                "ylabel":   "{\\rm Events~fraction}",
                "xrgs": (0., 500),
                "yrgs": (0., 0.25),
                "nbin": 50,
                "legond":   True
            },
            "deltaR_bb":{
                "xlabel":   "\Delta R_{bb}",
                "ylabel":   "{\\rm Events~fraction}",
                "xrgs": (0.4, 5.),
                "yrgs": (0., 0.15),
                "nbin": 50,
                "legond":   False
            },
            "mT2_min":{
                "xlabel":   "m_{\\rm T2}^{\\rm min}~[{\\rm GeV}]",
                "ylabel":   "{\\rm Events~fraction}",
                "xrgs": (0., 800),
                "yrgs": (0., 0.15),
                "nbin": 50,
                "legond":   False
            }
        } 
        plot(ds, info)