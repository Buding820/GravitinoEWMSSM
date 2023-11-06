#!/usr/bin/env python3
import json
import os, sys 
sys.path.append("/home/buding/Heptools/checkmate2/tools/python")
import matplotlib.pyplot as plt
import numpy as np
import pyhf
from pyhf.contrib.viz import brazil

bkg = [100.2]
bkg_uncertainty = [10]

x = [101., 100., 10]
# /home/buding/Heptools/checkmate2/tools/python/analysismanager_core.py
model = pyhf.simplemodels.uncorrelated_background(
    signal=[10], 
    bkg=bkg, 
    bkg_uncertainty=bkg_uncertainty
)
from analysismanager_core import s95_exp

res = s95_exp(x)
