# %% [markdown]
#
# [![Royi Avital](https://i.imgur.com/ghq7NUE.png)](https://github.com/RoyiAvital/StackExchangeCodes)
# 
# # StackExchange Signal Processing Q29744  
# https://dsp.stackexchange.com/questions/29744. </br>   
# Amplitude and Phase Recovery of a Signal Embedded in Linear Signal with Noise
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@yahoo.com
#
# References:
#  1.   A
#
# Remarks:
#  1.   B
#  2.   C
#
# To Do:
#  1.   D
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 0.1.000 | 28/06/2023 | Royi Avital | First version                                                                            |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, MeanShift, OPTICS, SpectralClustering
from sklearn.metrics import r2_score

# Image Processing & Computer Vision

# Miscellaneous
import datetime
import itertools
import os
from platform import python_version
import random
import warnings
import yaml


# Visualization
from bokeh.plotting import figure, show
from bokeh.palettes import Bokeh8 as paletteBokeh
import matplotlib.pyplot as plt
import seaborn as sns

# Jupyter
from ipywidgets import Dropdown, Layout
from ipywidgets import interact


# %% Configuration

paletteBokehCycler = itertools.cycle(paletteBokeh)
bokeTools ="hover, pan, wheel_zoom, zoom_in, zoom_out, box_zoom, reset, tap, save, box_select, poly_select, lasso_select, help"

# %matplotlib inline

# warnings.filterwarnings("ignore")

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

sns.set_theme() #>! Apply SeaBorn theme

# %% Constants



# %% Local Packages


# %% Auxiliary Functions

def CalcDistMat( mX: np.ndarray, maxLen: float = 1.0, maxMSE: float = 0.9, maxDist: float = np.inf ) -> np.ndarray:

    # mX[:, 0] - Ordered x
    # mX[:, 1] - Matching y
    numRows, numCols = mX.shape

    mD = np.zeros(shape = (numRows, numRows))

    for ii in range(numRows):
        for jj in range(ii + 2, numRows): #<! Between 2 points linear is perfect
            if (np.abs(mX[ii, 0] - mX[jj, 0]) > maxLen):
                mD[ii, jj] = maxDist
                continue

            vP = np.polyfit(mX[ii:(jj + 1), 0], mX[ii:(jj + 1), 1], deg = 1)
            vY = np.polyval(vP, mX[ii:(jj + 1), 0])
            # See https://stackoverflow.com/questions/893657
            # See https://stats.stackexchange.com/questions/524799
            estMse = np.mean(np.square(vY - mX[ii:(jj + 1), 1]))
            if estMse > maxMSE:
                mD[ii, jj] = maxDist
            else:
                mD[ii, jj] = estMse
    
    mD = np.maximum(mD, mD.T) #<! Make is symmetric

    return mD

def CalcAffMat( mX: np.ndarray, maxLen: float = 1.0, minR2: float = 0.9 ) -> np.ndarray:

    # mX[:, 0] - Ordered x
    # mX[:, 1] - Matching y
    numRows, numCols = mX.shape

    mA = np.ones(shape = (numRows, numRows))

    for ii in range(numRows):
        for jj in range(ii + 2, numRows): #<! Between 2 points linear is perfect
            if (np.abs(mX[ii, 0] - mX[jj, 0]) > maxLen):
                mA[ii, jj] = 0
                continue

            vP = np.polyfit(mX[ii:(jj + 1), 0], mX[ii:(jj + 1), 1], deg = 1)
            vY = np.polyval(vP, mX[ii:(jj + 1), 0])
            # See https://stackoverflow.com/questions/893657
            # See https://stats.stackexchange.com/questions/524799
            scoreR2 = r2_score(mX[ii:(jj + 1), 1], vY)
            if scoreR2 > minR2:
                mA[ii, jj] = scoreR2
            else:
                mA[ii, jj] = 0
    
    mA = np.minimum(mA, mA.T) #<! Make is symmetric

    return mA

# See https://win-vector.com/2018/12/31/introducing-rcppdynprog/
# See https://github.com/WinVector/RcppDynProg/blob/master/extras/DynProg.py
# See https://www.geeksforgeeks.org/minimum-number-of-jumps-to-reach-end-of-a-given-array/
def solve_dynamic_program(x, kmax):
    """x n by n inclusive interval cost array, kmax maximum number of steps to take"""
    # for cleaner notation
    # solution and x will be indexed from 1 using
    # R_INDEX_DELTA
    # intermediate arrays will be padded so indexing
    # does not need to be shifted
    R_INDEX_DELTA = -1
    R_SIZE_PAD = 1
  
    # get shape of problem
    n = x.shape[0]
    if kmax>n:
        kmax = n
 
    # get some edge-cases
    if (kmax<=1) or (n<=1):
         return [1, n+1]

    # best path cost up to i (row) with exactly k-steps (column)
    path_costs = np.zeros((n + R_SIZE_PAD, kmax + R_SIZE_PAD))
    # how many steps we actually took
    k_actual = np.zeros((n + R_SIZE_PAD, kmax + R_SIZE_PAD))
    # how we realized each above cost
    prev_step = np.zeros((n + R_SIZE_PAD, kmax + R_SIZE_PAD))
  
    # fill in path and costs tables
    for i in range(1, n+1):
        prev_step[i, 1] = 1
        path_costs[i, 1] = x[1 + R_INDEX_DELTA, i + R_INDEX_DELTA]
        k_actual[i, 1] = 1

    # refine dynprog table
    for ksteps in range(2, kmax+1):
        # compute larger paths
        for i in range(1, n+1):
            # no split case
            pick = i
            k_seen = 1
            pick_cost = x[1 + R_INDEX_DELTA, i + R_INDEX_DELTA]
            # split cases
            for candidate in range(1, i):
                cost = path_costs[candidate, ksteps-1] + \
                    x[candidate + 1 + R_INDEX_DELTA, i + R_INDEX_DELTA]
                k_cost = k_actual[candidate, ksteps-1] + 1
                if (cost<=pick_cost) and \
                    ((cost<pick_cost) or (k_cost<k_seen)):
                    pick = candidate
                    pick_cost = cost
                    k_seen = k_cost
            path_costs[i, ksteps] = pick_cost
            prev_step[i, ksteps] = pick
            k_actual[i, ksteps] = k_seen
 
    # now back-chain for solution
    k_opt = int(k_actual[n, kmax])
    solution = [0]*(k_opt+1)
    solution[1 + R_INDEX_DELTA] = int(1)
    solution[k_opt + 1 + R_INDEX_DELTA] = int(n+1)
    i_at = n
    k_at = k_opt
    while k_at>1:
        prev_i = int(prev_step[i_at, k_at])
        solution[k_at + R_INDEX_DELTA] = int(prev_i + 1)
        i_at = prev_i
        k_at = k_at - 1
    return solution



# %% Parameters

vYFileName = 'vY.csv'

# Distance / Affinity
maxLen = 15
maxMSE = 0.2
minR2  = 0.35
σ = 0.1

# Model
numClusters = 4


# %% Load / Generate Data

vY = np.loadtxt(vYFileName)
vX = np.linspace(0, len(vY) - 1, len(vY))


# %% Analysis

mX = np.column_stack((vX, vY))

mD = CalcDistMat(mX, maxLen = 50, maxMSE = 0.2, maxDist = 1e6)
# mA = np.exp(-0.5 * np.square(mD / σ))

# mA = CalcAffMat(mX, maxLen = maxLen, minR2 = minR2)
# mA = np.square(mA)


# mX = np.column_stack((vX / (len(vY) - 1), vY))

# oMeanShift = MeanShift()
# vL = oMeanShift.fit_predict(mX)

# oOptics = OPTICS(metric = 'precomputed')
# vL = oOptics.fit_predict(mD)

oHdbscan = HDBSCAN(metric = 'precomputed')
vL = oHdbscan.fit_predict(mD)

# oAggClust = AgglomerativeClustering(n_clusters = 4, linkage = 'complete')
# vL = oAggClust.fit_predict(mX)

# oSpectralClustering = SpectralClustering(n_clusters = numClusters, affinity = 'precomputed', assign_labels = 'cluster_qr')
# vL = oSpectralClustering.fit_predict(mA)


# %% Display Results

hP = figure(tools = bokeTools, x_range = (0, 500), y_range = (-0.5, 1.5))

for clsLabel, clsColor in zip(np.unique(vL), paletteBokehCycler):
    vIdx = np.flatnonzero(vL == clsLabel)
    vXX = vX[vIdx]
    vYY = vY[vIdx]
    hP.scatter(vXX, vYY, radius = 5, fill_color = clsColor, line_color = None)

show(hP)

# %%

# Bokeh Image 1:1
x = np.linspace(0, 10, 300)
y = np.linspace(0, 10, 300)
xx, yy = np.meshgrid(x, y)
d = np.sin(xx) * np.cos(yy)

# p = figure(tools = "hover", width=600, height=600, x_range = (0, 300), y_range = (0, 300))
p = figure(tools = "hover", width=600, height=600)
# p.x_range.range_padding = 0
# p.y_range.range_padding = 0

# must give a vector of image data for image parameter
p.image(image=[d], x = 0, y=0, dw = 300, dh = 300, dw_units = 'screen', dh_units = 'screen', palette = "Sunset11", level = "image")
p.grid.grid_line_width = 0.5

show(p)
