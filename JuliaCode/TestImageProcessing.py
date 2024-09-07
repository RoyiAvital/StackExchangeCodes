# %% [markdown]
# 
# # Test Image Processing
# Generates reference ot test Julia code.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 1.0.000 | 07/09/2024 | Royi Avital | First version                                                                            |
# |         |            |             |                                                                                          |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Typing
from typing import Any, Callable, Dict, Generator, List, Optional, Self, Set, Tuple, Union

# Image Processing & Computer Vision

# Machine Learning

# Deep Learning

# Miscellaneous
import datetime
import gdown
import json
import os
from platform import python_version
import random
import warnings
import shutil
import urllib.request


# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# Jupyter
from IPython import get_ipython


# %% Configuration

# %matplotlib inline

# warnings.filterwarnings("ignore")

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

# Matplotlib default color palette
lMatPltLibclr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# sns.set_theme() #>! Apply SeaBorn theme


# %% Constants

# Reference for `scipy.signal.convolve2d()`
L_BND_MODE      = ['fill', 'symm', 'wrap']
L_CONV_MODE     = ['full', 'same', 'valid']
L_GAUSS_MODE    = ['constant', 'mirror', 'nearest', 'reflect', 'wrap']


# %% Local Packages


# %% Auxiliary Functions


# %% Parameters

numRows = 123
numCols = 122

repFactor = 20


# %% Load / Generate Data

mI = np.random.rand(numRows, numCols)

vBndMode    = np.tile(L_BND_MODE, repFactor)
vCnvMode    = np.random.permutation(np.tile(L_CONV_MODE, repFactor))
vGaussMode  = np.tile(L_GAUSS_MODE, repFactor)


# %% Convolution

numTests = min(len(vBndMode), len(vCnvMode))
cConv    = np.empty((numTests, 5), dtype = np.ndarray)

for ii in range(numTests):
    vKernelSize = np.random.randint(1, 10, size = (2))
    vKernelSize[0] += (1 - np.mod(vKernelSize[0], 2))
    vKernelSize[1] += (1 - np.mod(vKernelSize[1], 2))
    mK = np.random.rand(*vKernelSize)
    mO = sp.signal.convolve2d(mI, mK, mode = vCnvMode[ii], boundary = vBndMode[ii])

    cConv[ii, 0] = mO
    cConv[ii, 1] = vKernelSize
    cConv[ii, 2] = mK
    cConv[ii, 3] = vCnvMode[ii]
    cConv[ii, 4] = vBndMode[ii]


# %% Gaussian Filter

numTests = len(vGaussMode)
cGauss   = np.empty((numTests, 4), dtype = np.ndarray)

for ii in range(numTests):
    kernelStd    = 5 * random.random()
    kernelRadius = random.randint(1, 11)
    
    mO = sp.ndimage.gaussian_filter(mI, kernelStd, mode = vGaussMode[ii], radius = kernelRadius)

    cGauss[ii, 0] = mO
    cGauss[ii, 1] = kernelStd
    cGauss[ii, 2] = kernelRadius
    cGauss[ii, 3] = vGaussMode[ii]


# %% Save Data

dData = {'mI': mI, 'cConv': cConv, 'cGauss': cGauss}
np.savez_compressed('TestImageProcessing', **dData)
sp.io.savemat('TestImageProcessingPython.mat', dData, appendmat = True)


# %%
