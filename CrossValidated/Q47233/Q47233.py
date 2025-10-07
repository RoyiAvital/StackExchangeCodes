# %% [markdown]
#
# # StackExchange Cross Validated Q47233
# https://stats.stackexchange.com/questions/47233
# Cost Sensitive Classifier to Minimize Bayesian Risk.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@yahoo.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                   |
# |---------|------------|-------------|--------------------------------------------------------------------|
# | 1.0.000 | 07/10/2025 | Royi Avital | First version                                                      |
# |         |            |             |                                                                    |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

from numba import njit

# Machine Learning

# Miscellaneous
import math
import os
from platform import python_version
import random
import sys

# Typing
from typing import List, Tuple, Union
from numpy.typing import ArrayLike, NDArray

# Visualization
import matplotlib.pyplot as plt

# Jupyter
from IPython import get_ipython

# %% Configuration

# %matplotlib inline

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

# sns.set_theme() #>! Apply SeaBorn theme

# %% Constants

# PROJECT_FOLDER      = os.path.abspath(os.path.join(__file__, '../../..'))
# DATA_FOLDER_NAME    = 'Data'

PROJECT_NAME        = 'StackExchangeCodes'
DATA_FOLDER_NAME    = 'Data'

PROJECT_BASE_FOLDER_PATH = os.getcwd()[:(len(os.getcwd()) - (os.getcwd()[::-1].lower().find(PROJECT_NAME.lower()[::-1])))]
DATA_FOLDER_PATH         = os.path.join(PROJECT_BASE_FOLDER_PATH, DATA_FOLDER_NAME)

# %% Self Modules / Packages

sys.path.append(PROJECT_BASE_FOLDER_PATH)

from SEPythonModule import *

# %% Auxiliary Functions

@njit(cache = True, fastmath = True)
def ComputerCost( mP: NDArray, mCost: NDArray ) -> NDArray:
    # `mP` - Probability per point per class. Each row is a point, each column is a class.
    # `mCost` - Cost per class. Each row is the true class, each column is the predicted class.
    numPts = mP.shape[0]
    numCls = mP.shape[1]
    
    mC = np.zeros((numPts, numCls)) #<! Per point cost per class

    for ii in range(numPts):
        for jj in range(numCls):
            # jj - True class
            for kk in range(numCls):
                # kk - Predicted class
                mC[ii, kk] += mP[ii, jj] * mCost[jj, kk]

    return mC



# %% Parameters

# Model
# Mean per Class
lModelMean = [
    np.array([1.0,  2.0]),
    np.array([2.0, -2.0]),
    np.array([1.0, -1.0]),
]

# Covariance per Class
lModelCov = [
    np.array([[2.0, 1.0], [1.0, 2.0]]),
    np.array([[1.0, 0.0], [0.0, 2.0]]),
    np.array([[7.0, 5.0], [5.0, 6.0]]),
]

# Prior probabilities per class
vModelProb  = np.array([1.0, 1.0, 1.0])
vModelProb /= np.sum(vModelProb)

# Cost Sensitive Model
mCostModel = np.array([
    [0.0, 1.0, 3.0],
    [2.0, 0.0, 2.0],
    [4.0, 3.0, 0.0],
])

# Visualization
tuGridLim  = (-8.0, 8.0)
numGridPts = 1_000

# %% Loading / Generating Data

# Model
numModels = len(lModelMean)
numCls    = numModels
lM = [sp.stats.multivariate_normal(mean = lModelMean[ii], cov = lModelCov[ii]) for ii in range(numModels)]


# %% Analysis

# Grid
vG = np.linspace(tuGridLim[0], tuGridLim[1], numGridPts)
mG = np.column_stack((np.tile(vG, numGridPts), np.repeat(vG, numGridPts)))

# Probability per Model
mP = np.column_stack([lM[ii].pdf(mG) for ii in range(numModels)]) #<! Each model is a column
mPx = np.reshape(mP @ vModelProb, (numGridPts, numGridPts)) #<! GMM Probability

# Cost per class
mC = ComputerCost(mP, mCostModel)
vY = np.argmin(mC, axis = 1) #<! Minimum Cost Class
mYY = np.reshape(vY, (numGridPts, numGridPts)) #<! 2D Grid for Contour Plot

# Maximum Probability Class
vPP = np.argmax(mP, axis = 1)


# %% Visualization

tuExtent = (tuGridLim[0], tuGridLim[1], tuGridLim[0], tuGridLim[1])
oCmap = plt.get_cmap('Set1', numModels)

hF, hA = plt.subplots(figsize = (6, 6))
hA.imshow(mPx, cmap = 'jet', origin = 'lower', extent = tuExtent)
# Contour per Model (Each with a single color)
for ii in range(numModels):
    mPi = np.reshape(mP[:, ii], (numGridPts, numGridPts))
    # hA.contour(vG, vG, mPi, levels = 7, colors = [f'C{ii + 2}'], alpha = 0.95) #<! See https://matplotlib.org/stable/users/explain/colors/colors.html
    hA.contour(vG, vG, mPi, levels = 7, colors = oCmap(ii), alpha = 0.95) #<! See https://matplotlib.org/stable/users/explain/colors/colors.html
    # Virtual Lines for Legend entry
    # hA.plot([], [], color = f'C{ii + 2}', label = f'Model {ii + 1}')
    hA.plot([], [], color = oCmap(ii), label = f'Model {ii + 1}')
hA.set_xlabel('$x_1$')
hA.set_ylabel('$x_2$')
hA.set_title('Combined Probability and per Model PDF Contours')
hA.legend()


hF, vHa = plt.subplots(nrows = 1, ncols = 2, figsize = (9, 5))
vHa = vHa.flatten()

hA = vHa[0]
hA.imshow(mYY, cmap = 'Set1', interpolation = 'nearest', origin = 'lower', extent = tuExtent, resample = False)
hA.set_xlabel('$x_1$')
hA.set_ylabel('$x_2$')
hA.set_title('Minimum Cost Decision Boundary')
hA.set_xticks([])
hA.set_yticks([])
# hA.legend()
# hA.scatter(mG[::20, 0], mG[::20, 1], s = 3, c = vPP[::20], marker = '+', cmap = 'Set1', edgecolors = 'none', alpha = 0.95)


hA = vHa[1]
hA.imshow(np.reshape(vPP, (numGridPts, numGridPts)), cmap = 'Set1', interpolation = 'nearest', origin = 'lower', extent = tuExtent, resample = False)
hA.set_xlabel('$x_1$')
hA.set_ylabel('$x_2$')
hA.set_title('Maximum Probability Decision Boundary')
hA.set_xticks([])
hA.set_yticks([])

# Create Manual Legend
lHandles = [plt.Line2D([0], [0], marker = 's', color = 'w', markerfacecolor = oCmap(ii), markersize = 10) for ii in range(numModels)]
lLabels  = [f'Class {i + 1}' for i in range(numModels)]
hA.legend(lHandles, lLabels, loc = 'upper right')




# hA.contourf(vGx, vGy, mYY, levels = [-0.5, 0.5, 1.5], vmin = 0, vmax = 2, alpha = 0.3, cmap = 'Set1')
# hA.set_xlabel('Principal Component 1')
# hA.set_ylabel('Principal Component 2')
# hA.set_title('SVC Decision Boundary with PCA Dimensionality Reduction');

# %%
