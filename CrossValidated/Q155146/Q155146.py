# %% [markdown]
#
# # StackExchange Cross Validated Q155146
# https://stats.stackexchange.com/questions/155146
# Visualizing Classifier Decision Boundary of High Dimensional Data.
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

# Machine Learning
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Miscellaneous
import os
from platform import python_version
import random
import sys

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

# %% Parameters

# Model
parameterC      = 1.0
kernelType      = 'rbf' #<! 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
decisionFunType = 'ovr' #<! 'ovo', 'ovr'

# Visualization
numGridPts = 1_000

# %% Loading / Generating Data

mX, vY = load_iris(return_X_y = True)
numSamples, numFeatures = mX.shape


# %% Analysis

oCls = SVC(C = parameterC, kernel = kernelType, decision_function_shape = decisionFunType, random_state = seedNum)
oCls = oCls.fit(mX, vY)

oDr = PCA(n_components = 2, random_state = seedNum)
oDr = oDr.fit(mX)

mX2D = oDr.transform(mX)
minValX, minValY = np.min(mX2D, axis = 0)
maxValX, maxValY = np.max(mX2D, axis = 0)

minValX = FloorFloat(minValX, numDigits = 1)
minValY = FloorFloat(minValY, numDigits = 1)
maxValX = CeilFloat(maxValX, numDigits = 1)
maxValY = CeilFloat(maxValY, numDigits = 1)

# Grid
vGx = np.linspace(minValX, maxValX, numGridPts)
vGy = np.linspace(minValX, maxValY, numGridPts)

# The 2D Grid
mG = np.column_stack((np.tile(vGx, numGridPts), np.repeat(vGy, numGridPts)))

# Inverse Transform and Prediction
mXX = oDr.inverse_transform(mG)
vYY = oCls.predict(mXX)
mYY = np.reshape(vYY, (numGridPts, numGridPts)) #<! 2D Grid for Contour Plot

# %% Visualization

hF, hA = plt.subplots(figsize = (10, 10))
hA.scatter(mX2D[:, 0], mX2D[:, 1], c = vY, s = 100, edgecolors = 'k', cmap = 'Set1', label = 'Ground Truth')
hA.contourf(vGx, vGy, mYY, levels = [-0.5, 0.5, 1.5], vmin = 0, vmax = 2, alpha = 0.3, cmap = 'Set1')
hA.set_xlabel('Principal Component 1')
hA.set_ylabel('Principal Component 2')
hA.set_title('SVC Decision Boundary with PCA Dimensionality Reduction');

# %%
