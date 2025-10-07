# %% [markdown]
#
# # StackExchange Signal Processing Q84301
# https://dsp.stackexchange.com/questions/84301
# Estimate the Blur Kernel of a Linear 2D Operator.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@yahoo.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                   |
# |---------|------------|-------------|--------------------------------------------------------------------|
# | 0.1.000 | 27/08/2022 | Royi Avital | First version                                                      |
# |         |            |             |                                                                    |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Misc
import datetime
import os
from platform import python_version
import random
import sys


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, show

# Jupyter
from ipywidgets import interact, Dropdown, Layout

# %% Configuration

# %matplotlib inline

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

sns.set_theme() #>! Apply SeaBorn theme

# %% Constants

PROJECT_FOLDER      = os.path.abspath(os.path.join(__file__, '../../..'))
DATA_FOLDER_NAME    = 'Data'


# %% Self Modules / Packages

sys.path.append(PROJECT_FOLDER)

import SEPythonModule

# %% Parameters

numRows         = 5 #<! Input matrix
noiseStd        = 0.01 #<! AWGN noise
kernelRadius    = 5

# %% Loading / Generating Data

mA = np.random.binomial(n = 1, p = 0.2, size = (numRows, numRows))
kernelLength = 2 * kernelRadius + 1
vX = np.linspace(-kernelRadius, kernelRadius, kernelLength)
mK = np.exp(-np.outer(vX, vX))
mK /= np.sum(mK)

# %%
