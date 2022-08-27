# %% [markdown]
#
# # StackExchnage Signal Processing Q84301
# Estimate the Blur Kernel of a Linear 2D Operator
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

# OpenCV
import cv2 as cv
import PIL 

# Misc
import datetime
import os
from platform import python_version
import random


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

DATA_FOLDER_NAME = 'Data'

# %% Self Modules / Packages

from StackExchnageAuxFun import *

# %% Parameters

numRows = 5

# %% Loading / Generating Data


# %%
