# %% [markdown]
#
# # Generate Binary Classification Data
# Generates samples for binary classification which requires a non linear classifier.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@yahoo.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                   |
# |---------|------------|-------------|--------------------------------------------------------------------|
# | 1.0.000 | 14/10/2025 | Royi Avital | First version                                                      |
# |         |            |             |                                                                    |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

from numba import njit

# Machine Learning
from sklearn.datasets import make_moons, make_blobs

# Miscellaneous
import math
from platform import python_version
import random

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


# %% Self Modules / Packages


# %% Auxiliary Functions


# %% Parameters



# %% Loading / Generating Data

mX1, vY1 = make_moons(n_samples = 500, noise = 0.075, random_state = seedNum)
mX2, vY2 = make_blobs(n_samples = 500, n_features = 2, centers = [[1.5, -1.0], [-0.5, 1.5]], cluster_std = [0.125, 0.195], random_state = seedNum)

mX = np.vstack((mX1, mX2))
vY = np.hstack((vY1, vY2))


# %% Analysis


# %% Visualization

hF, hA = plt.subplots(figsize = (6, 6))

hA.scatter(mX[:, 0], mX[:, 1], c = vY, s = 40, cmap = plt.cm.Spectral)
hA.set_title('Binary Classification Data - Non Linear')
hA.set_xlabel('$x_1$')
hA.set_ylabel('$x_2$')
hA.axis('equal')


# %% Export Data

# Export Data to CSV
dfData = pd.DataFrame(data = mX, columns = ['x1', 'x2'])
dfData['y'] = vY
dfData.to_csv('BinaryClassificationData.csv', index = False)


# %%
