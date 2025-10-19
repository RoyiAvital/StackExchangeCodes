# %% [markdown]
#
# # Generate Binary Classification Data
# Transforms LibSVM data into CSV.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@yahoo.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                   |
# |---------|------------|-------------|--------------------------------------------------------------------|
# | 1.0.000 | 18/10/2025 | Royi Avital | First version                                                      |
# |         |            |             |                                                                    |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

from numba import njit

# Machine Learning
from sklearn.datasets import load_svmlight_file

# Miscellaneous
import bz2
import math
import os
from platform import python_version
import random
import shutil
import urllib.request

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

# Choose from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
dataUrl         = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2'
baseFileName    = 'DataSet'
archiveFileName = baseFileName + '.bz2'
libSvmFileName  = baseFileName + '.t'
csvFileName     = baseFileName + '.csv'


# %% Loading / Generating Data

# Failing SSL Certificate
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# In Interactive Window downloads to the file folder
urllib.request.urlretrieve(dataUrl, archiveFileName)

# Assuming `bz2` file contains a single file
with bz2.BZ2File(archiveFileName) as fr, open(libSvmFileName, 'wb') as fw:
    shutil.copyfileobj(fr, fw)


# %% Loading into Array

mX, vY = load_svmlight_file(libSvmFileName)
mX = mX.toarray()
vY = vY.astype(int)

numFeatures = np.size(mX, 1)


# %% Export Data

dfData = pd.DataFrame(np.column_stack((mX, vY)))
lCol = [f'Feature_{(ii + 1):02d}' for ii in range(numFeatures)] + ['Label']
dfData.columns = lCol

# Write data to CSV
dfData.to_csv(csvFileName, index = False)


# %% Cleanup

os.remove(archiveFileName)
os.remove(libSvmFileName)



# %%
