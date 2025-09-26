# %% [markdown]
#
# Generate MNIST 12x12 CSV
# 
# # TMP File
# A playground.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@yahoo.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 0.1.000 | 23/08/2025 | Royi Avital | First version                                                                            |
# |         |            |             |                                                                                          |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Image Processing & Computer Vision
import skimage as ski

# Machine Learning

# Deep Learning

# Miscellaneous
import random

# Visualization
import matplotlib.pyplot as plt


# %% Configuration

# %matplotlib inline

# warnings.filterwarnings("ignore")

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

# sns.set_theme() #>! Apply SeaBorn theme

# %% Constants


# %% Local Packages


# %% Auxiliary Functions


# %% Parameters

# Data Source: [Downscaled 12x12 MNIST Handwritten Digits](https://www.kaggle.com/datasets/phillipkerger/downscaled-12x12-mnist-handwritten-digits)
testFeaturesFileName = r'mnist12x12_testfeats.npy'
testLabelsFileName = r'mnist12x12_testlabels.npy'
trainFeaturesFileName = r'mnist12x12_trainfeats.npy'
trainLabelsFileName = r'mnist12x12_trainlabels.npy'

csvFileName = r'MNIST12x12.csv'


# %% Load / Generate Data
# Data Source: [Downscaled 12x12 MNIST Handwritten Digits](https://www.kaggle.com/datasets/phillipkerger/downscaled-12x12-mnist-handwritten-digits)

mD = np.concatenate((
    np.load(trainFeaturesFileName),
    np.load(testFeaturesFileName)
), axis = 0, dtype = np.uint8, casting = 'unsafe')

vY = np.concatenate((
    np.load(trainLabelsFileName),
    np.load(testLabelsFileName)
), axis = 0, dtype = np.uint8, casting = 'unsafe')

lCol = [f'Pixel{(ii + 1):03d}' for ii in range(mD.shape[1])] + ['Label']
mX = np.concatenate((mD, vY.reshape((-1, 1))), axis = 1)

dfData = pd.DataFrame(data = mX, columns = lCol, dtype = np.uint8)
dfData.to_csv(csvFileName, index = False)


# %% Validate Data

dfData = pd.read_csv(csvFileName, dtype = np.uint8)
sampleIdx = random.randrange(dfData.shape[0])

mI = np.reshape(dfData.iloc[sampleIdx, :-1].to_numpy(), (12, 12))
valY = dfData.iloc[sampleIdx, -1]

plt.imshow(mI, cmap = 'gray')
plt.title(f'Sample Index: {sampleIdx}, Label: {valY}')



# %%
