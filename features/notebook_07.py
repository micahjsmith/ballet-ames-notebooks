# taken from https://www.kaggle.com/apapiu/regularized-linear-models
# this one is interesting in that it uses mean and chooses to ignore nans
# Adapted into the ballet framework
import ballet
import ballet.eng
from ballet import Feature
import numpy as np
import pandas as pd
import sklearn
import sklearn_pandas
from sklearn.model_selection import train_test_split

ballet.__version__

features = []

# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath :
# missing values are likely zero for having no basement
input = ['BsmtFin SF 1']
def mean_filler(df):
    df = df.copy()
    return df.fillna(df['BsmtFin SF 1'].mean())
transformer = ballet.eng.SimpleFunctionTransformer(func=mean_filler)
mean_fill = Feature(input=input, transformer=transformer)
features.append(mean_fill)

# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath :
# missing values are likely zero for having no basement

input = ['BsmtFin SF 2']
def mean_filler(df):
    df = df.copy()
    return df.fillna(df.mean()['BsmtFin SF 2'])
transformer = ballet.eng.SimpleFunctionTransformer(func=mean_filler)
mean_fill = Feature(input=input, transformer=transformer)
features.append(mean_fill)

# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath :
# missing values are likely zero for having no basement

input = ['Bsmt Unf SF']
def mean_filler(df):
    df = df.copy()
    return df.fillna(df.mean()['Bsmt Unf SF'])
transformer = ballet.eng.SimpleFunctionTransformer(func=mean_filler)
mean_fill = Feature(input=input, transformer=transformer)
features.append(mean_fill)

# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath :
# missing values are likely zero for having no basement

input = ['Total Bsmt SF']
def mean_filler(df):
    df = df.copy()
    return df.fillna(df.mean()['Total Bsmt SF'])
transformer = ballet.eng.SimpleFunctionTransformer(func=mean_filler)
mean_fill = Feature(input=input, transformer=transformer)
features.append(mean_fill)

# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath :
# missing values are likely zero for having no basement

input = ['Bsmt Full Bath']
def mean_filler(df):
    df = df.copy()
    return df.fillna(df.mean()['Bsmt Full Bath'])
transformer = ballet.eng.SimpleFunctionTransformer(func=mean_filler)
mean_fill = Feature(input=input, transformer=transformer)
features.append(mean_fill)

# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath :
# missing values are likely zero for having no basement

input = ['Bsmt Half Bath']
def mean_filler(df):
    df = df.copy()
    return df.fillna(df.mean()['Bsmt Half Bath'])
transformer = ballet.eng.SimpleFunctionTransformer(func=mean_filler)
mean_fill = Feature(input=input, transformer=transformer)
features.append(mean_fill)

input = ['Mas Vnr Area']
def mean_filler(df):
    df = df.copy()
    return df.fillna(df.mean()['Mas Vnr Area'])
transformer = ballet.eng.SimpleFunctionTransformer(func=mean_filler)
mean_fill = Feature(input=input, transformer=transformer)
features.append(mean_fill)

input = ['Fireplace Qu']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
fill = Feature(input=input, transformer=transformer, name='FireplaceQu Fill Zero')
features.append(fill)

input = ['Alley']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer, name='Alley Misc Fill')
features.append(misc_fill)

input = ['Land Slope']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Mas Vnr Type']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Type']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Cond']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Fence']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Street']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Lot Shape']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Land Contour']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['BsmtFin Type 1']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['BsmtFin Type 2']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Central Air']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Misc Feature']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Utilities']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Functional']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Sale Condition']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)
