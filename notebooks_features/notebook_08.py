# taken from https://www.kaggle.com/poonaml/house-prices-data-exploration-and-visualisation
# Adapted into the ballet framework
# some features are not in her kernel but implied from her analyses
import ballet
import ballet.eng
from ballet import Feature
import numpy as np
import pandas as pd
import scipy
import sklearn
import sklearn_pandas
from sklearn.model_selection import train_test_split
ballet.__version__

features = []

input = ['Lot Area']
def calc_sqrt(df):
    return np.sqrt(df['Lot Area'])
transformer = ballet.eng.SimpleFunctionTransformer(func=calc_sqrt)
sqrt = Feature(input=input, transformer=transformer)
features.append(sqrt)

input = ['Lot Area', 'Lot Frontage']
def fill_area(df):
    mask = pd.isnull(df['Lot Frontage'])
    df['Lot Frontage'][mask] = np.sqrt(df['Lot Area'])[mask]
    return df['Lot Frontage']
transformer = ballet.eng.SimpleFunctionTransformer(func=fill_area)
area = Feature(input=input, transformer=transformer)
features.append(area)

input = ['Mas Vnr Type']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Mas Vnr Area']
transformer = ballet.eng.missing.NullFiller(isnull=pd.isnull)
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Electrical']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Electrical'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Alley']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer, name='Alley Misc Fill')
features.append(misc_fill)

input = ['BsmtFin Type 1']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['BsmtFin Type 2']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Bsmt Qual']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Bsmt Qual'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Bsmt Cond']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Bsmt Cond'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Fireplace Qu']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Cond']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Qual']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Type']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Finish']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Cars']
transformer = ballet.eng.missing.NullFiller(isnull=pd.isnull)
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Area']
transformer = ballet.eng.missing.NullFiller(isnull=pd.isnull)
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Pool QC']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Fence']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Misc Feature']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

def has_qual(df):
    return (df['Fireplaces'] > 1) * 1
input = ['Fireplaces']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['Fireplace Qu'] == "Ex") * 1
input = ['Fireplace Qu']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['Misc Feature'] == "TenC") * 1
input = ['Misc Feature']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (~pd.isnull(df['Pool QC'])) * 1
input = ['Pool QC']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)
