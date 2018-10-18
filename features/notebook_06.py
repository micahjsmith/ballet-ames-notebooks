# taken from https://www.kaggle.com/tannercarbonati/detailed-data-analysis-ensemble-modeling
# Adapted into the ballet framework
import ballet
import ballet.eng
from ballet import Feature
import numpy as np
import pandas as pd
import sklearn

features = []

input = ['Pool QC']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer, name='PoolQC Misc Fill')
features.append(misc_fill)

input = ['Year Built', 'Garage Yr Blt']
def calc_age(df):
    mask = pd.isnull(df['Year Built'])
    df['Year Built'][mask] = df['Garage Yr Blt'][mask]
    return df['Year Built']
transformer = ballet.eng.SimpleFunctionTransformer(func=calc_age)
age = Feature(input=input, transformer=transformer)
features.append(age)

input = ['Garage Type']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Finish']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Qual']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Cond']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Cars']
transformer = [ballet.eng.missing.NullFiller(), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Area']
transformer = [ballet.eng.missing.NullFiller(), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Type']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Finish']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Qual']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Exterior 1st']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
factor = Feature(input=input, transformer=transformer)
features.append(factor)

input = ['Exterior 2nd']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
factor = Feature(input=input, transformer=transformer)
features.append(factor)

input = ['Electrical']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Electrical'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Utilities']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Utilities'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Sale Type']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Sale Type'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Kitchen Qual']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Kitchen Qual'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Mas Vnr Type']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Mas Vnr Area']
transformer = [ballet.eng.missing.NullFiller(), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Lot Frontage', 'Neighborhood']
def impute_lot_frontage(df):
    frontage = df['Lot Frontage']
    return frontage.fillna(frontage.median())
transformer = [
    ballet.eng.GroupedFunctionTransformer(func=impute_lot_frontage, groupby_kwargs={'by': 'Neighborhood'}),
    ballet.eng.SimpleFunctionTransformer(lambda s: s.fillna(s.median()))
]
frontage_feature = Feature(input=input, transformer=transformer,  name='Lot Frontage Fill')
features.append(frontage_feature)

input = ['Fence']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Misc Feature']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Fireplace Qu']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Alley']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer, name='Alley Misc Fill')
features.append(misc_fill)

def has_reg(df):
    return (df['Lot Shape'] == 'Reg') * 1
input = ['Lot Shape']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_reg)
reg = Feature(input=input, transformer=transformer)
features.append(reg)

def has_contour(df):
    return (df['Land Contour'] == 'Lvl') * 1
input = ['Land Contour']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_contour)
contour = Feature(input=input, transformer=transformer)
features.append(contour)

def has_qual(df):
    return (df['Paved Drive'] == 'Y') * 1
input = ['Paved Drive']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['Land Slope'] == 'Gtl') * 1
input = ['Land Slope']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['Electrical'] == 'SBrkr') * 1
input = ['Electrical']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['Sale Condition'] == 'Partial') * 1
input = ['Sale Condition']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['Wood Deck SF'] > 0) * 1
input = ['Wood Deck SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['Mas Vnr Area'] > 0) * 1
input = ['Mas Vnr Area']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['Neighborhood'].isin(['Crawfor', 'Somerst, Timber', 'StoneBr', 'NoRidge', 'NridgeHt'])) * 1
input = ['Neighborhood']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

input = ['Year Built', 'Year Remod/Add']
def is_remod(df):
    return df['Year Built'] != df['Year Remod/Add']
transformer = ballet.eng.SimpleFunctionTransformer(func=is_remod)
remod = Feature(input=input, transformer=transformer, name='Remodeled')
features.append(remod)

input = ['Yr Sold', 'Year Remod/Add']
def calc_age(df):
    return df['Yr Sold'] - df['Year Remod/Add']
transformer = ballet.eng.SimpleFunctionTransformer(func=calc_age)
age = Feature(input=input, transformer=transformer, name='Age')
features.append(age)

input = ['Year Built']
def calc_age(df):
    return 2018 - df['Year Built']
transformer = ballet.eng.SimpleFunctionTransformer(func=calc_age)
age = Feature(input=input, transformer=transformer, name='Age')
features.append(age)

input = ['Yr Sold']
def calc_age(df):
    return 2018 - df['Yr Sold']
transformer = ballet.eng.SimpleFunctionTransformer(func=calc_age)
age = Feature(input=input, transformer=transformer, name='Age')
features.append(age)
