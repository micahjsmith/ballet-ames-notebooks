# taken from https://www.kaggle.com/agehsbarg/top-10-0-10943-stacking-mice-and-brutal-force
# Adapted into the ballet framework
import ballet
import ballet.eng
from ballet import Feature
import numpy as np
import pandas as pd
import sklearn

features = []

input = ['Total Bsmt SF', '1st Flr SF', '2nd Flr SF']
def add_areas(df):
    total_area = df['Total Bsmt SF'] + df['1st Flr SF'] + df['2nd Flr SF']
    return total_area.fillna(0)
transformer = ballet.eng.SimpleFunctionTransformer(func=add_areas)
total_area = Feature(input=input, transformer=transformer, name='Total Area Calculation')
features.append(total_area)

input = ['Yr Sold', 'Year Remod/Add']
def calc_age(df):
    year = df['Yr Sold'] - df['Year Remod/Add']
    return year.fillna(0)
transformer = ballet.eng.SimpleFunctionTransformer(func=calc_age)
age = Feature(input=input, transformer=transformer, name='Age')
features.append(age)

input = ['Overall Qual', 'Overall Cond']
def calc_qual(df):
    return df['Overall Qual'] - df['Overall Cond']
transformer = ballet.eng.SimpleFunctionTransformer(func=calc_qual)
qual = Feature(input=input, transformer=transformer, name='Qual')
features.append(qual)

def has_qual(df):
    return (df['Open Porch SF'] != 0) * 1
input = ['Open Porch SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['Enclosed Porch'] != 0) * 1
input = ['Enclosed Porch']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['3Ssn Porch'] != 0) * 1
input = ['3Ssn Porch']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['Screen Porch'] != 0) * 1
input = ['Screen Porch']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

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
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Cond']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Fence']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Street']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Lot Shape']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Land Contour']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['BsmtFin Type 1']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['BsmtFin Type 2']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Central Air']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Misc Feature']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Utilities']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Functional']
transformer = [ballet.eng.missing.NullFiller(replacement="Typ", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Sale Condition']
transformer = [ballet.eng.missing.NullFiller(replacement="Typ", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Pool QC']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Pool QC'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
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
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Fireplace Qu'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Finish']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Garage Finish'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Qual']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Garage Qual'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Bsmt Exposure']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Bsmt Exposure'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Electrical']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Electrical'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['MS Zoning']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['MS Zoning'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Exterior 1st']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Exterior 1st'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Exterior 2nd']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Exterior 2nd'][0])
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

input = ['Sale Type']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Sale Type'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Yr Sold']
year = Feature(input=input, transformer=sklearn.preprocessing.OneHotEncoder(), name='Year Categorical')
features.append(year)

input = ['Mo Sold']
month = Feature(input=input, transformer=sklearn.preprocessing.OneHotEncoder(), name='Month Categorical')
features.append(month)

input = ['MS SubClass']
transformer = [ballet.eng.missing.NullFiller(replacement="None"), sklearn.preprocessing.OneHotEncoder()]
ms_fill = Feature(input=input, transformer=transformer, name='MS Fill None')
features.append(ms_fill)