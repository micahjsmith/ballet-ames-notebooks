# taken from https://www.kaggle.com/neviadomski/how-to-get-to-top-25-with-simple-model-sklearn
# Adapted into the ballet framework
import ballet
import ballet.eng
from ballet import Feature
import numpy as np
import pandas as pd
import sklearn

ballet.__version__

features = []

input = ['Alley']
transformer = [ballet.eng.missing.NullFiller(replacement="NOACCESS", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer, name='Alley Misc Fill')
features.append(misc_fill)

input = ['MS Zoning']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['MS Zoning'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['MS SubClass']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
ms_fill = Feature(input=input, transformer=transformer, name='MS Fill None')
features.append(ms_fill)

input = ['Lot Frontage']
def mean_filler(df):
    df = df.copy()
    return df.fillna(df.mean()['Lot Frontage'])
transformer = ballet.eng.SimpleFunctionTransformer(func=mean_filler)
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Overall Cond']
transformer =  sklearn.preprocessing.OneHotEncoder()
feature = Feature(input=input, transformer=transformer)
features.append(feature)

input = ['Mas Vnr Type']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Mas Vnr Type'][0])
transformer = [ballet.eng.SimpleFunctionTransformer(func=mode_filler), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['BsmtFin Type 1']
transformer = [ballet.eng.missing.NullFiller(replacement="NoBsmt", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['BsmtFin Type 2']
transformer = [ballet.eng.missing.NullFiller(replacement="NoBsmt", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Bsmt Qual']
transformer = [ballet.eng.missing.NullFiller(replacement="NoBsmt", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Bsmt Exposure']
transformer = [ballet.eng.missing.NullFiller(replacement="NoBsmt", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Bsmt Cond']
transformer = [ballet.eng.missing.NullFiller(replacement="NoBsmt", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Total Bsmt SF']
transformer = ballet.eng.missing.NullFiller()
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Electrical']
def mode_filler(df):
    df = df.copy()
    return df.fillna(df.mode()['Electrical'][0])
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

input = ['Kitchen AbvGr']
transformer = [ballet.eng.missing.NullFiller(replacement="missing", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Fireplace Qu']
transformer = [ballet.eng.missing.NullFiller(replacement="NoFP", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
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

input = ['Condition 1']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
factor = Feature(input=input, transformer=transformer)
features.append(factor)

input = ['Condition 2']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
factor = Feature(input=input, transformer=transformer)
features.append(factor)

input = ['Exterior 1st']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
factor = Feature(input=input, transformer=transformer)
features.append(factor)

input = ['Exterior 2nd']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
factor = Feature(input=input, transformer=transformer)
features.append(factor)

input = ['Total Bsmt SF', '1st Flr SF', '2nd Flr SF']
def add_areas(df):
    total_areas = df['Total Bsmt SF'] + df['1st Flr SF'] + df['2nd Flr SF']
    return total_areas.fillna(0)
transformer = ballet.eng.SimpleFunctionTransformer(func=add_areas)
total_area = Feature(input=input, transformer=transformer, name='Total Area Calculation')
features.append(total_area)