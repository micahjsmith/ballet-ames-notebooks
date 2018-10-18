# taken from https://www.kaggle.com/alvaromendoza/manual-fe-sklearn-pandas-pipelines-ensembling/output
# Adapted into the ballet framework
import ballet
import ballet.eng
from ballet import Feature
import numpy as np
import pandas as pd
import sklearn

all_features = []

def add_missing_0_to_mssubclass(df):
    """Zeros in 020-090 get cut off. This function prepends them back."""

    return df['MSSubClass'].apply(
        lambda x: '0' + str(x) if len(str(x)) == 2 else x)
transformer = ballet.eng.SimpleFunctionTransformer(func=add_missing_0_to_mssubclass)
feature = Feature(input=input, transformer=transformer)
all_features.append(feature)

input = ['Enclosed Porch', '3Ssn Porch', 'Open Porch SF']
def calc_porch_type(df):
    # Porch features
    porch_type = df['Total Porch Area'].apply(
        lambda x: 'Missing' if x == 0 else 'Multiple')
    porch_type[(df['Total Porch Area'] == df['Enclosed Porch'])
               & (df['Enclosed Porch'] > 0)] = 'Enclosed'
    porch_type[(df['Total Porch Area'] == df['3Ssn Porch'])
               & (df['3Ssn Porch'] > 0)] = '3Ssn'
    porch_type[(df['Total Porch Area'] == df['Open Porch SF'])
               & (df['Open Porch SF'] > 0)] = 'Open'
    return porch_type
transformer = [ballet.eng.SimpleFunctionTransformer(func=calc_porch_type), sklearn.preprocessing.OneHotEncoder()]
porch = Feature(input=input, transformer=transformer, name='Porch Type Calculation')
all_features.append(porch)

input = ['Enclosed Porch', '3Ssn Porch', 'Open Porch SF']
def calc_porch_area(df):
    return df['Enclosed Porch'] + df['3Ssn Porch'] + df['Open Porch SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=calc_porch_area)
total_area = Feature(input=input, transformer=transformer, name='Porch Area Calculation')
all_features.append(total_area)

input = ['Total Bsmt SF', '1st Flr SF', '2nd Flr SF']
def add_areas(df):
    return df['Total Bsmt SF'] + df['1st Flr SF'] + df['2nd Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=add_areas)
total_area = Feature(input=input, transformer=transformer, name='Total Area Calculation')
all_features.append(total_area)

input = ['Full Bath', 'Half Bath', 'Bsmt Full Bath', 'Bsmt Half Bath']
def calc_bath(df):
    return df['Full Bath'] + df['Half Bath'] + df['Bsmt Full Bath'] + df['Bsmt Half Bath']
transformer = ballet.eng.SimpleFunctionTransformer(func=calc_bath)
baths = Feature(input=input, transformer=transformer, name='Bathroom Count')
all_features.append(baths)

input = ['Garage Area', 'Garage Cars']
def calc_garage_per_car(df):
    df['Garage Area Per Car'] = df['Garage Area'] / df['Garage Cars']
    df.loc[~np.isfinite(df['Garage Area Per Car']), 'Garage Area Per Car'] = 0
    return df['Garage Area Per Car']
transformer = ballet.eng.SimpleFunctionTransformer(func=calc_garage_per_car)
cars = Feature(input=input, transformer=transformer, name='Garage Area Per Car')
all_features.append(cars)

input = ['Alley']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
all_features.append(misc_fill)

input = ['Bsmt Cond']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
all_features.append(misc_fill)

input = ['Bsmt Qual']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
all_features.append(misc_fill)

input = ['Bsmt Exposure']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
all_features.append(misc_fill)

input = ['BsmtFin Type 1']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
all_features.append(misc_fill)

input = ['BsmtFin Type 2']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
all_features.append(misc_fill)

input = ['Fireplace Qu']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
all_features.append(misc_fill)

input = ['Garage Type']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
all_features.append(misc_fill)

input = ['Garage Finish']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
all_features.append(misc_fill)

input = ['Garage Qual']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
all_features.append(misc_fill)

input = ['Garage Cond']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
all_features.append(misc_fill)

input = ['Pool QC']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
all_features.append(misc_fill)

input = ['Fence']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
all_features.append(misc_fill)

input = ['Misc Feature']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
all_features.append(misc_fill)
