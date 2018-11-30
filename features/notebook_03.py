# taken from https://www.kaggle.com/alvaromendoza/manual-fe-sklearn-pandas-pipelines-ensembling/output
# Adapted into the ballet framework
import ballet
import ballet.eng
from ballet import Feature
import numpy as np
import pandas as pd
import sklearn

features = []

input = ['MS SubClass']
def add_missing_0_to_mssubclass(df):
    """Zeros in 020-090 get cut off. This function prepends them back."""

    return df['MS SubClass'].apply(
        lambda x: '00' + str(x) if len(str(x)) == 2 else '0' + str(x))
transformer = [ballet.eng.SimpleFunctionTransformer(func=add_missing_0_to_mssubclass), sklearn.preprocessing.OneHotEncoder()]
feature = Feature(input=input, transformer=transformer)
features.append(feature)

input = ['Enclosed Porch', '3Ssn Porch', 'Open Porch SF']
def calc_porch_type(df):    
    # Porch features
    total_porch_area = df.apply(np.sum, axis=1)
    porch_type = pd.Series('Missing', index=df.index)
    porch_type[(total_porch_area == df['Enclosed Porch']) & (df['Enclosed Porch'] > 0) ] = 'Enclosed'
    porch_type[(total_porch_area == df['3Ssn Porch']) & (df['3Ssn Porch'] > 0) ] = '3Ssn'
    porch_type[(total_porch_area == df['Open Porch SF']) & (df['Open Porch SF'] > 0) ] = 'Open'
    return porch_type
transformer = [ballet.eng.SimpleFunctionTransformer(func=calc_porch_type), sklearn.preprocessing.OneHotEncoder()]
porch = Feature(input=input, transformer=transformer, name='Porch Type Calculation')
features.append(porch)

input = ['Enclosed Porch', '3Ssn Porch', 'Open Porch SF']
def calc_porch_area(df):
    total_area = df['Enclosed Porch'] + df['3Ssn Porch'] + df['Open Porch SF']
    return total_area.fillna(0)
transformer = ballet.eng.SimpleFunctionTransformer(func=calc_porch_area)
total_area = Feature(input=input, transformer=transformer, name='Porch Area Calculation')
features.append(total_area)

input = ['Total Bsmt SF', '1st Flr SF', '2nd Flr SF']
def add_areas(df):
    total_area = df['Total Bsmt SF'] + df['1st Flr SF'] + df['2nd Flr SF']
    return total_area.fillna(0)
transformer = ballet.eng.SimpleFunctionTransformer(func=add_areas)
total_area = Feature(input=input, transformer=transformer, name='Total Area Calculation')
features.append(total_area)

input = ['Full Bath', 'Half Bath', 'Bsmt Full Bath', 'Bsmt Half Bath']
def calc_bath(df):
    total_area = df['Full Bath'] + df['Half Bath'] + df['Bsmt Full Bath'] + df['Bsmt Half Bath']
    return total_area.fillna(0)
transformer = ballet.eng.SimpleFunctionTransformer(func=calc_bath)
baths = Feature(input=input, transformer=transformer, name='Bathroom Count')
features.append(baths)

input = ['Garage Area', 'Garage Cars']
def calc_garage_per_car(df):
    df['Garage Area Per Car'] = df['Garage Area'] / df['Garage Cars']
    df.loc[~np.isfinite(df['Garage Area Per Car']), 'Garage Area Per Car'] = 0
    return df['Garage Area Per Car']
transformer = ballet.eng.SimpleFunctionTransformer(func=calc_garage_per_car)
cars = Feature(input=input, transformer=transformer, name='Garage Area Per Car')
features.append(cars)

input = ['Alley']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Bsmt Cond']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Bsmt Qual']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Bsmt Exposure']
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

input = ['Fireplace Qu']
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

input = ['Garage Qual']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Garage Cond']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
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
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)