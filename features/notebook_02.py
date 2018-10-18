# taken from https://www.kaggle.com/erikbruin/house-prices-lasso-xgboost-and-a-detailed-eda
# Adapted into the ballet framework
import ballet
import ballet.eng
from ballet import Feature
import numpy as np
import pandas as pd
import sklearn

features = []

input = ['Yr Sold']
year = Feature(input=input, transformer=sklearn.preprocessing.OneHotEncoder(), name='Year Categorical')
features.append(year)

input = ['Mo Sold']
month = Feature(input=input, transformer=sklearn.preprocessing.OneHotEncoder(), name='Month Categorical')
features.append(month)

input = ['Garage Yr Blt', 'Year Built']
def fill_garage(df):
    new_garage = df['Garage Yr Blt'].copy()
    mask = pd.isnull(new_garage)
    new_garage[mask] = df['Year Built'][mask]
    return new_garage
transformer = ballet.eng.SimpleFunctionTransformer(func=fill_garage)
garage = Feature(input=input, transformer=transformer)
features.append(garage)

input = ['Heating']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
factor = Feature(input=input, transformer=transformer)
features.append(factor)

input = ['Foundation']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
factor = Feature(input=input, transformer=transformer)
features.append(factor)

input = ['Roof Style']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
factor = Feature(input=input, transformer=transformer)
features.append(factor)

input = ['Land Contour']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
factor = Feature(input=input, transformer=transformer)
features.append(factor)

input = ['Bldg Type']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
factor = Feature(input=input, transformer=transformer)
features.append(factor)

input = ['House Style']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
factor = Feature(input=input, transformer=transformer)
features.append(factor)

input = ['Neighborhood']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
factor = Feature(input=input, transformer=transformer)
features.append(factor)

input = ['Condition 1']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
factor = Feature(input=input, transformer=transformer)
features.append(factor)

input = ['Condition 2']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
factor = Feature(input=input, transformer=transformer)
features.append(factor)

input = ['Kitchen Qual']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
factor = Feature(input=input, transformer=transformer)
features.append(factor)

# PoolQC : data description says NA means "No Pool".
# That make sense, given the huge ratio of missing value (+99%)
# and majority of houses have no Pool at all in general.
input = ['Pool QC']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer, name='PoolQC Misc Fill')
features.append(misc_fill)

input = ['Misc Feature']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer, name='PoolQC Misc Fill')
features.append(misc_fill)

input = ['Fence']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer, name='Fence Misc Fill')
features.append(misc_fill)

input = ['Alley']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer, name='Alley Misc Fill')
features.append(misc_fill)

input = ['Lot Frontage', 'Neighborhood']
def neighborhood_null(df):
    df['Neighborhood'].fillna('None')
    return df
def impute_lot_frontage(df):
    frontage = df['Lot Frontage']
    return frontage.fillna(frontage.median())
transformer = [ballet.eng.SimpleFunctionTransformer(func=neighborhood_null), ballet.eng.GroupedFunctionTransformer(func=impute_lot_frontage, groupby_kwargs={'by': 'Neighborhood'})]
frontage_feature = Feature(input=input, transformer=transformer,  name='Lot Frontage Fill')
features.append(frontage_feature)

input = ['Fireplace Qu']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
fill = Feature(input=input, transformer=transformer)
features.append(fill)

input = ['Garage Type']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
garage_none_fill = Feature(input=input, transformer=transformer)
features.append(garage_none_fill)

input = ['Garage Finish']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
garage_none_fill = Feature(input=input, transformer=transformer)
features.append(garage_none_fill)

input = ['Garage Qual']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
garage_none_fill = Feature(input=input, transformer=transformer)
features.append(garage_none_fill)

input = ['Garage Cond']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
garage_none_fill = Feature(input=input, transformer=transformer)
features.append(garage_none_fill)

input = ['Mas Vnr Type']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
mason_fill = Feature(input=input, transformer=transformer, name='Mason Fill None')
features.append(mason_fill)

input = ['MS SubClass']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
ms_fill = Feature(input=input, transformer=transformer, name='MS Fill None')
features.append(ms_fill)

input = ['Mas Vnr Area']
transformer = ballet.eng.missing.NullFiller()
mason_fill = Feature(input=input, transformer=transformer, name='Mason Fill Zero')
features.append(mason_fill)

input = ['Bsmt Cond']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
garage_none_fill = Feature(input=input, transformer=transformer)
features.append(garage_none_fill)

input = ['Full Bath', 'Half Bath', 'Bsmt Full Bath', 'Bsmt Half Bath']
def calc_bath(df):
    return df['Full Bath'] + 0.5 * df['Half Bath'] + df['Bsmt Full Bath'] + 0.5 * df['Bsmt Half Bath']
transformer = ballet.eng.SimpleFunctionTransformer(func=calc_bath)
baths = Feature(input=input, transformer=transformer, name='Bathroom Count')
features.append(baths)

input = ['Yr Sold', 'Year Remod/Add']
def calc_age(df):
    return df['Yr Sold'] - df['Year Remod/Add']
transformer = ballet.eng.SimpleFunctionTransformer(func=calc_age)
age = Feature(input=input, transformer=transformer, name='Age')
features.append(age)

input = ['Year Built', 'Year Remod/Add']
def is_remod(df):
    return df['Year Built'] != df['Year Remod/Add']
transformer = ballet.eng.SimpleFunctionTransformer(func=is_remod)
remod = Feature(input=input, transformer=transformer, name='Remodeled')
features.append(remod)

input = ['Gr Liv Area', 'Total Bsmt SF']
def total_area(df):
    return df['Gr Liv Area'] + df['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=total_area)
area = Feature(input=input, transformer=transformer, name='Total Area')
features.append(area)
