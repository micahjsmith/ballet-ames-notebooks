# taken from https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-for-house-prices
# Adapted into the ballet framework
import ballet
import ballet.eng
from ballet import Feature
import numpy as np
import pandas as pd
import sklearn

# reset the feature set
features = []

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

# PoolQC : data description says NA means "No Pool". 
# That make sense, given the huge ratio of missing value (+99%) 
# and majority of houses have no Pool at all in general.
input = ['Pool QC']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Misc Feature']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Alley']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Fence']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer, name='Fence Misc Fill')
features.append(misc_fill)

# PoolQC : data description says NA means "No Pool". 
# That make sense, given the huge ratio of missing value (+99%) 
# and majority of houses have no Pool at all in general.
input = ['Garage Type']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
garage_none_fill = Feature(input=input, transformer=transformer, name='Garage Missing Fill None')
features.append(garage_none_fill)

# PoolQC : data description says NA means "No Pool". 
# That make sense, given the huge ratio of missing value (+99%) 
# and majority of houses have no Pool at all in general.
input = ['Garage Finish']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
garage_none_fill = Feature(input=input, transformer=transformer, name='Garage Missing Fill None')
features.append(garage_none_fill)

# PoolQC : data description says NA means "No Pool". 
# That make sense, given the huge ratio of missing value (+99%) 
# and majority of houses have no Pool at all in general.
input = ['Garage Qual']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
garage_none_fill = Feature(input=input, transformer=transformer, name='Garage Missing Fill None')
features.append(garage_none_fill)

# PoolQC : data description says NA means "No Pool". 
# That make sense, given the huge ratio of missing value (+99%) 
# and majority of houses have no Pool at all in general.
input = ['Garage Cond']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
garage_none_fill = Feature(input=input, transformer=transformer, name='Garage Missing Fill None')
features.append(garage_none_fill)

# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath :
# missing values are likely zero for having no basement

input = ['BsmtFin SF 1']
transformer = ballet.eng.missing.NullFiller()
basement_fill = Feature(input=input, transformer=transformer, name='Basement Fill None')
features.append(basement_fill)

# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath :
# missing values are likely zero for having no basement

input = ['BsmtFin SF 2']
transformer = ballet.eng.missing.NullFiller()
basement_fill = Feature(input=input, transformer=transformer, name='Missing Fill None')
features.append(basement_fill)

# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath :
# missing values are likely zero for having no basement

input = ['Bsmt Unf SF']
transformer = ballet.eng.missing.NullFiller()
basement_fill = Feature(input=input, transformer=transformer, name='Missing Fill None')
features.append(basement_fill)

# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath :
# missing values are likely zero for having no basement

input = ['Total Bsmt SF']
transformer = ballet.eng.missing.NullFiller()
basement_fill = Feature(input=input, transformer=transformer, name='Missing Fill None')
features.append(basement_fill)

# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath :
# missing values are likely zero for having no basement

input = ['Bsmt Full Bath']
transformer = ballet.eng.missing.NullFiller()
basement_fill = Feature(input=input, transformer=transformer, name='Missing Fill None')
features.append(basement_fill)

# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath :
# missing values are likely zero for having no basement

input = ['Bsmt Half Bath']
transformer = ballet.eng.missing.NullFiller()
basement_fill = Feature(input=input, transformer=transformer, name='Missing Fill None')
features.append(basement_fill)

input = ['Mas Vnr Type']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
mason_fill = Feature(input=input, transformer=transformer, name='Mason Fill None')
features.append(mason_fill)

input = ['Mas Vnr Area']
transformer = ballet.eng.missing.NullFiller()
mason_fill = Feature(input=input, transformer=transformer, name='Mason Fill Zero')
features.append(mason_fill)

input = ['MS SubClass']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
ms_fill = Feature(input=input, transformer=transformer, name='MS Fill None')
features.append(ms_fill)

input = ['Fireplace Qu']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
fill = Feature(input=input, transformer=transformer, name='FireplaceQu Fill Zero')
features.append(fill)

input = ['Total Bsmt SF', '1st Flr SF', '2nd Flr SF']
def add_areas(df):
    total_sf = df['Total Bsmt SF'] + df['1st Flr SF'] + df['2nd Flr SF']
    return total_sf.fillna(0)
transformer = ballet.eng.SimpleFunctionTransformer(func=add_areas)
total_area = Feature(input=input, transformer=transformer, name='Total Area Calculation')
features.append(total_area)

input = ['Mo Sold']
month = Feature(input=input, transformer=sklearn.preprocessing.OneHotEncoder(), name='Month Categorical')
features.append(month)

input = ['Yr Sold']
year = Feature(input=input, transformer=sklearn.preprocessing.OneHotEncoder(), name='Year Categorical')
features.append(year)
