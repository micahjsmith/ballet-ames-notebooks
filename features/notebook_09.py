# taken from https://www.kaggle.com/tannercarbonati/detailed-data-analysis-ensemble-modeling
# Adapted into the ballet framework
# Heavily based on his words and not his code.
import ballet
import ballet.eng
from ballet import Feature
import numpy as np
import pandas as pd
import sklearn

features = []

input = ['Bsmt Cond']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
misc_fill = Feature(input=input, transformer=transformer)
features.append(misc_fill)

input = ['Bsmt Qual']
transformer = [ballet.eng.missing.NullFiller(replacement="None", isnull=pd.isnull), sklearn.preprocessing.OneHotEncoder()]
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

def has_qual(df):
    return (df['BsmtFin SF 2'] > 0) * 1
input = ['BsmtFin SF 2']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['Bsmt Unf SF'] <= 778.667) * 1
input = ['Bsmt Unf SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df['Bsmt Unf SF'] > 778.667) & (df['Bsmt Unf SF'] <= 1557.333)) * 1
input = ['Bsmt Unf SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['Bsmt Unf SF'] > 1557.333) * 1
input = ['Bsmt Unf SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['Total Bsmt SF'] <= 509.5) * 1
input = ['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df['Total Bsmt SF'] > 509.5) & (df['Total Bsmt SF'] <= 1019.0)) * 1
input = ['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df['Total Bsmt SF'] > 1019.0) & (df['Total Bsmt SF'] <= 1528.5)) * 1
input = ['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df['Total Bsmt SF'] > 1528.5) & (df['Total Bsmt SF'] <= 2038.0)) * 1
input = ['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df['Total Bsmt SF'] > 2038.0) & (df['Total Bsmt SF'] <= 2547.5)) * 1
input = ['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df['Total Bsmt SF'] > 2547.5) & (df['Total Bsmt SF'] <= 3057.0)) * 1
input = ['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df['Total Bsmt SF'] > 3057.0) & (df['Total Bsmt SF'] <= 3566.5)) * 1
input = ['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['Total Bsmt SF'] > 3566.5) * 1
input = ['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['1st Flr SF'] <= 1127.5) * 1
input = ['1st Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df['1st Flr SF'] > 1127.5) & (df['1st Flr SF'] <= 1921.0)) * 1
input = ['1st Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df['1st Flr SF'] > 1921.0) & (df['1st Flr SF'] <= 2714.5)) * 1
input = ['1st Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df['1st Flr SF'] > 2714.5) & (df['1st Flr SF'] <= 3508.0)) * 1
input = ['1st Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df['1st Flr SF'] > 3508.0) & (df['1st Flr SF'] <= 4301.5)) * 1
input = ['1st Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['1st Flr SF'] > 4301.5) * 1
input = ['1st Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['2nd Flr SF'] <= 310.333) * 1
input = ['2nd Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df['2nd Flr SF'] > 310.333) & (df['2nd Flr SF'] <= 620.667)) * 1
input = ['2nd Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df['2nd Flr SF'] > 620.667) & (df['2nd Flr SF'] <= 931.0)) * 1
input = ['2nd Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df['2nd Flr SF'] > 931.0) & (df['2nd Flr SF'] <= 1241.333)) * 1
input = ['2nd Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df['2nd Flr SF'] > 1241.333) & (df['2nd Flr SF'] <= 1551.667)) * 1
input = ['2nd Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df['2nd Flr SF'] > 1551.667) * 1
input = ['2nd Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)