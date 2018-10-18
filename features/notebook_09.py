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
    return (df > 0) * 1
input = ['BsmtFin SF 2']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df <= 778.667) * 1
input = ['Bsmt Unf SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df > 778.667) & (df <= 1557.333)) * 1
input = ['Bsmt Unf SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df > 1557.333) * 1
input = ['Bsmt Unf SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df <= 509.5) * 1
input = ['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df > 509.5) & (df <= 1019.0)) * 1
input = ['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df > 1019.0) & (df <= 1528.5)) * 1
input = ['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df > 1528.5) & (df <= 2038.0)) * 1
input = ['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df > 2038.0) & (df <= 2547.5)) * 1
input = ['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df > 2547.5) & (df <= 3057.0)) * 1
input = ['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df > 3057.0) & (df <= 3566.5)) * 1
input = ['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df > 3566.5) * 1
input = ['Total Bsmt SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df <= 1127.5) * 1
input = ['1st Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df > 1127.5) & (df <= 1921.0)) * 1
input = ['1st Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df > 1921.0) & (df <= 2714.5)) * 1
input = ['1st Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df > 2714.5) & (df <= 3508.0)) * 1
input = ['1st Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df > 3508.0) & (df <= 4301.5)) * 1
input = ['1st Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df > 4301.5) * 1
input = ['1st Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df <= 310.333) * 1
input = ['2nd Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df > 310.333) & (df <= 620.667)) * 1
input = ['2nd Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df > 620.667) & (df <= 931.0)) * 1
input = ['2nd Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df > 931.0) & (df <= 1241.333)) * 1
input = ['2nd Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return ((df > 1241.333) & (df <= 1551.667)) * 1
input = ['2nd Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)

def has_qual(df):
    return (df > 1551.667) * 1
input = ['2nd Flr SF']
transformer = ballet.eng.SimpleFunctionTransformer(func=has_qual)
qual = Feature(input=input, transformer=transformer)
features.append(qual)
