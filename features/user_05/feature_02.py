from ballet import Feature
import sklearn.impute
import sklearn.preprocessing


input = ['MS Zoning']
transformer = [
    sklearn.impute.SimpleImputer(strategy='most_frequent'),
    sklearn.preprocessing.OneHotEncoder()
]
name = 'MS Zoning Fill'
feature = Feature(input=input, transformer=transformer, name=name)
