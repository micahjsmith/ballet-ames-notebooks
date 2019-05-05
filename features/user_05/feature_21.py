from ballet import Feature
import sklearn.preprocessing


input = "Yr Sold"
transformer = sklearn.preprocessing.OneHotEncoder()
name = "Year sold type"
feature = Feature(input=input, transformer=transformer, name=name)
