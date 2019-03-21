from ballet import Feature
from ballet.eng import SimpleFunctionTransformer

input = ['Enclosed Porch', '3Ssn Porch', 'Open Porch SF']
transformer = SimpleFunctionTransformer(lambda df: df.sum(axis=1))
name = 'Computed Porch Area'
feature = Feature(input=input, transformer=transformer, name=name)
