from ballet import Feature
from ballet.eng import GroupwiseTransformer, SimpleFunctionTransformer
from sklearn.impute import SimpleImputer


input = ['Lot Frontage', 'Neighborhood']
transformer = [
    SimpleFunctionTransformer(lambda df: df.set_index('Neighborhood',
                                                      append=True)),
    GroupwiseTransformer(
        SimpleImputer(strategy='median'),
        groupby_kwargs={'level': 'Neighborhood'},
        handle_error='ignore',
    ),
    SimpleImputer(strategy='median'),
]
feature = Feature(input=input,
                  transformer=transformer,
                  name='Imputed Lot Frontage')
