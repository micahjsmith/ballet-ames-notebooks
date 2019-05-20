#!/usr/bin/env python3

import pathlib
import textwrap

import ames.load_data

X, _ = ames.load_data.load_data()

mask = (X.dtypes == 'object') | (X.dtypes == 'category')
nonnumeric = X.columns[mask]
numeric = X.columns[~mask]

i = 1

for c in nonnumeric:
    feature = textwrap.dedent("""\
    from ballet import Feature
    import ballet.eng
    import sklearn.preprocessing


    input = ["{colname}"]
    transformer = [
        ballet.eng.missing.NullFiller(replacement="None"),
        sklearn.preprocessing.OneHotEncoder(),
    ]
    name = "{colname} type"
    feature = Feature(input=input, transformer=transformer, name=name)
    """.format(colname=c))

    filename = pathlib.Path().joinpath(
        'features', 'user_01', f'feature_{i:02d}.py')
    with filename.open('w') as f:
        f.write(feature)
    i += 1

for c in numeric:
    feature = textwrap.dedent("""\
    from ballet import Feature
    import ballet.eng
    import numpy as np
    import sklearn.impute
    
    
    input = "{colname}"
    transformer = [
        ballet.eng.ConditionalTransformer(
            lambda ser: ser.skew() > 0.75,
            lambda ser: np.log1p(ser)),
        ballet.eng.NamedFramer("{colname}".replace(" ", "")),
        sklearn.impute.SimpleImputer(strategy='mean'),
    ]
    name = "{colname} unskewed"
    feature = Feature(input=input, transformer=transformer, name=name)
    """.format(colname=c))

    filename = pathlib.Path().joinpath(
        'features', 'user_01', f'feature_{i:02d}.py')
    with filename.open('w') as f:
        f.write(feature)
    i += 1

print('done')
