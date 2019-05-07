from ballet import Feature
import ballet.eng
import numpy as np
import sklearn.impute


input = "Total Bsmt SF"
transformer = [
    ballet.eng.ConditionalTransformer(
        lambda ser: ser.skew() > 0.75,
        lambda ser: np.log1p(ser)),
    ballet.eng.NamedFramer("Total Bsmt SF".replace(" ", "")),
    sklearn.impute.SimpleImputer(strategy='mean'),
]
name = "Total Bsmt SF unskewed"
feature = Feature(input=input, transformer=transformer, name=name)
