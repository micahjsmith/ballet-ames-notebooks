from ballet import Feature
import ballet.eng
import pandas as pd

input = ["Garage Cars"]

transformer = ballet.eng.missing.NullFiller(isnull=pd.isnull)

feature = Feature(input=input, transformer=transformer)
