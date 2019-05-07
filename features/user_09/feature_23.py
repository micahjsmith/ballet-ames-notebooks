from ballet import Feature
import ballet.eng


input = "2nd Flr SF"
transformer = ballet.eng.SimpleFunctionTransformer(
    lambda ser: (ser <= 310.333).astype(int))
name = "Second floor square feet group 1"
feature = Feature(input=input, transformer=transformer, name=name)
