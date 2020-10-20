import numpy as np
from allennlp_dataframe_mapper.transforms import RegistrableTransform


@RegistrableTransform.register("logarithmer")
class Logarithmer(RegistrableTransform):
    def fit(self, X) -> "Logarithmer":
        return self

    def transform(self, X):
        return np.log1p(X)

    def interse_tranform(self, X):
        return np.expm1(X)
