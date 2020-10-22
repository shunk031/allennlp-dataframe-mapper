from allennlp.common import Registrable
from sklearn.base import BaseEstimator, TransformerMixin


class RegistrableTransform(BaseEstimator, TransformerMixin, Registrable):
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def transform(self, *args, **kwargs):
        raise NotImplementedError

    def inverse_transform(self, *args, **kwargs):
        raise NotImplementedError
