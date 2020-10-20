from allennlp_dataframe_mapper.transforms import RegistrableTransform


@RegistrableTransform.register("flatten")
class FlattenTransform(RegistrableTransform):
    def fit(self, X) -> "FlattenTransform":
        return self

    def transform(self, X):
        return X.flatten()
