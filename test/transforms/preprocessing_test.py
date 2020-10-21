from allennlp.common import Params
from allennlp_dataframe_mapper.common.testing import AllenNlpDataFrameMapperTestCase
from allennlp_dataframe_mapper.transforms import (
    FlattenTransformer,
    LabelEncoder,
    Logarithmer,
    MinMaxScaler,
    RegistrableTransform,
    StandardScaler,
)


class TestPreprocessingTransforms(AllenNlpDataFrameMapperTestCase):
    def test_standard_scalar(self):
        params = Params.from_file(
            self.FIXTURES_ROOT
            / "transforms"
            / "preprocessing"
            / "standard_scalar.jsonnet"
        )
        transform = RegistrableTransform.from_params(params=params)
        assert isinstance(transform, StandardScaler)

    def test_min_max_scalar(self):
        params = Params.from_file(
            self.FIXTURES_ROOT
            / "transforms"
            / "preprocessing"
            / "min_max_scalar.jsonnet"
        )
        transform = RegistrableTransform.from_params(params=params)
        assert isinstance(transform, MinMaxScaler)

    def test_label_encoder(self):
        params = Params.from_file(
            self.FIXTURES_ROOT
            / "transforms"
            / "preprocessing"
            / "label_encoder.jsonnet"
        )
        transform = RegistrableTransform.from_params(params=params)
        assert isinstance(transform, LabelEncoder)

    def test_logarithmer(self):
        params = Params.from_file(
            self.FIXTURES_ROOT / "transforms" / "preprocessing" / "logarithmer.jsonnet"
        )
        transform = RegistrableTransform.from_params(params=params)
        assert isinstance(transform, Logarithmer)

    def test_flatten(self):
        params = Params.from_file(
            self.FIXTURES_ROOT / "transforms" / "preprocessing" / "flatten.jsonnet"
        )
        transform = RegistrableTransform.from_params(params=params)
        assert isinstance(transform, FlattenTransformer)
