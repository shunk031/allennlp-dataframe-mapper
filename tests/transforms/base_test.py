from allennlp.common import Registrable
from allennlp_dataframe_mapper.common.testing import AllenNlpDataFrameMapperTestCase
from allennlp_dataframe_mapper.transforms import RegistrableTransform


class TestRegistrableTransform(AllenNlpDataFrameMapperTestCase):
    def test_registrable_transform(self) -> None:

        registered_list = set(Registrable._registry[RegistrableTransform].keys())

        assert registered_list == set(
            [
                "standard-scaler",
                "min-max-scaler",
                "label-encoder",
                "logarithmer",
                "flatten",
                "hashname",
            ]
        )

        assert (
            RegistrableTransform.by_name("standard-scaler").__name__ == "StandardScaler"
        )
        assert RegistrableTransform.by_name("min-max-scaler").__name__ == "MinMaxScaler"
        assert RegistrableTransform.by_name("label-encoder").__name__ == "LabelEncoder"
        assert RegistrableTransform.by_name("logarithmer").__name__ == "Logarithmer"
        assert RegistrableTransform.by_name("flatten").__name__ == "FlattenTransformer"
        assert RegistrableTransform.by_name("hashname").__name__ == "HashName"
