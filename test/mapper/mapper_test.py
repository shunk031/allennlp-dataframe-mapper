import numpy as np
import pandas as pd
import pytest
from allennlp.common import Params
from allennlp_dataframe_mapper.common.testing import AllenNlpDataFrameMapperTestCase
from allennlp_dataframe_mapper.mapper import DataFrameMapper
from allennlp_dataframe_mapper.transforms import (
    FlattenTransformer,
    LabelEncoder,
    Logarithmer,
    MinMaxScaler,
    StandardScaler,
)
from sklearn.datasets import load_iris, load_wine


@pytest.fixture
def wine_dataframe() -> pd.DataFrame:
    wine = load_wine()
    return pd.DataFrame(wine.data, columns=wine.feature_names)


@pytest.fixture
def iris_dataframe():
    iris = load_iris()
    return pd.DataFrame(
        data={
            iris.feature_names[0]: iris.data[:, 0],
            iris.feature_names[1]: iris.data[:, 1],
            iris.feature_names[2]: iris.data[:, 2],
            iris.feature_names[3]: iris.data[:, 3],
            "species": np.array([iris.target_names[e] for e in iris.target]),
        }
    )


class TestMapper(AllenNlpDataFrameMapperTestCase):
    def test_mapper_iris(self, iris_dataframe) -> None:
        params = Params.from_file(self.FIXTURES_ROOT / "mapper" / "mapper_iris.jsonnet")
        mapper = DataFrameMapper.from_params(params=params)

        features = mapper.features
        assert features[0] == (["sepal length (cm)"], None, {})
        assert features[1] == (["sepal width (cm)"], None, {})
        assert features[2] == (["petal length (cm)"], None, {})
        assert features[3] == (["petal width (cm)"], None, {})

        assert features[4][0] == ["species"]
        assert isinstance(features[4][1][0], FlattenTransformer)
        assert isinstance(features[4][1][1], LabelEncoder)
        assert features[4][2] == {}

        mapper.fit_transform(iris_dataframe)
        assert mapper.transformed_names_ == [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
            "species",
        ]

    def test_mapper_wine(self, wine_dataframe) -> None:
        params = Params.from_file(self.FIXTURES_ROOT / "mapper" / "mapper_wine.jsonnet")
        mapper = DataFrameMapper.from_params(params=params)

        features = mapper.features
        assert features[0][0] == ["alcohol"]
        assert isinstance(features[0][1], StandardScaler)
        assert features[0][2] == {"alias": "alcohol_standard_scalar"}

        assert features[1][0] == ["alcohol"]
        assert isinstance(features[1][1], MinMaxScaler)
        assert features[1][2] == {"alias": "alcohol_min_max_scalar"}

        assert features[2] == (["malic_acid"], None, {})
        assert features[3] == (["ash"], None, {})
        assert features[4] == (["alcalinity_of_ash"], None, {})

        assert features[5][0] == ["magnesium"]
        assert isinstance(features[5][1][0], Logarithmer)
        assert isinstance(features[5][1][1], StandardScaler)
        assert features[5][2] == {"alias": "magnesium_log_standard_scalar"}

        assert features[6][0] == ["magnesium"]
        assert isinstance(features[6][1][0], Logarithmer)
        assert isinstance(features[6][1][1], MinMaxScaler)
        assert features[6][2] == {"alias": "magnesium_log_min_max_scalar"}

        assert features[7] == (["total_phenols"], None, {})
        assert features[8] == (["flavanoids"], None, {})
        assert features[9] == (["nonflavanoid_phenols"], None, {})
        assert features[10] == (["proanthocyanins"], None, {})
        assert features[11] == (["color_intensity"], None, {})
        assert features[12] == (["hue"], None, {})
        assert features[13] == (["od280/od315_of_diluted_wines"], None, {})

        assert features[14][0] == ["proline"]
        assert isinstance(features[14][1], Logarithmer)
        assert features[14][2] == {}

        mapper.fit_transform(wine_dataframe)

        assert mapper.transformed_names_ == [
            "alcohol_standard_scalar",
            "alcohol_min_max_scalar",
            "malic_acid",
            "ash",
            "alcalinity_of_ash",
            "magnesium_log_standard_scalar",
            "magnesium_log_min_max_scalar",
            "total_phenols",
            "flavanoids",
            "nonflavanoid_phenols",
            "proanthocyanins",
            "color_intensity",
            "hue",
            "od280/od315_of_diluted_wines",
            "proline",
        ]
