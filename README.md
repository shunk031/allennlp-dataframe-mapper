# sklearn-pandas plugin for AllenNLP

![Python 3.7](https://img.shields.io/badge/python-3.7%2B-brightgreen.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


[sklearn-pandas](https://github.com/scikit-learn-contrib/sklearn-pandas) plugin / wrapper for [AllenNLP](https://github.com/allenai/allennlp).

## Install

```sh
$ pip install git+ssh://git@github.com/shunk031/allennlp-dataframe-mapper.git
```

## Usage

### Config

`mapper_iris.jsonnet`

```json
{
    "type": "default",
    "features": [
        [["sepal length (cm)"], null],
        [["sepal width (cm)"], null],
        [["petal length (cm)"], null],
        [["petal width (cm)"], null],
        [["species"], [{"type": "flatten"}, {"type": "label-encoder"}]],
    ],
    "df_out": true,
}
```

### Mapper

```python
from allennlp.common import Params
from allennlp_dataframe_mapper import DataFrameMapper

params = Params.from_file("mapper_iris.jsonnet")
mapper = DataFrameMapper.from_params(params=params)

print(mapper)
# DataFrameMapper(df_out=True,
#                 features=[(['sepal length (cm)'], None, {}),
#                           (['sepal width (cm)'], None, {}),
#                           (['petal length (cm)'], None, {}),
#                           (['petal width (cm)'], None, {}),
#                           (['species'], [FlattenTransform(), LabelEncoder()], {})])

mapper.fit_transform(df)
```

## License

MIT
