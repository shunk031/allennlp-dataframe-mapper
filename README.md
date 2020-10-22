# AllenNLP integration for sklearn-pandas

![CI](https://github.com/shunk031/allennlp-dataframe-mapper/workflows/CI/badge.svg?branch=master)
![Release](https://github.com/shunk031/allennlp-dataframe-mapper/workflows/Release/badge.svg)
![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue?logo=python)
[![PyPI](https://img.shields.io/pypi/v/allennlp-dataframe-mapper.svg)](https://pypi.python.org/pypi/allennlp-dataframe-mapper)

`allennlp-dataframe-mapper` is a Python library that provides [AllenNLP](https://github.com/allenai/allennlp) integration for [sklearn-pandas](https://github.com/scikit-learn-contrib/sklearn-pandas).

## Installation

Installing the library and dependencies is simple using `pip`.

```sh
$ pip allennlp-dataframe-mapper
```

## Example

This library enables users to specify the in a jsonnet config file.
Here is an example of the mapper for a famous [iris dataset](https://archive.ics.uci.edu/ml/datasets/iris).

### Config

`allennlp-dataframe-mapper` is specified the transformations of the mapper in `jsonnet` config file like following `mapper_iris.jsonnet`:

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

The mapper takes a param of transformations from the config file.
We can use the `fit_transform` shortcut to both fit the mapper and see what transformed data.

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
#                           (['species'], [FlattenTransformer(), LabelEncoder()], {})])

mapper.fit_transform(df)
```

## License

MIT
