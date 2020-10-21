from typing import Dict, List, Optional, Tuple, Union

from allennlp.common import Params, Registrable
from sklearn_pandas import DataFrameMapper as _DataFrameMapper

from allennlp_dataframe_mapper.transforms import RegistrableTransform


def _build_transforms(transforms: Union[Params, List[Params]]):
    if isinstance(transforms, list):
        transforms = [RegistrableTransform.from_params(param) for param in transforms]
    else:
        transforms = RegistrableTransform.from_params(transforms)
    return transforms


def _build_feature(
    columns: List[Union[List[str], List[List[str]]]],
    transforms: Union[Params, List[Params]],
    options: Optional[Params] = None,
) -> Tuple[
    List[
        Union[
            List[str],
            List[List[str]],
        ],
    ],
    Union[RegistrableTransform, List[RegistrableTransform]],
    Dict[str, str],
]:
    if options is None:
        options = {}
    else:
        options = options.params
    return (columns, _build_transforms(transforms), options)


class DataFrameMapper(_DataFrameMapper, Registrable):
    def __init__(
        self,
        features: List[
            Union[
                List[str],
                List[List[str]],
                Optional[RegistrableTransform],
                Optional[List[RegistrableTransform]],
                Optional[Dict[str, str]],
            ],
        ],
        default: bool = False,
        sparse: bool = False,
        df_out: bool = False,
        input_df: bool = False,
    ):
        build_features = [_build_feature(*f) for f in features]
        super().__init__(
            build_features,
            default=default,
            sparse=sparse,
            df_out=df_out,
            input_df=input_df,
        )


DataFrameMapper.register("default")(DataFrameMapper)
