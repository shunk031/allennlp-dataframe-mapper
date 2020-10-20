import logging
from typing import Tuple

from allennlp_dataframe_mapper.transforms import RegistrableTransform
from sklearn.preprocessing import LabelEncoder as _LabelEncoder
from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler
from sklearn.preprocessing import StandardScaler as _StandardScaler

logger = logging.getLogger(__name__)


@RegistrableTransform.register("standard-scalar")
class StandardScaler(_StandardScaler):
    def __init__(
        self, copy: bool = True, with_mean: bool = True, with_std: bool = True
    ) -> None:
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)


@RegistrableTransform.register("min-max-scalar")
class MinMaxScalar(_MinMaxScaler):
    def __init__(self, feature_range: Tuple[int, int] = (0, 1), copy: bool = True):
        super().__init__(feature_range=feature_range, copy=copy)


@RegistrableTransform.register("label-encoder")
class LabelEncoder(_LabelEncoder):
    def __init__(self):
        super().__init__()
