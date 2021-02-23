import hashlib
from typing import Optional

import numpy as np
from allennlp_dataframe_mapper.transforms import RegistrableTransform
from overrides import overrides


@RegistrableTransform.register("hashname")
class HashName(RegistrableTransform):
    def __init__(self, ext: Optional[str] = None) -> None:
        super().__init__()
        self._ext = ext

    def to_hash_name(self, url: str) -> str:
        hash_name = hashlib.md5(url.encode("utf8")).hexdigest()
        if self._ext is not None:
            return hash_name + self._ext
        return hash_name

    @overrides
    def fit(self, *args, **kwargs):
        return self

    @overrides
    def transform(self, X):
        return np.vectorize(self.to_hash_name)(X)
