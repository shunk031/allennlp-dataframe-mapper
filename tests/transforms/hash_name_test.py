import numpy as np
import pytest
from allennlp.common import Params
from allennlp_dataframe_mapper.common.testing import AllenNlpDataFrameMapperTestCase
from allennlp_dataframe_mapper.transforms import HashName, RegistrableTransform


class TestHashName(AllenNlpDataFrameMapperTestCase):
    def test_hash_name_from_params(self):
        params = Params.from_file(
            self.FIXTURES_ROOT / "transforms" / "hash_name" / "hash_name.jsonnet"
        )
        transform = RegistrableTransform.from_params(params=params)
        assert isinstance(transform, HashName)

    @pytest.mark.parametrize(
        "url, expected",
        (
            ("http://example.com/", "a6bf1757fff057f266b697df9cf176fd"),
            ("http://abehiroshi.la.coocan.jp/", "99daa109463db188c4160ee06a6ca8f7"),
        ),
    )
    def test_hash_name(self, url, expected):
        hash_name = HashName()
        url_hashed = hash_name.transform(url)

        expected_np = np.array(expected)
        assert np.array_equal(url_hashed, expected_np)

    def test_hash_names(self):

        urls = ["http://example.com/", "http://abehiroshi.la.coocan.jp/"]
        hash_name = HashName()
        url_hashed = hash_name.transform(urls)

        expected = np.array(
            [
                "a6bf1757fff057f266b697df9cf176fd",
                "99daa109463db188c4160ee06a6ca8f7",
            ]
        )
        assert np.array_equal(url_hashed, expected)

    @pytest.mark.parametrize(
        "ext, url, expected",
        (
            (".png", "http://example.com/", "a6bf1757fff057f266b697df9cf176fd"),
            (
                ".jpg",
                "http://abehiroshi.la.coocan.jp/",
                "99daa109463db188c4160ee06a6ca8f7",
            ),
        ),
    )
    def test_hash_name_with_ext(self, ext, url, expected):
        hash_name = HashName(ext=ext)
        url_hashed = hash_name.transform(url)

        expected_np = np.array(expected + ext)
        assert np.array_equal(url_hashed, expected_np)
