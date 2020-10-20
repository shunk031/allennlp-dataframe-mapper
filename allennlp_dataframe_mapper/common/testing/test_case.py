import pathlib

from allennlp.common.testing import AllenNlpTestCase


class AllenNlpDataFrameMapperTestCase(AllenNlpTestCase):
    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()
    MODULE_ROOT = PROJECT_ROOT / "allennlp_dataframe_mapper"
    TOOLS_ROOT = MODULE_ROOT / "tools"
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"
