"""Tests for converting to a stripped (no qualifier) query."""
import pathlib
import tempfile
import unittest

import pytest

from mphrqe.data.converter import StrippedSPARQLResultBuilder
from mphrqe.data.mapping import get_entity_mapper


class Test_assert_query_validity(unittest.TestCase):

    @pytest.mark.full_data
    def test_one(self):
        """Test creating a single query graph."""
        clazz = StrippedSPARQLResultBuilder()
        queryBuilder = clazz(3, 3)
        queryBuilder.set_subject_predicate_object(0,
                                                  "https://www.wikidata.org/wiki/Q167520",
                                                  "https://www.wikidata.org/wiki/Property:P1411",
                                                  "?var1")
        queryBuilder.set_subject_predicate_object(1,
                                                  "https://www.wikidata.org/wiki/Q167520",
                                                  "https://www.wikidata.org/wiki/Property:P1346",
                                                  "?var1")
        queryBuilder.set_subject_predicate_object(2,
                                                  "?var1",
                                                  "https://www.wikidata.org/wiki/Property:P1411",
                                                  get_entity_mapper().get_target_entity_name())
        # def set_qualifier_rel_val(self, tripleIndex: int, qualifier_index: int, predicate: str, value: str):
        queryBuilder.set_qualifier_rel_val(0, 0, "https://www.wikidata.org/wiki/Property:P355", "https://www.wikidata.org/wiki/Q1033465")
        queryBuilder.set_qualifier_rel_val(0, 1, "https://www.wikidata.org/wiki/Property:P190", "https://www.wikidata.org/wiki/Q274670")
        queryBuilder.set_qualifier_rel_val(1, 2, "https://www.wikidata.org/wiki/Property:P161", "https://www.wikidata.org/wiki/Q28389")
        queryBuilder.set_diameter(2)
        queryBuilder.set_targets(["https://www.wikidata.org/wiki/Q1648927", "https://www.wikidata.org/wiki/Q10855195", "https://www.wikidata.org/wiki/Q901462"])
        theQuery = queryBuilder.build()
        with tempfile.TemporaryDirectory() as directory:
            # we use a normal file instead of a python temp file because a temp file is not guaranteed to be openable twice, which we need here
            result_file_path = pathlib.Path(directory) / "queries_file.txt"
            queryBuilder.store([theQuery], result_file_path)


if __name__ == '__main__':
    unittest.main()
