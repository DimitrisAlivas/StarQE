"""Tests for reification."""
import pathlib
import tempfile
import unittest

import pytest

from mphrqe.data.converter import ProtobufBuilder
from mphrqe.data.loader import read_queries_from_proto_with_reification
from mphrqe.data.mapping import get_entity_mapper, get_relation_mapper


class Test_assert_query_validity(unittest.TestCase):

    @pytest.mark.full_data
    def test_one(self):
        """Test creating a single query graph."""
        queryBuilder = ProtobufBuilder(3, 3)
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
                                                  "https://www.wikidata.org/wiki/Q505449")
        # def set_qualifier_rel_val(self, tripleIndex: int, qualifier_index: int, predicate: str, value: str):
        queryBuilder.set_qualifier_rel_val(0, 0, "https://www.wikidata.org/wiki/Property:P355", "https://www.wikidata.org/wiki/Q1033465")
        queryBuilder.set_qualifier_rel_val(0, 1, "https://www.wikidata.org/wiki/Property:P190", "https://www.wikidata.org/wiki/Q274670")
        queryBuilder.set_qualifier_rel_val(1, 2, "https://www.wikidata.org/wiki/Property:P161", "https://www.wikidata.org/wiki/Q28389")
        queryBuilder.set_diameter(2)
        queryBuilder.set_targets(["https://www.wikidata.org/wiki/Q1648927", "https://www.wikidata.org/wiki/Q10855195", "https://www.wikidata.org/wiki/Q901462"])
        theQuery = queryBuilder.build()
        with tempfile.TemporaryDirectory() as directory:
            # we use a normal file instead of a python temp file because a temp file is not guaranteed to be openable twice, which we need here
            proto_file_path = pathlib.Path(directory) / "queries_file.proto"
            queryBuilder.store([theQuery], proto_file_path)
            reified = read_queries_from_proto_with_reification(proto_file_path)
            # start asserting stuff
            allQueries = list(reified)
            assert len(allQueries) == 1
            query = allQueries[0]

            assert query.edge_index.size(0) == 2
            assert query.edge_index.size(1) == 12  # 3*3 for triple + 3 for qualifiers

            # all subjects must be variables
            subjects = query.edge_index[0, :]
            for s in subjects:
                assert get_entity_mapper().is_entity_reified_statement(s.item()), "All subject must be reified statement ids"
            for triple_nr in range(3):
                # Note: it is not strictly necessary that they appear in this order
                assert query.edge_type[triple_nr * 3] == get_relation_mapper().reifiedSubject
                assert query.edge_type[triple_nr * 3 + 1] == get_relation_mapper().reifiedPredicate
                assert query.edge_type[triple_nr * 3 + 2] == get_relation_mapper().reifiedObject
            # testing the reified triples
            assert query.edge_index[1, 0] == get_entity_mapper().lookup("https://www.wikidata.org/wiki/Q167520")
            assert query.edge_index[1, 1] == get_entity_mapper().get_entity_for_predicate(get_relation_mapper().lookup("https://www.wikidata.org/wiki/Property:P1411"))
            assert query.edge_index[1, 2] == get_entity_mapper().lookup("?var1")

            assert query.edge_index[1, 3] == get_entity_mapper().lookup("https://www.wikidata.org/wiki/Q167520")
            assert query.edge_index[1, 4] == get_entity_mapper().get_entity_for_predicate(get_relation_mapper().lookup("https://www.wikidata.org/wiki/Property:P1346"))
            assert query.edge_index[1, 5] == get_entity_mapper().lookup("?var1")

            assert query.edge_index[1, 6] == get_entity_mapper().lookup("?var1")
            assert query.edge_index[1, 7] == get_entity_mapper().get_entity_for_predicate(get_relation_mapper().lookup("https://www.wikidata.org/wiki/Property:P1411"))
            assert query.edge_index[1, 8] == get_entity_mapper().lookup("https://www.wikidata.org/wiki/Q505449")

            # test whether the qualifiers have been correctly attached
            assert query.edge_index[0, 9] == query.edge_index[0, 0]
            assert query.edge_index[0, 10] == query.edge_index[0, 0]
            assert query.edge_index[0, 11] == query.edge_index[0, 3]

            # test whether the qualifiers ahve the correct realtion types and values
            assert query.edge_type[9] == get_relation_mapper().lookup("https://www.wikidata.org/wiki/Property:P355")
            assert query.edge_index[1, 9] == get_entity_mapper().lookup("https://www.wikidata.org/wiki/Q1033465")
            assert query.edge_type[10] == get_relation_mapper().lookup("https://www.wikidata.org/wiki/Property:P190")
            assert query.edge_index[1, 10] == get_entity_mapper().lookup("https://www.wikidata.org/wiki/Q274670")
            assert query.edge_type[11] == get_relation_mapper().lookup("https://www.wikidata.org/wiki/Property:P161")
            assert query.edge_index[1, 11] == get_entity_mapper().lookup("https://www.wikidata.org/wiki/Q28389")

            assert len(query.targets) == 3
            assert query.targets[0] == get_entity_mapper().lookup("https://www.wikidata.org/wiki/Q1648927")
            assert query.targets[1] == get_entity_mapper().lookup("https://www.wikidata.org/wiki/Q10855195")
            assert query.targets[2] == get_entity_mapper().lookup("https://www.wikidata.org/wiki/Q901462")

            assert query.query_diameter == 2


if __name__ == '__main__':
    unittest.main()
