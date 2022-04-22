import unittest

import numpy as np
import pandas as pd

from mphrqe.oracle_metrics import (MacroReducer, MicroReducer, OracleMeanRank,
                                   OracleMeanReciprocalRank, OracleMetric,
                                   Reducer, _grouped_histogram,
                                   _optimal_answer_set_with_extended_input,
                                   _reduce_frame,
                                   create_oracle_hits_at_k_class)


class GroupedHistogramTests(unittest.TestCase):

    def test_reduce_frame(self):
        df = pd.DataFrame(
            {
                "s0": ["foo", "bar", "foo", "bar", "foo"],
                "p0": ["spa", "ham", "ham", "spa", "spa"],
                "o0_target": ["one|two", "two|three", "three|four", "four|five", "five|six"],
                "qr0_1": ["qual", "qual", "qual", "qual", "qual"],
                "qv0_1": ["qual", "qual", "qual", "qual", "qual"],
                "diameter": np.random.randint(5),
            }
        )
        reduced_frame, group_by_names = _reduce_frame(df)
        assert set(group_by_names) == {"s0", "p0"}
        columns = reduced_frame.columns.values
        assert set(columns) == {"s0", "p0", "TARGETS"}
        assert reduced_frame["TARGETS"][3] == {"four", "five"}

    def test_grouped_histogram(self):
        reduced_frame = pd.DataFrame(
            {
                "s0": ["foo", "bar", "foo", "bar", "foo"],
                "p0": ["spa", "ham", "spa", "spa", "spa"],
                "TARGETS": [{"one", "two", "three"}, {"two", "three"}, {"three", "four"}, {"four", "five"}, {"five", "six"}],
            }
        )
        group_by_names = ["s0", "p0"]
        grouped = reduced_frame.groupby(group_by_names, sort=False)
        histograms = _grouped_histogram(grouped)
        self.assertEqual(len(histograms), 3)  # 3 different queries
        self.assertEqual(histograms[0][0], (2, {"three"}))
        self.assertEqual(histograms[0][1], (1, {"one", "two", "four", "five", "six"}))
        self.assertEqual(len(histograms[0]), 2)
        self.assertEqual(histograms[1][0], (1, {"two", "three"}))


class OptimalAnswerSetTest(unittest.TestCase):
    def test_optimal_answer_set_with_extended_input(self):
        training_data = pd.DataFrame(
            {
                "s0": ["foo", "bar", "foo", "bar", "foo", "foo"],
                "p0": ["spa", "ham", "spa", "spa", "spa", "ham"],
                "o0_target": ["one|two|three", "two|three", "three|four", "four|five", "five|six", "seven"],
            }
        )
        test_data = pd.DataFrame(
            {
                "s0": ["foo", "bar", "foo", "bar", "foo"],
                "p0": ["spa", "ham", "spa", "spa", "spa"],
                "o0_target": ["vijf", "two|drie", "zeven", "four|five", "six"],
            }
        )
        training_data = training_data.append(test_data)
        _optimal_answer_set_with_extended_input(training_data, test_data, [OracleMeanRank], MicroReducer())


class ReducerTest(unittest.TestCase):

    # in order: (input, micro result, macro result)
    measures_and_expected_result = [
        ([[[1.0]]], [1.0], [1.0]),
        ([[[1.0, 2.0]]], [1.0, 2.0], [1.0, 2.0]),
        ([[[1.0], [2.0]]], [1.5], [1.5]),
        ([[[1.0], [2.0]], [[1.0], [2.0]]], [1.5], [1.5]),
        ([[[1.0, 2.0], [1.0, 2.0]], [[4.0, 5.0]]], [2.0, 3.0], [5.0 / 2.0, 7.0 / 2.0])

    ]

    @staticmethod
    def for_one_reducer(reducer: Reducer, result_index: int):
        for case in ReducerTest.measures_and_expected_result:
            (input, _, _) = case
            assert reducer.reduce(input) == case[result_index], f"Wrong answer for {input} "
            # all reducers must be independent regarding re-ordering
            if len(input) > 1:
                # swap first and last
                input[0], input[-1] = input[-1], input[0]
                assert reducer.reduce(input) == case[result_index], f"Wrong answer for {input} "

    def test_micro_reducer(self):
        ReducerTest.for_one_reducer(MicroReducer(), 1)

    def test_macro_reducer(self):
        ReducerTest.for_one_reducer(MacroReducer(), 2)


class OracleMetricTests(unittest.TestCase):
    @staticmethod
    def fill_metric_1(OracleMetricClass) -> float:
        m: OracleMetric = OracleMetricClass()
        m.seen_chunck_without_answer(3)
        m.seen_chunck_without_answer(5)
        m.chunck_with_answer(1)
        return m.metric_value()

    @staticmethod
    def fill_metric_2(OracleMetricClass) -> float:
        m: OracleMetric = OracleMetricClass()
        m.seen_chunck_without_answer(3)
        m.seen_chunck_without_answer(5)
        m.chunck_with_answer(5)
        return m.metric_value()

    @staticmethod
    def fill_metric_3(OracleMetricClass) -> float:
        m: OracleMetric = OracleMetricClass()
        m.seen_chunck_without_answer(3)
        m.seen_chunck_without_answer(5)
        m.chunck_with_answer(4)
        return m.metric_value()

    @staticmethod
    def fill_metric_4(OracleMetricClass) -> float:
        m: OracleMetric = OracleMetricClass()
        m.chunck_with_answer(1)
        return m.metric_value()

    @staticmethod
    def fill_metric_5(OracleMetricClass) -> float:
        m: OracleMetric = OracleMetricClass()
        m.chunck_with_answer(5)
        return m.metric_value()

    def test_oracle_mean_rank(self):
        assert OracleMetricTests.fill_metric_1(OracleMeanRank) == 9
        assert OracleMetricTests.fill_metric_2(OracleMeanRank) == 11
        assert OracleMetricTests.fill_metric_3(OracleMeanRank) == 10.5
        assert OracleMetricTests.fill_metric_4(OracleMeanRank) == 1
        assert OracleMetricTests.fill_metric_5(OracleMeanRank) == 3

    def test_oracle_mrr(self):
        assert OracleMetricTests.fill_metric_1(OracleMeanReciprocalRank) == 1 / 9
        assert OracleMetricTests.fill_metric_2(OracleMeanReciprocalRank) == (1 / 9 + 1 / 10 + 1 / 11 + 1 / 12 + 1 / 13) / 5
        assert OracleMetricTests.fill_metric_3(OracleMeanReciprocalRank) == (1 / 9 + 1 / 10 + 1 / 11 + 1 / 12) / 4
        assert OracleMetricTests.fill_metric_4(OracleMeanReciprocalRank) == 1
        assert OracleMetricTests.fill_metric_5(OracleMeanReciprocalRank) == (1 / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 5) / 5

    def test_oracle_hits_1(self):
        assert OracleMetricTests.fill_metric_1(create_oracle_hits_at_k_class(1)) == 0
        assert OracleMetricTests.fill_metric_2(create_oracle_hits_at_k_class(1)) == 0
        assert OracleMetricTests.fill_metric_3(create_oracle_hits_at_k_class(1)) == 0
        assert OracleMetricTests.fill_metric_4(create_oracle_hits_at_k_class(1)) == 1
        assert OracleMetricTests.fill_metric_5(create_oracle_hits_at_k_class(1)) == 1 / 5

    def test_oracle_hits_4(self):
        assert OracleMetricTests.fill_metric_1(create_oracle_hits_at_k_class(4)) == 0
        assert OracleMetricTests.fill_metric_2(create_oracle_hits_at_k_class(4)) == 0
        assert OracleMetricTests.fill_metric_3(create_oracle_hits_at_k_class(4)) == 0
        assert OracleMetricTests.fill_metric_4(create_oracle_hits_at_k_class(4)) == 1
        assert OracleMetricTests.fill_metric_5(create_oracle_hits_at_k_class(4)) == 4 / 5

    def test_oracle_hits_5(self):
        assert OracleMetricTests.fill_metric_1(create_oracle_hits_at_k_class(5)) == 0
        assert OracleMetricTests.fill_metric_2(create_oracle_hits_at_k_class(5)) == 0
        assert OracleMetricTests.fill_metric_3(create_oracle_hits_at_k_class(5)) == 0
        assert OracleMetricTests.fill_metric_4(create_oracle_hits_at_k_class(5)) == 1
        assert OracleMetricTests.fill_metric_5(create_oracle_hits_at_k_class(5)) == 1

    def test_oracle_hits_10(self):
        assert OracleMetricTests.fill_metric_1(create_oracle_hits_at_k_class(10)) == 1
        assert OracleMetricTests.fill_metric_2(create_oracle_hits_at_k_class(10)) == 2 / 5
        assert OracleMetricTests.fill_metric_3(create_oracle_hits_at_k_class(10)) == 2 / 4
        assert OracleMetricTests.fill_metric_4(create_oracle_hits_at_k_class(10)) == 1
        assert OracleMetricTests.fill_metric_5(create_oracle_hits_at_k_class(10)) == 1

# TODO rewrite for class metric objects

# class Test_assert_query_validity(unittest.TestCase):

#     values = [(5, 10), (5, 5), (10, 100)]
#     mean_ranks = [3.5, 1.0, 46]
#     ks = [1, 5, 10]
#     hits = [[0.16666666, 0.833333333333,  1.0000000000], [1.0, 1.0, 1.0], [0.010989011, 0.054945055, 0.10989011]]

#     def test_mean_rank(self):
#         for (gold, oracle), mean_rank in zip(Test_assert_query_validity.values, Test_assert_query_validity.mean_ranks):
#             assert expected_mean_rank(oracle_answers=oracle, gold_answers=gold) == mean_rank


#     def test_hits_at_k(self):
#         for (gold, oracle), hits in zip(Test_assert_query_validity.values, Test_assert_query_validity.hits):
#             for k, hits_k in zip(Test_assert_query_validity.ks, hits):
#                 self.assertAlmostEqual(expected_hits_at_k(oracle_answers=oracle, gold_answers=gold, k=k), hits_k)
