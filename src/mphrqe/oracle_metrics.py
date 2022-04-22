"""Calculation of the metrics for the oracle predictions"""
import collections
import copy
from abc import abstractmethod
from pathlib import Path
from typing import DefaultDict, Iterable, List, Tuple, Type

import pandas as pd


class OracleMetric:
    @abstractmethod
    def seen_chunck_without_answer(self, filtered_size: int):
        pass

    @abstractmethod
    def chunck_with_answer(self, filtered_size_plus_one: int):
        pass

    def metric_value(self) -> float:
        pass


class OracleMeanRank(OracleMetric):
    def __init__(self) -> None:
        self.current_rank = 1.0

    def seen_chunck_without_answer(self, filtered_size: int):
        assert filtered_size >= 0
        self.current_rank += filtered_size

    def chunck_with_answer(self, filtered_size_plus_one: int):
        assert filtered_size_plus_one > 0
        self.current_rank += ((filtered_size_plus_one - 1) / 2)
        self.expected_mean_rank = self.current_rank

    def metric_value(self) -> float:
        return self.expected_mean_rank


class OracleMeanReciprocalRank(OracleMeanRank):
    def __init__(self) -> None:
        self.current_rank: int = 1

    def seen_chunck_without_answer(self, filtered_size: int):
        assert filtered_size >= 0
        self.current_rank += filtered_size

    def chunck_with_answer(self, filtered_size_plus_one: int):
        assert filtered_size_plus_one > 0
        # the reciprocal rank is going to be between 1/(self.current_rank+1) and 1/(self.current_rank + 1 + filtered_size)
        sum_of_mrr = 0.0
        for i in range(self.current_rank, self.current_rank + filtered_size_plus_one):
            sum_of_mrr += 1.0 / i
        self.expected_mrr = sum_of_mrr / (filtered_size_plus_one)

    def metric_value(self) -> float:
        return self.expected_mrr


def create_oracle_hits_at_k_class(k: int) -> Type[OracleMetric]:
    protected_k = k

    class OracleHitsAtKImp(OracleMetric):
        def __init__(self) -> None:
            self.k = protected_k
            self.current_rank = 1.0

        def seen_chunck_without_answer(self, filtered_size: int):
            assert filtered_size >= 0
            self.current_rank += filtered_size

        def chunck_with_answer(self, filtered_size_plus_one: int):
            assert filtered_size_plus_one > 0
            if self.current_rank > self.k:
                self.expect_hits_at_k = 0.0
            elif self.current_rank + filtered_size_plus_one - 1 <= self.k:
                self.expect_hits_at_k = 1.0
            else:
                k_left = self.k - (self.current_rank - 1)
                assert k_left <= filtered_size_plus_one
                self.expect_hits_at_k = k_left / filtered_size_plus_one

        def metric_value(self) -> float:
            return self.expect_hits_at_k
    return OracleHitsAtKImp


class Reducer:
    def reduce(self, measures: List[List[List[float]]]) -> Iterable[float]:
        """
        measures is a list with an element for each query.
        The element for each query is a list with an element for each answer.
        The element for each answer is a list with an element for each metric.

        Return
        ------
            A list with floats, each is the reduced value for the metric (retaining the order used in the lowest level input lists)
        """
        pass


class MicroReducer(Reducer):
    def reduce(self, measures: List[List[List[float]]]) -> Iterable[float]:
        number_of_metrics = len(measures[0][0])
        total_sums = [0.0 for _ in range(number_of_metrics)]
        total_count = 0
        for metrics_for_answers_of_one_query in measures:
            for metrics_for_one_answer_of_one_query in metrics_for_answers_of_one_query:
                for i in range(number_of_metrics):
                    total_sums[i] += metrics_for_one_answer_of_one_query[i]
                total_count += 1

        reduced_metrics = [total_sum / total_count for total_sum in total_sums]
        return reduced_metrics


class MacroReducer(Reducer):
    def reduce(self, measures: List[List[List[float]]]) -> Iterable[float]:
        number_of_metrics = len(measures[0][0])
        average_metrics_per_query: List[List[float]] = [[] for _ in range(number_of_metrics)]
        for metrics_for_answers_of_one_query in measures:
            for i in range(number_of_metrics):
                metric_sum_for_query = 0.0
                for metrics_for_one_answer_of_one_query in metrics_for_answers_of_one_query:
                    metric_sum_for_query += metrics_for_one_answer_of_one_query[i]
                query_answer_count = len(metrics_for_answers_of_one_query)
                average_metrics_per_query[i].append(metric_sum_for_query / query_answer_count)
        return [sum(average_metric_per_query) / len(average_metric_per_query) for average_metric_per_query in average_metrics_per_query]


def optimal_answer_set(dataset: Path, metric_classes: Iterable[Type[OracleMetric]], reducer: Reducer) -> Iterable[float]:
    data = pd.read_csv(dataset, dtype="string")
    return _optimal_answer_set(data, metric_classes, reducer)


def _optimal_answer_set(data: pd.DataFrame, metric_classes: Iterable[Type[OracleMetric]], reducer: Reducer) -> Iterable[float]:
    reduced_frame, group_by_names = _reduce_frame(data)
    # we now have all needed columns in the reduced frame. This needs to be grouped by everything except "TARGETS"
    grouped = reduced_frame.groupby(group_by_names, sort=False)
    histograms = _grouped_histogram(grouped)
    measures = []
    for ((query_answers, group), histogram) in zip(grouped, histograms):
        targets_for_all = group["TARGETS"]
        # TODO check whether this takes number of queries or number of unique stripped queries??
        for targets_for_one in targets_for_all:
            # Use the answer from the histogram as an answer for this query
            results = _expected_metrics(histogram, targets_for_one, metric_classes=metric_classes)
            measures.append(results)
    final_result = reducer.reduce(measures)
    return final_result


def optimal_answer_set_with_extended_input(training_data: Iterable[Path], test_data: Path, metric_classes: Iterable[Type[OracleMetric]], reducer: Reducer) -> Iterable[float]:
    combined_training_data = pd.concat([pd.read_csv(dataset) for dataset in training_data])
    test_data_frame = pd.read_csv(test_data)
    return _optimal_answer_set_with_extended_input(combined_training_data, test_data_frame, metric_classes, reducer)


def _optimal_answer_set_with_extended_input(training_data: pd.DataFrame, test_data: pd.DataFrame, metric_classes: Iterable[Type[OracleMetric]], reducer: Reducer) -> Iterable[float]:
    reduced_train_frame, group_by_names = _reduce_frame(training_data)
    reduced_test_frame, group_by_names_2 = _reduce_frame(test_data)
    assert group_by_names == group_by_names_2
    del group_by_names_2

    # we now have all needed columns in the reduced frame. This needs to be grouped by everything except "TARGETS"
    grouped_train = reduced_train_frame.groupby(group_by_names, sort=False)
    histograms = _grouped_histogram(grouped_train)
    # TODO it is maybe possible to somehow join the grouped frame with the reduced_test_frame, but no clue how.
    query_to_answers = {}
    for ((query, group), histogram) in zip(grouped_train, histograms):
        query_to_answers[query] = histogram

    grouped_test = reduced_test_frame.groupby(group_by_names, sort=False)
    measures = []
    for (query, group) in grouped_test:
        histogram = query_to_answers[query]
        targets_for_all = group["TARGETS"]
        # TODO check whether this takes number of queries or number of unique stripped queries??
        for targets_for_one in targets_for_all:
            # Use the answer from the histogram as an answer for this query
            results = _expected_metrics(histogram, targets_for_one, metric_classes=metric_classes)
            measures.append(results)
    final_result = reducer.reduce(measures)
    return final_result


def _reduce_frame(query_frame: pd.DataFrame) -> Tuple[pd.DataFrame, Iterable[str]]:
    reduced_frame = pd.DataFrame()
    group_by_names: List[str] = []
    for part in query_frame.columns.values:
        sub_parts = part.split("_")
        if sub_parts[-1] == "target":
            reduced_frame["TARGETS"] = query_frame[part].apply(lambda x: set(x.split("|")))
            continue
        elif sub_parts[-1] == "var":
            # ignore, variables are not relevant for the grouping
            continue
        elif sub_parts == "diameter":
            # ignore, variables are not relevant for the grouping
            continue
        else:
            for subpart in sub_parts:
                subpart = subpart.strip()
                # we treat each different: subject, predicate, object, qr, qv
                if subpart.startswith("s") or subpart.startswith("p") or subpart.startswith("o"):
                    group_by_names.append(part)
                    reduced_frame[part] = query_frame[part]
                    break
    return reduced_frame, group_by_names


def _grouped_histogram(dataframe_groups):
    """Takes each group of the dataframe, and withing that group, it creates a histogram. """
    histograms = []
    for query, group in dataframe_groups:
        histogram_collector: DefaultDict[str, int] = collections.defaultdict(int)
        targets_for_all = group["TARGETS"]
        for targets_for_one_query in targets_for_all:
            for target in targets_for_one_query:
                histogram_collector[target] += 1
        # print(histogram_collector)
        # the ideal answer is according to descending frequency. But, within the same bin the order is not fixed. So we need to use an expected rank within.
        histogram_sorted_not_grouped = sorted([(v, k) for (k, v) in histogram_collector.items()], key=lambda v_k: v_k[0], reverse=True)
        grouped_histogram = []
        last_count = float('inf')
        for (count, entity) in histogram_sorted_not_grouped:
            if count < last_count:
                last_count = count
                entities_with_count = set()
                grouped_histogram.append((count, entities_with_count))
            entities_with_count.add(entity)
        # print (grouped_histogram)
        histograms.append(grouped_histogram)
    return histograms


def _expected_metrics(oracle_answer, correct_answer_set, metric_classes: Iterable[Type[OracleMetric]]) -> List[List[float]]:
    """
    This computes the metrics specified in the metric_classes for the given correct_answer_set and the oracle_answer.

    For each entry in the correct answer set, we have to see at which filtered rank it occurs in the oracle answer set.
    Since the oracle answer has bins with answers, we have to compute the expected metric assuming a uniform probability for each rank in the bin.

    Return:
        An Iterable with the resulting measures, one for each answer in the correct_answer_set.
        Each element is itself an Iterable, with one float for each metric in metric_classes.
    """
    results = []
    for correct_answer in correct_answer_set:
        metrics = [m() for m in metric_classes]
        found = False
        for (_, answers) in oracle_answer:
            bin_found = correct_answer in answers
            wrong_answers = answers - correct_answer_set
            number_of_wrong_answers = len(wrong_answers)
            if bin_found:
                # the assumption is that the answer is in the middle of the wrong answers
                for metric in metrics:
                    metric.chunck_with_answer(number_of_wrong_answers + 1)
                # we are done for this one
                found = True
                break
            else:
                # it is not in this chunk, it must be in one of the next chunks
                for metric in metrics:
                    metric.seen_chunck_without_answer(number_of_wrong_answers)
        assert found, "The answer was not in the provided oracle_answer, which is impossible since the oracle_answer is created from the correct_answer_set's"
        results_for_answer = [m.metric_value() for m in metrics]
        results.append(results_for_answer)
    return results


def _expected_ranks(oracle_answer, correct_answer_set):
    # for each entry in the correct answer set, we have to see where it occurs in the oracle answer set. At the same time, we have to filter correct answers.
    # then, we have to make sure to compute the expected rank of the answers
    oracle_answer_original = copy.deepcopy(oracle_answer)
    filtered_ranks = []
    for correct_answer in correct_answer_set:
        # modifiable copy
        oracle_answer = copy.deepcopy(oracle_answer_original)
        rank = 0
        found = False
        for (_, answers) in oracle_answer:
            bin_found = correct_answer in answers
            wrong_answers = answers - correct_answer_set
            if bin_found:
                # the assumption is that the answer is in the middle of the wrong answers
                number_of_wrong_answers = len(wrong_answers)
                rank_in_chunck = (number_of_wrong_answers / 2) + 1
                rank += rank_in_chunck
                # we are done for this one
                found = True
                break
            else:
                # it is not in this chunk, it must be in one of the next chunks
                number_of_wrong_answers = len(wrong_answers)
                rank += number_of_wrong_answers
        assert found
        filtered_ranks.append(rank)
    return filtered_ranks
