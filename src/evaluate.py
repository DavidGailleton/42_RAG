"""Evaluate retrieval quality using recall at different values of k."""

import json
from pathlib import Path

from .classes.models import AnsweredQuestion, StudentSearchResults


class Evaluate:
    """Calculate retrieval recall against a ground-truth dataset."""

    def __init__(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Initialize the retrieval evaluator.

        Args:
            student_search_results_path: Path to the student's search-results
                JSON file.
            dataset_path: Path to the answered ground-truth dataset.
        """
        self.student_search_results_path = Path(student_search_results_path)
        self.dataset_path = Path(dataset_path)

    def overlap_ok(
        self,
        correct_range: tuple[int, int],
        testing_range: tuple[int, int],
        threshold: float = 0.05,
    ) -> bool:
        """Check whether two character ranges have sufficient IoU.

        Args:
            correct_range: Ground-truth character range.
            testing_range: Retrieved character range.
            threshold: Minimum accepted intersection over union.

        Returns:
            ``True`` when the ranges have an IoU greater than or equal to the
            threshold; otherwise, ``False``.
        """
        correct_start, correct_end = sorted(correct_range)
        testing_start, testing_end = sorted(testing_range)

        intersection = max(
            0,
            min(correct_end, testing_end) - max(correct_start, testing_start),
        )

        union = max(correct_end, testing_end) - min(
            correct_start,
            testing_start,
        )

        if union <= 0:
            return False

        iou = intersection / union
        return iou >= threshold

    def evaluate(self, attended_recall: int) -> float:
        """Calculate recall at the requested retrieval depth.

        A ground-truth source is considered retrieved when the result has the
        same file path and its character range reaches the required IoU
        threshold.

        Args:
            attended_recall: Number of highest-ranked sources to evaluate.

        Returns:
            Mean recall as a percentage between 0 and 100. A value of zero is
            returned if evaluation cannot be completed.
        """
        try:
            with self.student_search_results_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                ssr = StudentSearchResults(**json.load(file))

            with self.dataset_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                ground_truth = [
                    AnsweredQuestion(**question)
                    for question in json.load(file)["rag_questions"]
                ]

            recalls: list[float] = []

            for question in ground_truth:
                correct_sources = question.sources
                search_result = ssr.get_msr_by_question(question.question)

                if search_result is None:
                    recalls.append(0.0)
                    continue

                retrieved_sources = search_result.retrieved_sources[
                    :attended_recall
                ]

                matches: list[int] = []

                for position, correct_source in enumerate(
                    correct_sources,
                    start=1,
                ):
                    for retrieved_source in retrieved_sources:
                        if (
                            retrieved_source.file_path
                            == correct_source.file_path
                            and self.overlap_ok(
                                (
                                    retrieved_source.first_character_index,
                                    retrieved_source.last_character_index,
                                ),
                                (
                                    correct_source.first_character_index,
                                    correct_source.last_character_index,
                                ),
                            )
                        ):
                            matches.append(1)
                            break

                    if len(matches) != position:
                        matches.append(0)

                if correct_sources:
                    recalls.append(sum(matches) / len(matches))
                else:
                    recalls.append(0.0)

            if not recalls:
                return 0.0

            return sum(recalls) / len(recalls) * 100

        except Exception as error:
            print(f"Error during evaluation: {error}")
            return 0.0
