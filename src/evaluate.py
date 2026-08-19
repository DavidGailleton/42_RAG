from pathlib import Path
import json

from .classes.models import AnsweredQuestion, RagDataset, StudentSearchResults


class Evaluate:
    def __init__(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
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
            True when the ranges have an IoU greater than or equal to
            the threshold.
        """
        correct_start, correct_end = sorted(correct_range)
        testing_start, testing_end = sorted(testing_range)

        intersection = max(
            0,
            min(correct_end, testing_end) - max(correct_start, testing_start),
        )

        union = max(correct_end, testing_end) - min(
            correct_start, testing_start
        )

        if union <= 0:
            return False

        iou = intersection / union
        return iou >= threshold

    def evaluate(self, attended_recall: int) -> float:
        try:
            with open(self.student_search_results_path) as file:
                ssr = StudentSearchResults(**json.load(file))
            with open(self.dataset_path) as file:
                sd = [
                    AnsweredQuestion(**rq)
                    for rq in json.load(file)["rag_questions"]
                ]

            recalls: list[float] = []

            for q in sd:
                sd_rs = q.sources

                ssr_msr = ssr.get_msr_by_question(q.question)
                if ssr_msr is None:
                    recalls.append(0)
                    continue

                ssr_rs = ssr_msr.retrieved_sources[:attended_recall]

                temp_calc: list[int] = []

                for i, source in enumerate(sd_rs, start=1):
                    for s in ssr_rs:
                        if (
                            s.file_path == source.file_path
                            and self.overlap_ok(
                                (
                                    s.first_character_index,
                                    s.last_character_index,
                                ),
                                (
                                    source.first_character_index,
                                    source.last_character_index,
                                ),
                            )
                        ):
                            temp_calc.append(1)
                            break
                    if len(temp_calc) != i:
                        temp_calc.append(0)

                recalls.append(sum(temp_calc) / len(temp_calc))

            return sum(recalls) / len(recalls) * 100

        except Exception as e:
            print(f"error during evaluate: {e}")
            return 0
