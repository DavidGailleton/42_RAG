"""Main entry point for the project."""

import fire


class RAG(object):
    """Expose the RAG pipeline commands through Python Fire."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Build indexes from the raw source corpus.

        Args:
            max_chunk_size: Maximum number of characters allowed in a chunk.

        Returns:
            None.
        """
        from src.indexer import Indexer

        indexer = Indexer(max_chunk_size=max_chunk_size)

        indexer.index()

    def search(
        self,
        query: str,
        k: int = 5,
        retrieval_mode: str = "bm25",
        semantic_weight: float = 0.05,
        candidate_multiplier: int = 10,
    ) -> None:
        """Print the top-k sources for a single query.

        Args:
            query: Natural-language or code-oriented search query.
            k: Maximum number of sources to retrieve.
            retrieval_mode: Retrieval method to use. Supported values are
                ``bm25``, ``semantic``, and ``hybrid``.
            semantic_weight: Semantic contribution used in hybrid mode.
            candidate_multiplier: Multiplier used to size the candidate pool.

        Returns:
            None.
        """
        from src.search import Search

        search_engine = Search(
            retrieval_mode=retrieval_mode,
            semantic_weight=semantic_weight,
            candidate_multiplier=candidate_multiplier,
        )

        results = search_engine.search(query, k)

        for source in results:
            print(source.model_dump_json())

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        retrieval_mode: str = "bm25",
        semantic_weight: float = 0.05,
        candidate_multiplier: int = 10,
    ) -> None:
        """Retrieve sources for every question in a dataset.

        Args:
            dataset_path: Path to a dataset JSON file or directory.
            k: Maximum number of sources to retrieve per question.
            save_directory: Directory in which results are written.
            retrieval_mode: Retrieval method to use.
            semantic_weight: Semantic contribution used in hybrid mode.
            candidate_multiplier: Multiplier used to size the candidate pool.

        Returns:
            None.
        """
        from src.search import SearchDataset

        dataset_search = SearchDataset(
            dataset_path=dataset_path,
            k=k,
            save_directory=save_directory,
            retrieval_mode=retrieval_mode,
            semantic_weight=semantic_weight,
            candidate_multiplier=candidate_multiplier,
        )
        dataset_search.search_dataset()

    def answer(self, query: str, k: int) -> None:
        """Generate and print an answer for a single query.

        Args:
            query: Natural-language question to answer.
            k: Number of retrieved sources to provide to the model.

        Returns:
            None.
        """
        from src.answer import Answer

        answer = Answer()

        print(answer.answer(query=query, k=k))

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers for a retrieval-results dataset.

        Args:
            student_search_results_path: Path to a search-results JSON file or
                directory.
            save_directory: Directory in which generated answers are saved.

        Returns:
            None.
        """
        from src.answer import AnswerDataset

        answer_dataset = AnswerDataset(
            student_search_results_path, save_directory
        )

        answer_dataset.answer_dataset()

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Print recall scores against a ground-truth dataset.

        Args:
            student_search_results_path: Path to student retrieval results.
            dataset_path: Path to the answered ground-truth dataset.

        Returns:
            None.
        """
        from src.evaluate import Evaluate

        eva = Evaluate(student_search_results_path, dataset_path)
        for r_to_test in [1, 3, 5, 10]:
            res = eva.evaluate(r_to_test)
            print(f"Recall {r_to_test}: {res:.2f}")


def main() -> int:
    """Run the Python Fire command-line interface.

    Returns:
        Zero when execution succeeds and one when execution is interrupted or
        an error occurs.
    """
    try:
        fire.Fire(RAG)
        return 0
    except KeyboardInterrupt:
        return 1
    except Exception as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
