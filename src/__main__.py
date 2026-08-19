"""Main entry point for the project."""

import fire


class RAG(object):
    """Will you answer my questions?"""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/"""
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
        """Return the top-k sources for a single query."""
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
        """Run search over a question dataset."""
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
        """Answer a single query using the retrieved context"""
        from src.answer import Answer

        answer = Answer()

        print(answer.answer(query=query, k=k))

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers for a dataset, producing a StudentSearchResultsAndAnswer JSON file"""
        from src.answer import AnswerDataset

        answer_dataset = AnswerDataset(
            student_search_results_path, save_directory
        )

        answer_dataset.answer_dataset()

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Report your own recall@k against a ground-truth dataset, for your own testing"""
        from src.evaluate import Evaluate

        eva = Evaluate(student_search_results_path, dataset_path)
        for r_to_test in [1, 3, 5, 10]:
            res = eva.evaluate(r_to_test)
            print(f"Recall {r_to_test}: {res:.2f}")


def main() -> int:
    """Run the main program.

    Returns:
        Exit status code.
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
