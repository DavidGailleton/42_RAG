"""Main entry point for the project."""

import fire


class RAG(object):
    """Will you answer my questions?"""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/"""
        from src.indexer import Indexer

        indexer = Indexer(max_chunk_size=max_chunk_size)

        indexer.index()

    def search(self, query: str, k: int) -> None:
        """Return the top-k sources for a single query"""
        from src.search import Search

        search = Search(query=query, k=k)
        search.search()

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> None:
        """Run search over a whole dataset and write a StudentSearchResults JSON file"""
        print("index")

    def answer(self, query: str, k: int) -> None:
        """Answer a single query using the retrieved context"""
        print("index")

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers for a dataset, producing a StudentSearchResultsAndAnswer JSON file"""
        print("index")

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Report your own recall@k against a ground-truth dataset, for your own testing"""
        print("index")


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
