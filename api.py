"""Expose the RAG pipeline through a local FastAPI HTTP API.

This module provides endpoints for indexing a codebase, retrieving relevant
sources, and generating grounded answers for individual queries or datasets.
"""

from fastapi import FastAPI

from .src.classes.models import MinimalSource

app = FastAPI(
    title="RAG Against the Machine API",
    description="Local HTTP API for indexing, \
    retrieval, and answer generation.",
    version="1.0.0",
)


@app.get("/")
def write_man() -> list[dict[str, str]]:
    """Return a short description of the available API endpoints.

    Returns:
        A list containing each endpoint and a description of its purpose.
    """
    return [
        {
            "/index/{max_chunk_size}": (
                "Ingest data/raw/ and build the index under data/processed/."
            )
        },
        {
            "/search/{query}/{k}": (
                "Return the top-k sources for a single query."
            )
        },
        {
            "/search_dataset/{dataset_path}/{k}/{save_directory}": (
                "Run search over a dataset and write a "
                "StudentSearchResults JSON file."
            )
        },
        {
            "/answer/{query}/{k}": (
                "Answer a single query using the retrieved context."
            )
        },
        {
            (
                "/answer_dataset/{student_search_results_path}/"
                "{save_directory}"
            ): (
                "Generate answers for a dataset and produce a "
                "StudentSearchResultsAndAnswer JSON file."
            )
        },
    ]


@app.get("/index/{max_chunk_size}")
def index(max_chunk_size: int = 2000) -> dict[str, str]:
    """Ingest the raw codebase and create the retrieval index.

    The generated index is persisted under ``data/processed/``.

    Args:
        max_chunk_size: Maximum number of characters allowed in one chunk.
            The project specification limits this value to 2000 characters.

    Returns:
        A dictionary containing ``"ok"`` when indexing succeeds or ``"ko"``
        when an error occurs.
    """
    from src.indexer import Indexer

    try:
        Indexer(max_chunk_size=max_chunk_size)
        return {"index": "ok"}
    except Exception:
        return {"index": "ko"}


@app.get("/search/{query}/{k}")
def search(query: str, k: int) -> dict[str, str | list[MinimalSource]]:
    """Retrieve the top-k source chunks for a single query.

    Args:
        query: Natural-language question or search query.
        k: Maximum number of source chunks to retrieve.

    Returns:
        A dictionary containing the original query and its retrieved sources.
        If retrieval fails, a dictionary containing an error status is
        returned.
    """
    from src.search import Search

    try:
        search_cls = Search()
        res = search_cls.search(query, k)
        return {"query": query, "results": res}
    except Exception:
        return {"search": "ko"}


@app.get("/search_dataset/{dataset_path}/{k}/{save_directory}")
def search_dataset(
    dataset_path: str,
    k: int,
    save_directory: str,
) -> dict[str, str]:
    """Retrieve sources for every question in a dataset.

    The search results are validated and written as a
    ``StudentSearchResults`` JSON file.

    Args:
        dataset_path: Path to the JSON dataset containing the questions.
        k: Maximum number of source chunks to retrieve for each question.
        save_directory: Directory in which to save the generated results.

    Returns:
        A dictionary containing ``"ok"`` when dataset retrieval succeeds or
        ``"ko"`` when an error occurs.
    """
    from src.search import SearchDataset

    try:
        search_cls = SearchDataset(
            dataset_path=dataset_path,
            k=k,
            save_directory=save_directory,
        )
        search_cls.search_dataset()
        return {"search_dataset": "ok"}
    except Exception:
        return {"search_dataset": "ko"}


@app.get("/answer/{query}/{k}")
def answer(query: str, k: int) -> dict[str, str]:
    """Generate a grounded answer for a single query.

    The answer generator retrieves the top-k relevant chunks and uses them as
    context for the language model.

    Args:
        query: Natural-language question to answer.
        k: Maximum number of source chunks to use as context.

    Returns:
        A dictionary containing the query and generated answer. If generation
        fails, a dictionary containing an error status is returned.
    """
    from src.answer import Answer

    try:
        answer_cls = Answer()
        res = answer_cls.answer(query, k)
        return {"query": query, "results": res}
    except Exception:
        return {"search": "ko"}


@app.get("/answer_dataset/{student_search_results_path}/{save_directory}")
def answer_dataset(
    student_search_results_path: str,
    save_directory: str,
) -> dict[str, str]:
    """Generate answers from previously retrieved dataset search results.

    The generated output is written as a
    ``StudentSearchResultsAndAnswer`` JSON file.

    Args:
        student_search_results_path: Path to the JSON file containing the
            previously retrieved sources.
        save_directory: Directory in which to save the generated answers.

    Returns:
        A dictionary containing ``"ok"`` when answer generation succeeds or
        ``"ko"`` when an error occurs.
    """
    from src.answer import AnswerDataset

    try:
        answer_cls = AnswerDataset(
            student_search_results_path=student_search_results_path,
            save_directory=save_directory,
        )
        answer_cls.answer_dataset()
        return {"answer_dataset": "ok"}
    except Exception:
        return {"answer_dataset": "ko"}
