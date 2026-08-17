from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def write_man():
    return [
        {
            "/index/{max_chunk_size}": "Ingest data/raw/ and build the index under data/processed/"
        },
        {"/search/{query}/{k}": "Return the top-k sources for a single query"},
        {
            "/search_dataset/{dataset_path}/{k}/{save_directory}": "Run search over a whole dataset and write a StudentSearchResults JSON file"
        },
        {
            "/answer/{query}/{k}": "Answer a single query using the retrieved context"
        },
        {
            "/answer_dataset/{student_search_results_path}/{save_directory}": "Generate answers for a dataset, producing a StudentSearchResultsAndAnswer JSON file"
        },
    ]


@app.get("/index/{max_chunk_size}")
def index(max_chunk_size: int = 2000):
    from src.indexer import Indexer

    try:
        Indexer(max_chunk_size=max_chunk_size)
        return {"index": "ok"}
    except Exception:
        return {"index": "ko"}


@app.get("/search/{query}/{k}")
def search(query: str, k: int):
    from src.search import Search

    try:
        search_cls = Search()
        res = search_cls.search(query, k)
        return {"query": query, "results": res}
    except Exception:
        return {"search": "ko"}


@app.get("/search_dataset/{dataset_path}/{k}/{save_directory}")
def search_dataset(dataset_path: str, k: int, save_directory: str):
    from src.search import SearchDataset

    try:
        search_cls = SearchDataset(
            dataset_path=dataset_path, k=k, save_directory=save_directory
        )
        search_cls.search_dataset()
        return {"search_dataset": "ok"}
    except Exception:
        return {"search_dataset": "ko"}


@app.get("/answer/{query}/{k}")
def answer(query: str, k: int):
    from src.answer import Answer

    try:
        answer_cls = Answer()
        res = answer_cls.answer(query, k)
        return {"query": query, "results": res}
    except Exception:
        return {"search": "ko"}


@app.get("/answer_dataset/{student_search_results_path}/{save_directory}")
def answer_dataset(student_search_results_path: str, save_directory: str):
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
