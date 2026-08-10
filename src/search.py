import bm25s
import json
from pathlib import Path

from src.classes.models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)

from typing import Any


class Search:
    def __init__(self, query: str, k: int) -> None:
        self.query = query
        self.k = k

    def search(self) -> None:
        retriever = bm25s.BM25().load("data/processed", load_corpus=True)
        docs, scores = retriever.retrieve(bm25s.tokenize(self.query), k=self.k)

        self.save_student_search_results(docs, scores)

    def save_student_search_results(self, docs: Any, scores: Any) -> None:
        output_path = Path("data/output/search_results")

        try:
            with open(
                output_path.__str__() + f"/{self.k}.json", encoding="utf-8"
            ) as file:
                ssr = StudentSearchResults(**json.load(file)[0])
        except FileNotFoundError:
            ssr = StudentSearchResults(search_results=[], k=self.k)

        ssr.search_results.append(
            MinimalSearchResults(
                question_id=f"q{len(ssr.search_results)}",
                question=self.query,
                retrieved_sources=[MinimalSource(**doc) for doc in docs[0]],
            )
        )

        with open(
            output_path.__str__() + f"/{self.k}.json", "w", encoding="utf-8"
        ) as file:
            json.dump(ssr.dict(), file, indent=4, ensure_ascii=False)
