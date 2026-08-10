from re import search

import bm25s
import json
from pathlib import Path

from numpy import save

from src.classes.models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    RagDataset,
    AnsweredQuestion,
    UnansweredQuestion,
)

from typing import Any


class Search:
    @staticmethod
    def search(query: str, k: int) -> list[MinimalSource]:
        retriever = bm25s.BM25().load("data/processed", load_corpus=True)
        docs, _ = retriever.retrieve(bm25s.tokenize(query), k=k)

        return [MinimalSource(**doc) for doc in docs[0]]


class SearchDataset:
    def __init__(self, dataset_path: str, k: int, save_directory: str) -> None:
        self.dataset_path = Path(dataset_path)
        self.k = k
        self.save_directory = save_directory

    def search_dataset(self) -> None:
        for file in self.dataset_path.rglob("*.json"):
            dataset = self.load_dataset(file)

            ssr = StudentSearchResults(search_results=[], k=self.k)

            for question in dataset:
                ssr.search_results.append(
                    MinimalSearchResults(
                        question_id=question.question_id,
                        question=question.question,
                        retrieved_sources=Search.search(
                            question.question, self.k
                        ),
                    )
                )

            with open(
                f"{self.save_directory}/{file.name}", "w", encoding="utf-8"
            ) as save_file:
                json.dump(
                    json.loads(ssr.model_dump_json()),
                    save_file,
                    indent=4,
                    ensure_ascii=False,
                )

    def load_dataset(self, path: Path) -> list[UnansweredQuestion]:
        dataset: list[UnansweredQuestion] = []

        with open(path, "r", encoding="utf-8") as doc:
            questions = json.load(doc)["rag_questions"]
        for question in questions:
            dataset.append(UnansweredQuestion(**question))

        return dataset
