from tqdm.std import tqdm

from src.classes.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.local_llm import LocalQwen
from src.search import Search

from pathlib import Path
import json


class Answer:
    """Generate answers from retrieved codebase chunks."""

    def __init__(
        self,
        llm: LocalQwen | None = None,
        search_engine: Search | None = None,
    ) -> None:
        """Initialize the answer generator."""
        self.llm = llm or LocalQwen()
        self.search_engine = search_engine

    def answer(
        self,
        query: str,
        k: int,
        top_k: list[MinimalSource] | None = None,
    ) -> str:
        """Answer a query using retrieved sources."""
        query = query.strip()

        if not query:
            return "Unable to answer: the query is empty."

        if k <= 0:
            return "Unable to answer: k must be greater than zero."

        if top_k is None:
            if self.search_engine is None:
                self.search_engine = Search(retrieval_mode="bm25")

            top_k = self.search_engine.search(query, k)

        if not top_k:
            return "No relevant sources were found."

        context_parts: list[str] = []

        for number, source in enumerate(top_k, start=1):
            context_parts.append(
                f"[Source {number}]\n"
                f"File: {source.file_path}\n"
                f"Character range: "
                f"{source.first_character_index}-"
                f"{source.last_character_index}\n"
                f"Content:\n{source.get_text()}\n"
            )

        context = "\n\n".join(context_parts)

        system_prompt = (
            "You answer questions about a codebase. "
            "Use only the retrieved sources. "
            "Do not invent information. "
            'If the sources are insufficient, say "i don\'t know".'
        )

        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Retrieved sources:\n{context}\n\n"
            "Provide a concise, grounded answer."
        )

        try:
            return self.llm.invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except ValueError:
            return "Invalid source found"
        except OSError as error:
            return f"Unable to load the language model: {error}"
        except RuntimeError as error:
            return f"Model generation failed: {error}"


class AnswerDataset:
    def __init__(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        self.llm = LocalQwen()
        self.student_search_results_path = student_search_results_path
        self.save_directory = save_directory

    def answer_dataset(self) -> None:
        answer_cls = Answer(self.llm)

        for file in Path(self.student_search_results_path).rglob("*.json"):
            with open(file) as f:
                file_content = json.load(f)
            ssr = StudentSearchResults(**file_content)

            ssraa = StudentSearchResultsAndAnswer(search_results=[], k=ssr.k)

            for search in tqdm(ssr.search_results):
                ssraa.search_results.append(
                    MinimalAnswer(
                        question=search.question,
                        question_id=search.question_id,
                        retrieved_sources=search.retrieved_sources,
                        answer=answer_cls.answer(
                            search.question, ssr.k, search.retrieved_sources
                        ),
                    )
                )

            with open(
                self.save_directory + "/" + file.name,
                "w",
                encoding="utf-8",
            ) as save_file:
                json.dump(
                    ssraa.model_dump(),
                    save_file,
                    indent=4,
                    ensure_ascii=False,
                )
