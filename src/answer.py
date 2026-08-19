import json
from pathlib import Path

from pydantic import BaseModel, ValidationError
from tqdm.std import tqdm

from src.classes.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.local_llm import LocalQwen
from src.search import Search


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

    class Cache(BaseModel):
        """Cached retrieval results."""

        questions_answer: list[dict[str, str]]
        k: int

        def get_res(self, query: str) -> str | None:
            for qa in self.questions_answer:
                for q in qa:
                    if q == query:
                        return qa[q]
            return None

    def get_cached_sources(self, query: str, k: int) -> str | None:
        """Return cached sources for a query, if available."""
        cached_file = Path("data/processed/answer_cache.json")

        try:
            with cached_file.open("r", encoding="utf-8") as file:
                cache = self.Cache.model_validate(json.load(file))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            print(f"Warning: could not load search cache: {error}")
            return None

        if cache.k != k:
            return None

        return cache.get_res(query)

    def save_cached_sources(self, query: str, answer: str, k: int) -> None:
        """Save retrieved sources for a query."""
        cached_file = Path("data/processed/answer_cache.json")
        temporary_file = cached_file.with_suffix(".tmp")

        cached_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with cached_file.open("r", encoding="utf-8") as file:
                cache = self.Cache.model_validate(json.load(file))
            if cache.k != k:
                raise ValueError("k not the same in cache")
        except FileNotFoundError:
            cache = self.Cache(questions_answer=[], k=k)
        except (
            OSError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as error:
            print(f"Warning: rebuilding invalid search cache: {error}")
            cache = self.Cache(questions_answer=[], k=k)

        cache.questions_answer.append({query: answer})

        try:
            with temporary_file.open("w", encoding="utf-8") as file:
                json.dump(
                    cache.model_dump(mode="json"),
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            temporary_file.replace(cached_file)
        except OSError as error:
            print(f"Warning: could not save search cache: {error}")
            temporary_file.unlink(missing_ok=True)

    def answer(
        self,
        query: str,
        k: int,
        top_k: list[MinimalSource] | None = None,
    ) -> str:
        """Answer a query using retrieved sources."""
        query = query.strip()

        answer_cache = self.get_cached_sources(query, k)

        if answer_cache:
            return answer_cache

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
            res = self.llm.invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            self.save_cached_sources(query, res, k)

            return res
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
