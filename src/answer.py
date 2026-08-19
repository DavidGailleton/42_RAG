"""Generate grounded answers from retrieved codebase sources."""

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
    """Generate grounded answers from retrieved codebase chunks.

    Attributes:
        llm: Local language model used for answer generation.
        search_engine: Optional retrieval engine used to locate sources.
    """

    def __init__(
        self,
        llm: LocalQwen | None = None,
        search_engine: Search | None = None,
    ) -> None:
        """Initialize the answer generator.

        Args:
            llm: Language model to use. A local Qwen model is loaded when this
                argument is ``None``.
            search_engine: Retrieval engine to use. A BM25 search engine is
                created lazily when this argument is ``None``.
        """
        self.llm = llm or LocalQwen()
        self.search_engine = search_engine

    class Cache(BaseModel):
        """Represent cached answers for a specific retrieval depth.

        Attributes:
            questions_answer: Query-to-answer mappings.
            k: Number of retrieved sources used to generate each answer.
        """

        questions_answer: list[dict[str, str]]
        k: int

        def get_res(self, query: str) -> str | None:
            """Find a cached answer for an exact query.

            Args:
                query: Query whose cached answer should be returned.

            Returns:
                The cached answer, or ``None`` when the query is not present.
            """
            for qa in self.questions_answer:
                for q in qa:
                    if q == query:
                        return qa[q]
            return None

    def get_cached_sources(self, query: str, k: int) -> str | None:
        """Load a cached answer for a query.

        Args:
            query: Query whose answer should be loaded.
            k: Number of sources expected to have been used for the answer.

        Returns:
            The cached answer when both the query and ``k`` match, or ``None``
            when no suitable cached answer exists.
        """
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
        """Store a generated answer in the answer cache.

        Args:
            query: Query associated with the generated answer.
            answer: Generated answer to cache.
            k: Number of retrieved sources used for generation.

        Returns:
            None.
        """
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
        """Generate a grounded answer for a query.

        Sources are either supplied by the caller or retrieved using the
        configured search engine. Their contents are added to the model
        prompt as supporting context.

        Args:
            query: Natural-language question to answer.
            k: Number of retrieved sources to use.
            top_k: Optional precomputed source locations. When omitted, the
                search engine retrieves the sources.

        Returns:
            A generated answer or a user-readable error message when the
            request cannot be completed.
        """
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
    """Generate answers for a dataset of existing retrieval results.

    Attributes:
        llm: Local language model used for generation.
        student_search_results_path: Input JSON file or directory.
        save_directory: Directory in which generated answers are stored.
    """

    def __init__(
        self,
        student_search_results_path: str,
        save_directory: str,
    ) -> None:
        """Initialize dataset answer generation.

        Args:
            student_search_results_path: Path to a search-results JSON file or
                a directory containing search-results JSON files.
            save_directory: Directory where generated answer files are saved.
        """
        self.llm = LocalQwen()
        self.student_search_results_path = student_search_results_path
        self.save_directory = save_directory

    def answer_dataset(self) -> None:
        """Generate and save answers for all available search-result files.

        Each input file is validated as ``StudentSearchResults``. Its
        retrieved sources are passed to the language model, and the resulting
        answers are saved as ``StudentSearchResultsAndAnswer`` JSON files.
        """
        answer_cls = Answer(self.llm)

        for file in Path(self.student_search_results_path).rglob("*.json"):
            with open(file, encoding="utf-8") as input_file:
                file_content = json.load(input_file)

            ssr = StudentSearchResults(**file_content)

            ssraa = StudentSearchResultsAndAnswer(
                search_results=[],
                k=ssr.k,
            )

            for search in tqdm(
                ssr.search_results,
                desc=f"Answering {file.name}",
            ):
                ssraa.search_results.append(
                    MinimalAnswer(
                        question=search.question,
                        question_id=search.question_id,
                        retrieved_sources=search.retrieved_sources,
                        answer=answer_cls.answer(
                            search.question,
                            ssr.k,
                            search.retrieved_sources,
                        ),
                    )
                )

            output_directory = Path(self.save_directory)
            output_directory.mkdir(parents=True, exist_ok=True)
            output_path = output_directory / file.name

            with output_path.open(
                "w",
                encoding="utf-8",
            ) as save_file:
                json.dump(
                    ssraa.model_dump(),
                    save_file,
                    indent=4,
                    ensure_ascii=False,
                )
