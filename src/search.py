"""Lexical, semantic, and hybrid retrieval operations."""

import json
from pathlib import Path
from typing import Any

import bm25s
import numpy as np
import Stemmer
from bm25s.tokenization import Tokenized
from numpy.typing import NDArray
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.classes.models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    UnansweredQuestion,
)


class Search:
    """Search an indexed codebase."""

    VALID_RETRIEVAL_MODES = {"bm25", "semantic", "hybrid"}

    def __init__(
        self,
        index_path: str = "data/processed",
        retrieval_mode: str = "bm25",
        semantic_weight: float = 0.05,
        candidate_multiplier: int = 10,
    ) -> None:
        """Initialize the search engine.

        Args:
            index_path: Directory containing the persisted indexes.
            retrieval_mode: Retrieval method. Accepted values are
                ``bm25``, ``semantic``, and ``hybrid``.
            semantic_weight: Semantic weight used for hybrid retrieval.
            candidate_multiplier: Number of candidates retrieved before
                selecting the final top-k results.

        Raises:
            ValueError: If an argument is invalid.
            FileNotFoundError: If an index file is missing.
        """
        mode = retrieval_mode.strip().lower()

        if mode not in self.VALID_RETRIEVAL_MODES:
            valid_modes = ", ".join(sorted(self.VALID_RETRIEVAL_MODES))
            raise ValueError(
                f"Invalid retrieval mode '{retrieval_mode}'. "
                f"Expected one of: {valid_modes}"
            )

        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be between 0.0 and 1.0")

        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")

        self.index_path = Path(index_path)
        self.retrieval_mode = mode
        self.semantic_weight = semantic_weight
        self.lexical_weight = 1.0 - semantic_weight
        self.candidate_multiplier = candidate_multiplier

        if not self.index_path.is_dir():
            raise FileNotFoundError(
                f"Index directory does not exist: {self.index_path}"
            )

        self.metadata = self._load_metadata()
        self.retriever, self.tokenizer = self._load_bm25()

        self.embeddings: NDArray[np.float32] | None = None
        self.embedding_model: SentenceTransformer | None = None

        if self.retrieval_mode in {"semantic", "hybrid"}:
            self._load_semantic_index()

    class Cache(BaseModel):
        """Cached retrieval results."""

        questions: dict[str, list[MinimalSource]]

    def get_cached_sources(
        self,
        query: str,
    ) -> list[MinimalSource] | None:
        """Return cached sources for a query, if available."""
        cached_file = Path("data/processed/search_cache.json")

        try:
            with cached_file.open("r", encoding="utf-8") as file:
                cache = self.Cache.model_validate(json.load(file))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            print(f"Warning: could not load search cache: {error}")
            return None

        return cache.questions.get(query)

    def save_cached_sources(
        self,
        query: str,
        sources: list[MinimalSource],
    ) -> None:
        """Save retrieved sources for a query."""
        cached_file = Path("data/processed/search_cache.json")
        temporary_file = cached_file.with_suffix(".tmp")

        cached_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with cached_file.open("r", encoding="utf-8") as file:
                cache = self.Cache.model_validate(json.load(file))
        except FileNotFoundError:
            cache = self.Cache(questions={})
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            print(f"Warning: rebuilding invalid search cache: {error}")
            cache = self.Cache(questions={})

        cache.questions[query] = sources

        try:
            with temporary_file.open("w", encoding="utf-8") as file:
                json.dump(
                    cache.model_dump(mode="json"),
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            # Atomic replacement prevents a partially written cache.
            temporary_file.replace(cached_file)
        except OSError as error:
            print(f"Warning: could not save search cache: {error}")
            temporary_file.unlink(missing_ok=True)

    def _load_metadata(self) -> list[dict[str, Any]]:
        """Load and validate chunk metadata.

        Returns:
            Chunk metadata ordered by chunk ID.

        Raises:
            FileNotFoundError: If chunks.json is missing.
            ValueError: If its contents are invalid.
        """
        metadata_path = self.index_path / "chunks.json"

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Metadata file does not exist: {metadata_path}"
            )

        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                content: object = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Malformed metadata JSON: {metadata_path}"
            ) from error
        except OSError as error:
            raise OSError(
                f"Unable to read metadata: {metadata_path}"
            ) from error

        if not isinstance(content, list):
            raise ValueError("chunks.json must contain a JSON list")

        metadata: list[dict[str, Any]] = []

        for position, item in enumerate(content):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Metadata entry {position} must be an object"
                )

            required_fields = {
                "file_path",
                "first_character_index",
                "last_character_index",
                "chunk_id",
            }

            missing_fields = required_fields.difference(item)

            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(
                    f"Metadata entry {position} is missing: {missing}"
                )

            try:
                chunk_id = int(item["chunk_id"])
                first_index = int(item["first_character_index"])
                last_index = int(item["last_character_index"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Metadata entry {position} has invalid indices"
                ) from error

            if chunk_id != position:
                raise ValueError(
                    "Chunk IDs must match their positions in chunks.json"
                )

            if first_index < 0 or last_index < first_index:
                raise ValueError(
                    f"Metadata entry {position} has an invalid range"
                )

            if last_index - first_index > 2000:
                raise ValueError(
                    f"Metadata entry {position} exceeds 2000 characters"
                )

            metadata.append(item)

        return metadata

    def _load_bm25(
        self,
    ) -> tuple[bm25s.BM25, bm25s.tokenization.Tokenizer]:
        """Load the BM25 retriever and tokenizer.

        Returns:
            Loaded BM25 retriever and tokenizer.
        """
        try:
            retriever = bm25s.BM25.load(
                str(self.index_path),
                load_corpus=True,
            )

            stemmer = Stemmer.Stemmer("english")
            tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer)
            tokenizer.load_vocab(str(self.index_path))
            tokenizer.load_stopwords(str(self.index_path))
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Unable to load BM25 index from {self.index_path}"
            ) from error

        return retriever, tokenizer

    def _load_semantic_index(self) -> None:
        """Load semantic embeddings and the embedding model.

        Raises:
            FileNotFoundError: If the embedding file is missing.
            ValueError: If the embedding array does not match metadata.
        """
        embeddings_path = self.index_path / "semantic_embeddings.npy"

        if not embeddings_path.is_file():
            raise FileNotFoundError(
                f"Embedding file does not exist: {embeddings_path}"
            )

        embeddings = np.load(
            embeddings_path,
            allow_pickle=False,
            mmap_mode="r",
        )

        if embeddings.ndim != 2:
            raise ValueError("semantic_embeddings.npy must contain a 2D array")

        if len(embeddings) != len(self.metadata):
            raise ValueError("Embedding count does not match metadata count")

        self.embeddings = embeddings
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

    @staticmethod
    def _deduplicate_ranking(ranking: list[int]) -> list[int]:
        """Remove repeated chunk IDs while preserving rank order.

        Args:
            ranking: Ranked chunk IDs.

        Returns:
            Deduplicated chunk IDs.
        """
        seen: set[int] = set()
        result: list[int] = []

        for chunk_id in ranking:
            if chunk_id in seen:
                continue

            seen.add(chunk_id)
            result.append(chunk_id)

        return result

    def _bm25_ranking(
        self,
        query: str,
        candidate_count: int,
    ) -> list[int]:
        """Retrieve candidate chunk IDs using BM25.

        Args:
            query: Search query.
            candidate_count: Maximum number of candidates.

        Returns:
            BM25-ranked chunk IDs.
        """
        if candidate_count <= 0:
            return []

        query_tokens = self.tokenizer.tokenize([query], return_as="tuple")

        if not isinstance(query_tokens, Tokenized):
            raise TypeError("query_tokens should by 'Tokenized' type")

        documents, _ = self.retriever.retrieve(
            query_tokens,
            k=candidate_count,
        )

        if len(documents) == 0:
            return []

        ranking: list[int] = []

        for document in documents[0]:
            try:
                chunk_id = int(document["chunk_id"])
            except (KeyError, TypeError, ValueError):
                continue

            if 0 <= chunk_id < len(self.metadata):
                ranking.append(chunk_id)

        return self._deduplicate_ranking(ranking)

    def _semantic_ranking(
        self,
        query: str,
        candidate_count: int,
    ) -> tuple[list[int], NDArray[np.float32]]:
        """Retrieve candidate chunk IDs using cosine similarity.

        Both query and document embeddings are normalized. Their dot
        product therefore corresponds to cosine similarity.

        Args:
            query: Search query.
            candidate_count: Maximum number of candidates.

        Returns:
            Semantic ranking and similarity scores for every chunk.

        Raises:
            RuntimeError: If the semantic index was not loaded.
        """
        if self.embedding_model is None or self.embeddings is None:
            raise RuntimeError("The semantic index is not loaded")

        encoded_query = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            encoded_query,
            dtype=np.float32,
        )

        scores = np.asarray(
            self.embeddings @ query_embedding,
            dtype=np.float32,
        )

        count = min(candidate_count, len(scores))

        if count <= 0:
            return [], scores

        if count == len(scores):
            ranking_array = np.argsort(-scores)
        else:
            candidate_ids = np.argpartition(
                scores,
                -count,
            )[-count:]

            ranking_array = candidate_ids[np.argsort(-scores[candidate_ids])]

        ranking = [int(chunk_id) for chunk_id in ranking_array.tolist()]

        return self._deduplicate_ranking(ranking), scores

    def _fuse_rankings(
        self,
        lexical_ranking: list[int],
        semantic_ranking: list[int],
        k: int,
    ) -> list[int]:
        """Combine lexical and semantic rankings using weighted RRF.

        RRF is preferable to directly combining BM25 and cosine scores
        because the two score types use different scales.

        Args:
            lexical_ranking: BM25-ranked chunk IDs.
            semantic_ranking: Semantically ranked chunk IDs.
            k: Number of final results.

        Returns:
            Fused top-k chunk IDs.
        """
        rrf_constant = 60
        fused_scores: dict[int, float] = {}

        for rank, chunk_id in enumerate(lexical_ranking, start=1):
            contribution = self.lexical_weight / (rrf_constant + rank)
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0) + contribution
            )

        for rank, chunk_id in enumerate(
            semantic_ranking,
            start=1,
        ):
            contribution = self.semantic_weight / (rrf_constant + rank)
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0) + contribution
            )

        ordered_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (
                fused_scores[chunk_id],
                -chunk_id,
            ),
            reverse=True,
        )

        return ordered_ids[:k]

    def _to_minimal_source(self, chunk_id: int) -> MinimalSource:
        """Convert chunk metadata to a MinimalSource.

        Args:
            chunk_id: Indexed chunk identifier.

        Returns:
            Validated source location.
        """
        metadata = self.metadata[chunk_id]

        return MinimalSource(
            file_path=str(metadata["file_path"]),
            first_character_index=int(metadata["first_character_index"]),
            last_character_index=int(metadata["last_character_index"]),
            chunk_id=int(metadata["chunk_id"]),
        )

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[MinimalSource]:
        """Return the top-k sources for a query.

        Args:
            query: Natural-language or code search query.
            k: Number of sources to retrieve.

        Returns:
            Ranked source locations.
        """
        clean_query = query.strip()

        cached_answer = self.get_cached_sources(query=clean_query)
        if cached_answer and len(cached_answer) >= k:
            return cached_answer[:k]

        if not clean_query or k <= 0 or not self.metadata:
            return []

        result_count = min(k, len(self.metadata))
        candidate_count = min(
            max(result_count, result_count * self.candidate_multiplier),
            len(self.metadata),
        )

        if self.retrieval_mode == "bm25":
            ranking = self._bm25_ranking(
                clean_query,
                candidate_count,
            )
            final_ranking = ranking[:result_count]

        elif self.retrieval_mode == "semantic":
            ranking, _ = self._semantic_ranking(
                clean_query,
                candidate_count,
            )
            final_ranking = ranking[:result_count]

        else:
            lexical_ranking = self._bm25_ranking(
                clean_query,
                candidate_count,
            )
            semantic_ranking, _ = self._semantic_ranking(
                clean_query,
                candidate_count,
            )

            final_ranking = self._fuse_rankings(
                lexical_ranking=lexical_ranking,
                semantic_ranking=semantic_ranking,
                k=result_count,
            )

        res = [self._to_minimal_source(chunk_id) for chunk_id in final_ranking]

        self.save_cached_sources(clean_query, res)

        return res


class SearchDataset:
    """Run retrieval over a JSON question dataset."""

    def __init__(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        retrieval_mode: str = "bm25",
        semantic_weight: float = 0.05,
        candidate_multiplier: int = 10,
    ) -> None:
        """Initialize dataset retrieval.

        Args:
            dataset_path: A dataset JSON file or directory.
            k: Number of results to retrieve per question.
            save_directory: Directory where results are written.
            retrieval_mode: BM25, semantic, or hybrid retrieval.
            semantic_weight: Semantic weight for hybrid retrieval.
            candidate_multiplier: Candidate pool multiplier.
        """
        self.dataset_path = Path(dataset_path)
        self.k = k
        self.save_directory = Path(save_directory)

        self.search_engine = Search(
            retrieval_mode=retrieval_mode,
            semantic_weight=semantic_weight,
            candidate_multiplier=candidate_multiplier,
        )

    def _find_dataset_files(self) -> list[Path]:
        """Find input dataset files.

        Returns:
            Sorted JSON dataset paths.
        """
        if self.dataset_path.is_file():
            if self.dataset_path.suffix.lower() == ".json":
                return [self.dataset_path]

            return []

        if self.dataset_path.is_dir():
            return sorted(self.dataset_path.rglob("*.json"))

        return []

    def load_dataset(
        self,
        path: Path,
    ) -> list[UnansweredQuestion]:
        """Load and validate a question dataset.

        Args:
            path: Input JSON path.

        Returns:
            Validated questions.

        Raises:
            ValueError: If the JSON structure is invalid.
            OSError: If the file cannot be read.
        """
        try:
            with path.open("r", encoding="utf-8") as file:
                content: object = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed dataset JSON: {path}") from error

        if not isinstance(content, dict):
            raise ValueError("The dataset root must be an object")

        questions = content.get("rag_questions")

        if not isinstance(questions, list):
            raise ValueError("The dataset must contain a rag_questions list")

        dataset: list[UnansweredQuestion] = []

        for position, question in enumerate(questions):
            if not isinstance(question, dict):
                raise ValueError(f"Question {position} must be an object")

            dataset.append(UnansweredQuestion(**question))

        return dataset

    def _save_results(
        self,
        results: StudentSearchResults,
        output_path: Path,
    ) -> None:
        """Save search results as JSON.

        Args:
            results: Validated search results.
            output_path: Destination JSON path.
        """
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                results.model_dump(),
                file,
                indent=4,
                ensure_ascii=False,
            )

    def search_dataset(self) -> None:
        """Search all questions and save their retrieval results."""
        if self.k < 0:
            print("Error: k cannot be negative")
            return

        dataset_files = self._find_dataset_files()

        if not dataset_files:
            print(f"No JSON dataset found at {self.dataset_path}")
            return

        self.save_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for dataset_file in dataset_files:
            try:
                questions = self.load_dataset(dataset_file)
            except (OSError, ValueError) as error:
                print(f"Unable to load {dataset_file}: {error}")
                continue

            search_results: list[MinimalSearchResults] = []

            for question in tqdm(
                questions,
                desc=f"Searching {dataset_file.name}",
            ):
                retrieved_sources = self.search_engine.search(
                    query=question.question,
                    k=self.k,
                )

                search_results.append(
                    MinimalSearchResults(
                        question_id=question.question_id,
                        question=question.question,
                        retrieved_sources=retrieved_sources,
                    )
                )

            student_results = StudentSearchResults(
                search_results=search_results,
                k=self.k,
            )

            output_path = self.save_directory / dataset_file.name

            try:
                self._save_results(
                    student_results,
                    output_path,
                )
            except OSError as error:
                print(f"Unable to save results to " f"{output_path}: {error}")
                continue

            print(f"Saved student search results to {output_path}")
