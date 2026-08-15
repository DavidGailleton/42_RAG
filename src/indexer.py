from pathlib import Path
from typing import Any

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from pydantic import ValidationError

from src.classes.models import FileInformation, MinimalSource

import bm25s
import Stemmer

from sentence_transformers import SentenceTransformer

import numpy as np

import json


class Indexer:
    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size = max_chunk_size
        self.input_path = Path("data/raw/")
        self.output_path = Path("data/processed/")

    def make_splitter(
        self,
        suffix: str,
    ) -> RecursiveCharacterTextSplitter:
        """Create a splitter appropriate for the file type."""
        common_options = {
            "chunk_size": self.max_chunk_size,
            "chunk_overlap": 200,
            "add_start_index": True,
            "strip_whitespace": False,
        }

        if suffix == ".py":
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                **common_options,
            )

        if suffix == ".md":
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                **common_options,
            )

        if suffix == ".rst":
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.RST,
                **common_options,
            )

        return RecursiveCharacterTextSplitter(**common_options)

    def split_text(
        self,
        file: Path,
    ) -> list[tuple[str, int, int]]:
        """Split a file while preserving exact character positions."""
        text = file.read_text(encoding="utf-8", errors="replace")
        splitter = self.make_splitter(file.suffix.lower())
        documents = splitter.create_documents([text])

        chunks: list[tuple[str, int, int]] = []

        for document in documents:
            start = int(document.metadata["start_index"])
            end = start + len(document.page_content)

            chunks.append((document.page_content, start, end))

        return chunks

    def _load_file_cache(
        self,
        cache_path: Path,
    ) -> dict[str, FileInformation]:
        """Load cached source-file metadata.

        Args:
            cache_path: Path to the metadata JSON file.

        Returns:
            Metadata indexed by source file path. An empty dictionary is returned
            when the cache is missing or invalid.
        """
        try:
            with cache_path.open("r", encoding="utf-8") as file:
                raw_data: Any = json.load(file)

            if not isinstance(raw_data, list):
                print(f"Warning: invalid cache format in {cache_path}")
                return {}

            file_information = [
                FileInformation.model_validate(item) for item in raw_data
            ]

            return {
                information.file_path: information
                for information in file_information
            }
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            print(f"Warning: cannot load cache {cache_path}: {error}")
            return {}

    def _save_file_cache(
        self,
        cache_path: Path,
        file_information: list[FileInformation],
    ) -> None:
        """Atomically save source-file metadata.

        Args:
            cache_path: Destination metadata file.
            file_information: Metadata for successfully indexed files.
        """
        temporary_path = cache_path.with_suffix(".tmp")

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            serialized_data = [
                information.model_dump(mode="json")
                for information in file_information
            ]

            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(serialized_data, file, indent=2)

            temporary_path.replace(cache_path)
        except OSError as error:
            print(f"Warning: cannot save cache {cache_path}: {error}")

            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    def chunking(
        self,
        old_chunks: list[MinimalSource],
    ) -> list[MinimalSource]:
        """Incrementally chunk source files.

        Existing chunks are preserved when their source file has not changed.
        New or modified files are chunked again. Chunks belonging to deleted
        files are removed.

        Args:
            old_chunks: Chunks loaded from the previously persisted index.

        Returns:
            The complete list of current chunks, including preserved and newly
            generated chunks.
        """
        SUPPORTED_SUFFIXES = {".py", ".md", ".rst", ".txt"}

        cache_path = self.output_path / "file_inf_cache.json"
        cached_files = self._load_file_cache(cache_path)

        old_chunks_by_path: dict[str, list[MinimalSource]] = {}
        for chunk in old_chunks:
            old_chunks_by_path.setdefault(chunk.file_path, []).append(chunk)

        current_files = sorted(
            (
                file
                for file in self.input_path.rglob("*")
                if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES
            ),
            key=lambda path: str(path),
        )

        result: list[MinimalSource] = []
        updated_cache: list[FileInformation] = []

        # Start after the largest existing ID to avoid duplicate chunk IDs.
        next_chunk_id = (
            max(
                (
                    chunk.chunk_id
                    for chunk in old_chunks
                    if chunk.chunk_id is not None
                ),
                default=-1,
            )
            + 1
        )

        for file_path in current_files:
            path_string = str(file_path)

            try:
                stat = file_path.stat()
            except OSError as error:
                print(f"Warning: cannot inspect {path_string}: {error}")
                continue

            current_information = FileInformation(
                file_path=path_string,
                last_update_ns=stat.st_mtime_ns,
                file_size=stat.st_size,
            )
            cached_information = cached_files.get(path_string)

            is_unchanged = (
                cached_information is not None
                and cached_information.last_update_ns
                == current_information.last_update_ns
                and cached_information.file_size
                == current_information.file_size
                and path_string in old_chunks_by_path
            )

            if is_unchanged:
                # Preserve chunks from the previous index.
                result.extend(old_chunks_by_path[path_string])
                updated_cache.append(current_information)
                continue

            # The file is new or changed. Do not preserve its old chunks.
            try:
                split_chunks = self.split_text(file_path)
            except (OSError, UnicodeError, ValueError) as error:
                print(f"Warning: cannot chunk {path_string}: {error}")
                # Do not cache the file, allowing the next run to retry it.
                continue

            for chunk in split_chunks:
                result.append(
                    MinimalSource(
                        file_path=path_string,
                        first_character_index=chunk[1],
                        last_character_index=chunk[2],
                        chunk_id=next_chunk_id,
                    )
                )
                next_chunk_id += 1

            updated_cache.append(current_information)

        self._save_file_cache(cache_path, updated_cache)
        return result

    def embedding(self, chunks: list[str]) -> np.typing.NDArray[np.float32]:
        """Create normalized semantic embeddings for chunks."""
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

        embeddings = model.encode(
            chunks,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def index(self) -> None:
        """Build and persist lexical and semantic indexes."""
        chunks_path = self.output_path / "chunks.json"

        try:
            with open(chunks_path) as file:
                old_chunks = [
                    MinimalSource(**chunk) for chunk in json.load(file)
                ]
        except FileNotFoundError:
            old_chunks = []

        chunks = self.chunking(old_chunks)
        chunks_lst = [chunk.get_text() for chunk in chunks]

        embeddings = self.embedding(chunks_lst)
        metadata_chunks = [chunk.model_dump() for chunk in chunks]

        stemmer = Stemmer.Stemmer("english")
        tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer)

        corpus_tokens = tokenizer.tokenize(
            chunks_lst,
            return_as="tuple",
        )

        retriever = bm25s.BM25(corpus=metadata_chunks)
        retriever.index(corpus_tokens)

        self.output_path.mkdir(parents=True, exist_ok=True)

        retriever.save(
            self.output_path,
            corpus=metadata_chunks,
        )
        tokenizer.save_vocab(str(self.output_path))
        tokenizer.save_stopwords(str(self.output_path))

        embeddings_path = self.output_path / "semantic_embeddings.npy"
        np.save(
            embeddings_path,
            embeddings,
            allow_pickle=False,
        )

        # This file maps semantic embedding row N to chunk metadata N.
        with (chunks_path).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata_chunks, file, indent=2, ensure_ascii=False)

        print(
            f"Indexed {len(chunks_lst)} chunks under "
            f"{self.output_path.as_posix()}"
        )
