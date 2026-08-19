import json
from pathlib import Path
from typing import Any

import bm25s
import numpy as np
import Stemmer
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer

from src.classes.models import FileInformation, MinimalSource


"""Build and persist lexical and semantic indexes for the source corpus."""


class Indexer:
    """Build retrieval indexes from files in the raw-data directory.

    The indexer splits supported source files into character-bounded chunks,
    builds a BM25 lexical index and a semantic embedding index, and persists
    the generated data under ``data/processed/``.

    Attributes:
        max_chunk_size: Maximum number of characters in a chunk.
        input_path: Directory containing source files to index.
        output_path: Directory in which generated indexes are stored.
    """

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialize the indexer.

        Args:
            max_chunk_size: Maximum number of characters allowed in a chunk.

        Raises:
            ValueError: If ``max_chunk_size`` is outside the range 1 to 2000.
        """
        if not 1 <= max_chunk_size <= 2000:
            raise ValueError(
                "max_chunk_size must be between 1 and 2000 characters"
            )

        self.max_chunk_size = max_chunk_size
        self.input_path = Path("data/raw/")
        self.output_path = Path("data/processed/")

    def make_splitter(
        self,
        suffix: str,
    ) -> RecursiveCharacterTextSplitter:
        """Create a splitter appropriate for the file type.

        Args:
            suffix: Lowercase source-file suffix.

        Returns:
            A text splitter configured for the source language.
        """
        if suffix == ".py":
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=self.max_chunk_size,
                chunk_overlap=200,
                add_start_index=True,
                strip_whitespace=False,
            )

        if suffix == ".md":
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=self.max_chunk_size,
                chunk_overlap=200,
                add_start_index=True,
                strip_whitespace=False,
            )

        if suffix == ".rst":
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.RST,
                chunk_size=self.max_chunk_size,
                chunk_overlap=200,
                add_start_index=True,
                strip_whitespace=False,
            )

        return RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=200,
            add_start_index=True,
            strip_whitespace=False,
        )

    def split_text(
        self,
        file: Path,
    ) -> list[tuple[str, int, int]]:
        """Split a source file while preserving character positions.

        Args:
            file: Path to the source file to split.

        Returns:
            A list of tuples containing the chunk text, inclusive start
            position, and exclusive end position.

        Raises:
            OSError: If the source file cannot be read.
            ValueError: If the splitter cannot process the file contents.
        """
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
            Metadata indexed by source file path.
            An empty dictionary is returned when \
            the cache is missing or invalid.
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
        for old_chunk in old_chunks:
            old_chunks_by_path.setdefault(
                old_chunk.file_path,
                [],
            ).append(old_chunk)

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

        from tqdm import tqdm

        for _, file_path in tqdm(
            enumerate(current_files), desc="Chunking files"
        ):
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
                result.extend(old_chunks_by_path[path_string])
                updated_cache.append(current_information)
                continue

            try:
                split_chunks = self.split_text(file_path)
            except (OSError, UnicodeError, ValueError) as error:
                print(f"Warning: cannot chunk {path_string}: {error}")
                continue

            for _chunk_text, start_index, end_index in split_chunks:
                result.append(
                    MinimalSource(
                        file_path=path_string,
                        first_character_index=start_index,
                        last_character_index=end_index,
                        chunk_id=next_chunk_id,
                    )
                )
                next_chunk_id += 1

            updated_cache.append(current_information)

        self._save_file_cache(cache_path, updated_cache)
        return result

    def embedding(
        self,
        chunks: list[str],
    ) -> np.typing.NDArray[np.float32]:
        """Create normalized semantic embeddings for source chunks.

        Args:
            chunks: Source texts to encode.

        Returns:
            A two-dimensional NumPy array containing one normalized embedding
            per source chunk.

        Raises:
            OSError: If the embedding model cannot be loaded.
            RuntimeError: If embedding generation fails.
        """
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
        """Build and persist the lexical and semantic indexes.

        Existing chunk metadata is loaded when possible to support incremental
        indexing. The method writes the BM25 index, tokenizer data, semantic
        embeddings, chunk metadata, and file-information cache under the
        processed-data directory.

        Returns:
            None.
        """
        chunks_path = self.output_path / "chunks.json"

        try:
            with chunks_path.open("r", encoding="utf-8") as file:
                raw_chunks: Any = json.load(file)

            if not isinstance(raw_chunks, list):
                raise ValueError("chunks.json must contain a JSON list")

            old_chunks = [
                MinimalSource.model_validate(chunk) for chunk in raw_chunks
            ]
        except (
            FileNotFoundError,
            OSError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as error:
            print(f"Warning: cannot load previous chunks: {error}")
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
