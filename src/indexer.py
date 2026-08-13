from pathlib import Path

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from src.classes.models import MinimalSource

import bm25s
import Stemmer

from sentence_transformers import SentenceTransformer

import numpy as np


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

    def chunking(self) -> list[MinimalSource]:
        res: list[MinimalSource] = []

        for file in self.input_path.rglob("*"):
            if file.name.endswith(
                (
                    ".py",
                    ".md",
                    ".rst",
                    ".txt",
                )
            ):
                for chunk in self.split_text(file):
                    res.append(
                        MinimalSource(
                            file_path=file.__str__(),
                            first_character_index=chunk[1],
                            last_character_index=chunk[2],
                        )
                    )

        return res

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

        return embeddings.astype(np.float32)

    def index(self) -> None:
        """Build and persist lexical and semantic indexes."""
        chunks = self.chunking()
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
