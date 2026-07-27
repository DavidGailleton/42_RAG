from pathlib import Path

import json

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from pydantic.types import T

from src.classes.models import MinimalSource


class Indexer:
    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size = max_chunk_size
        self.input_path = Path("data/raw/")
        self.output_path = Path("data/processed/")

    def split_text(self, file: Path) -> list[str]:
        with open(file, encoding="utf-8") as content:
            match file.name.split(".")[-1]:
                case "py":
                    text_splitter = (
                        RecursiveCharacterTextSplitter.from_language(
                            language=Language.PYTHON,
                            chunk_size=self.max_chunk_size,
                            chunk_overlap=10,
                        )
                    )
                case "md":
                    text_splitter = (
                        RecursiveCharacterTextSplitter.from_language(
                            language=Language.MARKDOWN,
                            chunk_size=self.max_chunk_size,
                            chunk_overlap=10,
                        )
                    )
                case "rst":
                    text_splitter = (
                        RecursiveCharacterTextSplitter.from_language(
                            language=Language.RST,
                            chunk_size=self.max_chunk_size,
                            chunk_overlap=10,
                        )
                    )
                case _:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.max_chunk_size,
                        chunk_overlap=10,
                    )
            return text_splitter.split_text(content.read())

    def chunking(self) -> list[MinimalSource]:
        res: list[MinimalSource] = []

        for file in self.input_path.rglob("*"):
            if file.name.endswith(
                (
                    ".py",
                    ".md",
                    ".rst",
                    ".txt",
                    ".toml",
                    ".yaml",
                    ".yml",
                    ".json",
                )
            ):
                file_chunks: list[str] = self.split_text(file)

                with open(file, encoding="utf-8") as content:
                    text = content.read()

                    for chunk in file_chunks:
                        start = text.index(chunk)
                        end = start + len(chunk)

                        res.append(
                            MinimalSource(
                                file_path=file.name,
                                first_character_index=start,
                                last_character_index=end,
                                text=chunk,
                            )
                        )

        return res

    def index(self):
        chunks = self.chunking()
        with open("data/processed/chunks.json", "w", encoding="utf-8") as file:
            json.dump(
                [obj.model_dump_json() for obj in chunks],
                file,
                ensure_ascii=True,
            )
