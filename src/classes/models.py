"""Pydantic data models used by the RAG pipeline."""

import uuid
from pathlib import Path
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represent a source location in the indexed corpus."""

    file_path: str
    first_character_index: int
    last_character_index: int
    chunk_id: int | None = Field(default=None)

    def get_text(self) -> str:
        """Read the source text covered by this character range."""
        path = Path(self.file_path)

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:
            content = file.read()

        return content[self.first_character_index : self.last_character_index]


class FileInformation(BaseModel):
    """Metadata used to detect whether a source file changed."""

    file_path: str
    last_update_ns: int
    file_size: int


class ExtendedMinimalSource(MinimalSource):
    text: str


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int

    def get_msr_by_question(
        self, question: str
    ) -> MinimalSearchResults | None:
        for sr in self.search_results:
            if sr.question == question:
                return sr
        return None


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int
