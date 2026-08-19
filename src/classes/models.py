"""Define the Pydantic data models used by the RAG pipeline."""

import uuid
from pathlib import Path

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represent a source location in the indexed corpus.

    Attributes:
        file_path: Path to the source file relative to the project root.
        first_character_index: Inclusive start position of the source text.
        last_character_index: Exclusive end position of the source text.
        chunk_id: Optional identifier assigned to the indexed chunk.
    """

    file_path: str
    first_character_index: int
    last_character_index: int
    chunk_id: int | None = Field(default=None)

    def get_text(self) -> str:
        """Read the text covered by this source's character range.

        Invalid byte sequences are replaced while reading the source file.

        Returns:
            The portion of the file between the source's start and end
            character indices.

        Raises:
            OSError: If the source file cannot be opened or read.
        """
        path = Path(self.file_path)

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:
            content = file.read()

        return content[self.first_character_index:self.last_character_index]


class FileInformation(BaseModel):
    """Store metadata used to determine whether a source file changed.

    Attributes:
        file_path: Path to the source file.
        last_update_ns: File modification time expressed in nanoseconds.
        file_size: Size of the file in bytes.
    """

    file_path: str
    last_update_ns: int
    file_size: int


class ExtendedMinimalSource(MinimalSource):
    """Represent a source location together with its extracted text.

    Attributes:
        text: Text extracted from the source's character range.
    """

    text: str


class UnansweredQuestion(BaseModel):
    """Represent a question that has not yet been answered.

    Attributes:
        question_id: Unique identifier for the question. A UUID is generated
            automatically when no identifier is provided.
        question: Natural-language question submitted to the RAG pipeline.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with its expected sources and answer.

    Attributes:
        sources: Source locations containing evidence for the answer.
        answer: Natural-language answer grounded in the sources.
    """

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a dataset of answered or unanswered RAG questions.

    Attributes:
        rag_questions: Questions contained in the dataset.
    """

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represent retrieved sources for one question.

    Attributes:
        question_id: Unique identifier of the searched question.
        question: Natural-language question used for retrieval.
        retrieved_sources: Ranked source locations returned by retrieval.
    """

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represent retrieval results together with a generated answer.

    Attributes:
        answer: Natural-language answer generated from the retrieved sources.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Represent retrieval results for a collection of questions.

    Attributes:
        search_results: Retrieval results for each processed question.
        k: Maximum number of sources requested for each question.
    """

    search_results: list[MinimalSearchResults]
    k: int

    def get_msr_by_question(
        self,
        question: str,
    ) -> MinimalSearchResults | None:
        """Find search results associated with an exact question.

        Args:
            question: Question text to find.

        Returns:
            The first matching search result, or ``None`` if the question is
            not present.
        """
        for search_result in self.search_results:
            if search_result.question == question:
                return search_result
        return None


class StudentSearchResultsAndAnswer(BaseModel):
    """Represent generated answers and sources for multiple questions.

    Attributes:
        search_results: Answered search results for each processed question.
        k: Maximum number of sources requested for each question.
    """

    search_results: list[MinimalAnswer]
    k: int
