from src.classes.models import MinimalSource
from src.local_llm import LocalQwen
from src.search import Search


class Answer:
    """Generate answers from retrieved codebase chunks."""

    def __init__(self, llm: LocalQwen | None = None) -> None:
        """Initialize the answer generator."""
        self.llm = llm or LocalQwen()

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
            top_k = Search.search(query, k)

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
                f"Content:\n{source.text}"
            )

        context = "\n\n".join(context_parts)

        system_prompt = (
            "You answer questions about a codebase. "
            "Use only the retrieved sources. "
            "Do not invent information. "
            "If the sources are insufficient, clearly say so."
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
        except OSError as error:
            return f"Unable to load the language model: {error}"
        except RuntimeError as error:
            return f"Model generation failed: {error}"
