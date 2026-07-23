"""Main entry point for the project."""

from typing import NoReturn


def greet(name: str) -> str:
    """Return a greeting message.

    Args:
        name: The name to greet.

    Returns:
        A formatted greeting string.
    """
    return f"Hello, {name}!"


def main() -> int:
    """Run the main program.

    Returns:
        Exit status code.
    """
    try:
        message = greet("42")
        print(message)
        return 0
    except Exception as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
