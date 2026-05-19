"""Load configuration from secrets.txt."""

from pathlib import Path


def load_secrets(path: Path | None = None) -> dict[str, str]:
    """Read key=value pairs from secrets.txt, ignoring comments and blanks."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "secrets.txt"
    secrets: dict[str, str] = {}
    if not path.exists():
        return secrets
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value:
            secrets[key] = value
    return secrets


_secrets: dict[str, str] | None = None


def _get_secrets() -> dict[str, str]:
    global _secrets
    if _secrets is None:
        _secrets = load_secrets()
    return _secrets


def get_anthropic_client():
    """Return an Anthropic client using the key from secrets.txt."""
    import anthropic

    api_key = _get_secrets().get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in secrets.txt")
    return anthropic.Anthropic(api_key=api_key)


def get_semantic_scholar_headers() -> dict[str, str]:
    """Return request headers for Semantic Scholar API.

    Returns an empty dict if no key is configured (unauthenticated, 1 req/s).
    """
    key = _get_secrets().get("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        return {"x-api-key": key}
    return {}


def get_ncbi_api_key() -> str | None:
    """Return NCBI API key if configured, else None."""
    return _get_secrets().get("NCBI_API_KEY")
