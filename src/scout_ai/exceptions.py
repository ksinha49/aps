"""Exception hierarchy for scout-ai."""


class ScoutError(Exception):
    """Base exception for all scout-ai errors."""


class IndexBuildError(ScoutError):
    """Raised when tree index construction fails."""


class RetrievalError(ScoutError):
    """Raised when tree search / retrieval fails."""


class ExtractionError(ScoutError):
    """Raised when answer extraction from retrieved context fails."""


class LLMClientError(ScoutError):
    """Raised when LLM API calls fail after exhausting retries."""


class TokenizerError(ScoutError):
    """Raised when token counting encounters an error."""
