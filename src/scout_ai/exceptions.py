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


class RetryableError(LLMClientError):
    """Rate limits, timeouts, 5xx — should be retried."""


class NonRetryableError(LLMClientError):
    """Auth errors, bad requests, 4xx (non-429) — fail immediately."""


class JSONParseError(ScoutError):
    """LLM response could not be parsed as JSON."""

    def __init__(self, message: str, raw_response: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response


class TokenizerError(ScoutError):
    """Raised when token counting encounters an error."""
