"""Exception hierarchy for pageindex-rag."""


class PageIndexError(Exception):
    """Base exception for all pageindex-rag errors."""


class IndexBuildError(PageIndexError):
    """Raised when tree index construction fails."""


class RetrievalError(PageIndexError):
    """Raised when tree search / retrieval fails."""


class ExtractionError(PageIndexError):
    """Raised when answer extraction from retrieved context fails."""


class LLMClientError(PageIndexError):
    """Raised when LLM API calls fail after exhausting retries."""


class TokenizerError(PageIndexError):
    """Raised when token counting encounters an error."""
