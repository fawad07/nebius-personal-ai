class LLMError(Exception):
    """Base class for LLM client errors."""
    pass


class LLMAuthError(LLMError):
    """Raised when the configured OpenAI API key is missing or invalid."""
    pass


class LLMRateLimitExceeded(LLMError):
    """Raised when the local call-rate guard rejects a request."""
    pass


class LLMBudgetExceededError(LLMError):
    """Raised when the local token-budget guard rejects a request."""
    pass


class LLMResponseError(LLMError):
    """Raised when the model's response can't be parsed as expected (e.g. no JSON)."""
    pass
