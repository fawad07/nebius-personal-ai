class ConversationError(Exception):
    """Base class for conversation-related errors."""
    pass


class PromptBuildError(ConversationError):
    """Raised when the conversation manager fails to build a prompt."""
    pass


class LLMResponseError(ConversationError):
    """Raised when the LLM returns an invalid or unusable response."""
    pass

