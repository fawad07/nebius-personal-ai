class AgentError(Exception):
    """Base class for agent-level errors."""
    pass


class SessionError(AgentError):
    """Raised when the session manager encounters a fatal error."""
    pass


class InvalidStateError(AgentError):
    """Raised when the agent enters an invalid or unexpected state."""
    pass

