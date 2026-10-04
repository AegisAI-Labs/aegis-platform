class LLMProviderError(Exception):
    """Base exception for LLM provider failures."""


class ProviderUnavailable(LLMProviderError):
    """Provider is unavailable."""


class ProviderTimeout(LLMProviderError):
    """Provider request timed out."""


class ProviderRateLimited(LLMProviderError):
    """Provider rejected the request due to rate limiting."""


class ProviderAuthenticationError(LLMProviderError):
    """Provider authentication failed."""


class ProviderRequestError(LLMProviderError):
    """Provider rejected or could not process the request."""
