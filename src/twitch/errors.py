"""Twitch provider exceptions.

All provider failures inherit from TwitchAPIError so the pipeline can handle
them uniformly, while callers can still distinguish auth/schema/limit errors.
"""


class TwitchAPIError(RuntimeError):
    """Base class for Twitch configuration or provider errors."""


class TwitchProviderError(TwitchAPIError):
    """Generic provider/transport failure."""


class TwitchAuthError(TwitchProviderError):
    """Credentials or public client id were rejected (HTTP 401/403)."""


class TwitchRateLimitError(TwitchProviderError):
    """Provider kept returning HTTP 429 after retries."""


class TwitchGameNotFound(TwitchProviderError):
    """The requested game/category does not exist on Twitch."""


class TwitchSchemaError(TwitchProviderError):
    """The provider response does not match the expected schema."""
