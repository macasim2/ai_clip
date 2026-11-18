"""Twitch facade with provider selection.

The rest of the application only depends on this facade, so switching between
the temporary GraphQL provider and the official Helix provider is a config
change, not a pipeline/database rewrite.
"""
from __future__ import annotations

from typing import List

from ..logging_setup import get_logger
from ..models import ProviderHealth, TwitchClip
from .base import TwitchProvider
from .errors import (TwitchAPIError, TwitchAuthError, TwitchGameNotFound,
                     TwitchProviderError, TwitchRateLimitError,
                     TwitchSchemaError)
from .graphql_provider import TwitchGraphQLProvider
from .helix_provider import TwitchHelixProvider

log = get_logger(__name__)

__all__ = [
    "TwitchAPI", "TwitchAPIError", "TwitchProvider", "TwitchClip",
    "TwitchAuthError", "TwitchProviderError", "TwitchRateLimitError",
    "TwitchGameNotFound", "TwitchSchemaError",
]


class TwitchAPI:
    def __init__(self, provider: TwitchProvider):
        self._provider = provider

    @classmethod
    def from_config(cls, cfg) -> "TwitchAPI":
        mode = (cfg.twitch_provider or "graphql").lower()
        if mode == "helix":
            if not (cfg.twitch_client_id and cfg.twitch_client_secret):
                raise TwitchAPIError(
                    "TWITCH_PROVIDER=helix requires TWITCH_CLIENT_ID and "
                    "TWITCH_CLIENT_SECRET in .env. Without own credentials "
                    "use TWITCH_PROVIDER=graphql."
                )
            provider = TwitchHelixProvider(
                cfg.twitch_client_id,
                cfg.twitch_client_secret,
                cfg.twitch_timeout,
                cfg.twitch_max_pages,
            )
        elif mode == "graphql":
            provider = TwitchGraphQLProvider(
                cfg.twitch_web_client_id,
                cfg.twitch_timeout,
                cfg.twitch_max_pages,
            )
        else:
            raise TwitchAPIError(f"Unknown TWITCH_PROVIDER: {mode!r}")
        log.info("Twitch provider selected: %s", provider.name)
        return cls(provider)

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def check_health(self) -> ProviderHealth:
        return self._provider.check_health()

    def health_check(self) -> bool:
        return self._provider.health_check()

    def get_clips(self, game_name: str, limit: int,
                  period: str = "week") -> List[TwitchClip]:
        # Provider exceptions are intentionally not converted to a generic
        # error: TwitchGameNotFound, TwitchAuthError and TwitchSchemaError
        # require different user-facing messages.
        return self._provider.get_clips(game_name, limit, period)
