"""Twitch facade with provider selection."""
from typing import List

from ..logging_setup import get_logger
from ..models import TwitchClip
from .base import TwitchProvider
from .graphql_provider import TwitchGraphQLProvider
from .helix_provider import TwitchHelixProvider

log = get_logger(__name__)


class TwitchAPIError(RuntimeError):
    pass


class TwitchAPI:
    def __init__(self, provider: TwitchProvider):
        self._provider = provider

    @classmethod
    def from_config(cls, cfg) -> "TwitchAPI":
        mode = (cfg.twitch_provider or "graphql").lower()
        if mode == "helix":
            if not cfg.twitch_client_id or not cfg.twitch_client_secret:
                raise TwitchAPIError(
                    "TWITCH_PROVIDER=helix requires TWITCH_CLIENT_ID and "
                    "TWITCH_CLIENT_SECRET in .env. Without credentials use "
                    "TWITCH_PROVIDER=graphql."
                )
            provider = TwitchHelixProvider(
                cfg.twitch_client_id, cfg.twitch_client_secret, cfg.twitch_timeout
            )
        elif mode == "graphql":
            provider = TwitchGraphQLProvider(cfg.twitch_web_client_id,
                                             cfg.twitch_timeout)
        else:
            raise TwitchAPIError(f"Unknown TWITCH_PROVIDER: {mode}")
        log.info("Twitch provider selected: %s", provider.name)
        return cls(provider)

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def health_check(self) -> bool:
        return self._provider.health_check()

    def get_clips(self, game_name: str, limit: int, period: str = "week") -> List[TwitchClip]:
        try:
            return self._provider.get_clips(game_name, limit, period)
        except Exception as e:
            raise TwitchAPIError(f"Twitch provider '{self._provider.name}' "
                                f"failed: {e}") from e
