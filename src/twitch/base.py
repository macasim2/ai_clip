"""Common Twitch provider interface."""
from abc import ABC, abstractmethod
from typing import List

from ..models import ProviderHealth, TwitchClip


class TwitchProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def health_check(self) -> bool: ...

    @abstractmethod
    def get_clips(self, game_name: str, limit: int, period: str = "week",
                  max_pages: int = 20) -> List[TwitchClip]: ...

    def check_health(self) -> ProviderHealth:
        """Detailed health information; providers may override this."""
        try:
            ok = self.health_check()
            return ProviderHealth(ok=ok, provider=self.name,
                                  detail="OK" if ok else "unreachable")
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return ProviderHealth(ok=False, provider=self.name,
                                  detail=f"{type(exc).__name__}: {exc}")
