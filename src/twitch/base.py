from abc import ABC, abstractmethod
from typing import List
from ..models import TwitchClip


class TwitchProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def health_check(self) -> bool: ...

    @abstractmethod
    def get_clips(self, game_name: str, limit: int, period: str = "week",
                  max_pages: int = 10) -> List[TwitchClip]: ...
