"""Official Twitch Helix provider. Ready for future official API access."""
import time
from typing import Any, Dict, List, Optional

import requests

from ..logging_setup import get_logger
from ..models import TwitchClip
from .base import TwitchProvider

log = get_logger(__name__)

HELIX = "https://api.twitch.tv/helix"
OAUTH = "https://id.twitch.tv/oauth2/token"


class TwitchProviderError(RuntimeError):
    pass


class TwitchHelixProvider(TwitchProvider):
    def __init__(self, client_id: str, client_secret: str, timeout: float = 20.0):
        if not client_id or not client_secret:
            raise ValueError(
                "Helix provider requires TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET"
            )
        self._client_id = client_id
        self._client_secret = client_secret
        self._timeout = timeout
        self._token: Optional[str] = None
        self._expires_at = 0.0
        self._session = requests.Session()

    @property
    def name(self) -> str:
        return "helix"

    def _ensure_token(self):
        if self._token and time.time() < self._expires_at - 60:
            return
        r = self._session.post(OAUTH, params={
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "grant_type": "client_credentials",
        }, timeout=self._timeout)
        if r.status_code != 200:
            raise TwitchProviderError(
                f"OAuth failed: HTTP {r.status_code}: {r.text[:200]}"
            )
        d = r.json()
        self._token = d["access_token"]
        self._expires_at = time.time() + int(d.get("expires_in", 3600))

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        for attempt in range(3):
            self._ensure_token()
            r = self._session.get(
                HELIX + path,
                headers={"Client-ID": self._client_id,
                         "Authorization": f"Bearer {self._token}"},
                params=params, timeout=self._timeout,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code == 401:
                self._token = None
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                continue
            raise TwitchProviderError(
                f"Helix HTTP {r.status_code}: {r.text[:200]}"
            )
        raise TwitchProviderError("Helix: exhausted retries")

    def _resolve_game_id(self, name: str) -> Optional[str]:
        items = (self._get("/games", {"name": name}) or {}).get("data") or []
        return items[0]["id"] if items else None

    def get_clips(self, game_name, limit, period="week", max_pages=10) -> List[TwitchClip]:
        game_id = self._resolve_game_id(game_name)
        if not game_id:
            log.warning("Game not found on Helix: %s", game_name)
            return []
        clips: List[TwitchClip] = []
        cursor = None
        for _ in range(max_pages):
            if len(clips) >= limit:
                break
            params: Dict[str, Any] = {"game_id": game_id,
                                     "first": min(100, limit - len(clips))}
            if cursor:
                params["after"] = cursor
            d = self._get("/clips", params)
            data = d.get("data") or []
            if not data:
                break
            for it in data:
                clips.append(TwitchClip(
                    id=it["id"], title=it.get("title", ""),
                    game_name=game_name,
                    broadcaster_name=it.get("broadcaster_name", ""),
                    url=it.get("url", ""), created_at=it.get("created_at", ""),
                    duration=float(it.get("duration", 0)),
                    view_count=int(it.get("view_count", 0)),
                    thumbnail_url=it.get("thumbnail_url", ""),
                ))
            cursor = (d.get("pagination") or {}).get("cursor")
            if not cursor:
                break
            time.sleep(0.2)
        return clips[:limit]

    def health_check(self) -> bool:
        try:
            self._ensure_token()
            return True
        except Exception as e:
            log.warning("Helix health check failed: %s", e)
            return False
