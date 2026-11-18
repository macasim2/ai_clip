"""Official Twitch Helix provider (optional; requires app credentials).

The project cannot register its own Twitch application yet, so this provider
is implemented for the future and is covered by mock tests.  It is never
required for the default GraphQL mode.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from ..logging_setup import get_logger
from ..models import ProviderHealth, TwitchClip
from ..utils import normalize_game_name
from .base import TwitchProvider
from .errors import (TwitchAPIError, TwitchAuthError, TwitchGameNotFound,
                     TwitchProviderError, TwitchRateLimitError,
                     TwitchSchemaError)

log = get_logger(__name__)

HELIX = "https://api.twitch.tv/helix"
OAUTH = "https://id.twitch.tv/oauth2/token"
PERIOD_DAYS = {"day": 1, "week": 7, "month": 30}


class TwitchHelixProvider(TwitchProvider):
    def __init__(self, client_id: str, client_secret: str,
                 timeout: float = 20.0, max_pages: int = 20):
        if not client_id or not client_secret:
            raise TwitchAPIError(
                "Helix provider requires TWITCH_CLIENT_ID and "
                "TWITCH_CLIENT_SECRET"
            )
        self._client_id = client_id
        self._client_secret = client_secret
        self._timeout = timeout
        self._max_pages = max(1, int(max_pages))
        self._token: Optional[str] = None
        self._expires_at = 0.0
        self._token_lock = threading.Lock()
        self._session = requests.Session()

    @property
    def name(self) -> str:
        return "helix"

    # ------------------------------------------------------------------- token
    def _ensure_token(self, force: bool = False) -> None:
        with self._token_lock:
            if not force and self._token and time.time() < self._expires_at - 60:
                return
            try:
                response = self._session.post(
                    OAUTH,
                    params={
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                        "grant_type": "client_credentials",
                    },
                    timeout=self._timeout,
                )
            except requests.RequestException as exc:
                raise TwitchProviderError(
                    f"Cannot reach Twitch OAuth endpoint: {exc}"
                ) from exc
            if response.status_code != 200:
                if response.status_code in (400, 401, 403):
                    raise TwitchAuthError(
                        f"Twitch OAuth rejected credentials (HTTP "
                        f"{response.status_code}): {response.text[:200]}"
                    )
                raise TwitchProviderError(
                    f"Twitch OAuth failed (HTTP {response.status_code}): "
                    f"{response.text[:200]}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise TwitchProviderError("Twitch OAuth returned invalid JSON") from exc
            token = payload.get("access_token")
            if not token:
                raise TwitchAuthError("Twitch OAuth response has no access_token")
            self._token = str(token)
            self._expires_at = time.time() + float(payload.get("expires_in", 3600))

    # ---------------------------------------------------------------- requests
    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Optional[str] = None
        last_status: Optional[int] = None
        for attempt in range(4):
            self._ensure_token()
            try:
                response = self._session.get(
                    HELIX + path,
                    headers={
                        "Client-ID": self._client_id,
                        "Authorization": f"Bearer {self._token}",
                    },
                    params=dict(params),
                    timeout=self._timeout,
                )
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                time.sleep(1.5 ** attempt)
                continue
            status = response.status_code
            last_status = status
            if status == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise TwitchProviderError(
                        f"Helix {path} returned invalid JSON"
                    ) from exc
                if not isinstance(payload, dict):
                    raise TwitchSchemaError(
                        f"Helix {path} returned {type(payload).__name__}"
                    )
                return payload
            if status == 401:
                self._token = None
                last_error = "unauthorized (token refreshed)"
                if attempt == 0:
                    continue
                raise TwitchAuthError(
                    "Twitch Helix rejected the access token twice"
                )
            if status == 403:
                raise TwitchAuthError(
                    f"Twitch Helix forbidden (HTTP 403): {response.text[:200]}"
                )
            if status == 429:
                wait = _retry_after(response, default=2.0 ** attempt)
                last_error = f"rate limited, waiting {wait:.1f}s"
                log.warning("Helix rate limit: waiting %.1fs", wait)
                time.sleep(wait)
                continue
            if 500 <= status < 600:
                wait = _retry_after(response, default=2.0 ** attempt)
                last_error = f"server error HTTP {status}"
                time.sleep(wait)
                continue
            raise TwitchProviderError(
                f"Helix {path} HTTP {status}: {response.text[:300]}"
            )
        if last_status == 429:
            raise TwitchRateLimitError(
                f"Twitch Helix rate limit persisted after retries ({last_error})"
            )
        raise TwitchProviderError(
            f"Twitch Helix {path} failed after retries: {last_error}"
        )

    # -------------------------------------------------------------------- game
    def _resolve_game(self, game_name: str) -> Tuple[str, str]:
        query_name = normalize_game_name(game_name) or game_name
        data = self._get("/games", {"name": query_name})
        games = data.get("data") or []
        exact = next(
            (game for game in games
             if str(game.get("name", "")).casefold() == query_name.casefold()),
            None,
        )
        chosen = exact or (games[0] if games else None)
        if chosen and chosen.get("id"):
            return str(chosen["id"]), str(chosen.get("name") or query_name)

        # Twitch categories sometimes differ from game names ("Counter-Strike 2"
        # resolves to "Counter-Strike").  Helix has a category search endpoint.
        search = self._get("/search/categories", {"query": query_name,
                                                  "first": 5})
        categories = search.get("data") or []
        exact_cat = next(
            (item for item in categories
             if str(item.get("name", "")).casefold() == query_name.casefold()),
            None,
        )
        category = exact_cat or (categories[0] if categories else None)
        if category and category.get("id"):
            return str(category["id"]), str(category.get("name") or query_name)
        raise TwitchGameNotFound(
            f"Game not found on Twitch Helix: {game_name!r}"
        )

    # ------------------------------------------------------------------- clips
    def get_clips(self, game_name: str, limit: int, period: str = "week",
                  max_pages: int = 20) -> List[TwitchClip]:
        limit = int(limit)
        if limit <= 0:
            return []
        if period not in ("day", "week", "month", "all"):
            raise ValueError(f"Unknown period: {period!r}")
        game_id, official_name = self._resolve_game(game_name)
        cutoff: Optional[datetime] = None
        if period != "all":
            cutoff = datetime.now(timezone.utc) - timedelta(days=PERIOD_DAYS[period])

        pages = max(1, min(int(max_pages or self._max_pages), self._max_pages))
        found: Dict[str, TwitchClip] = {}
        cursor: Optional[str] = None
        for page in range(pages):
            if len(found) >= limit:
                break
            params: Dict[str, Any] = {
                "game_id": game_id,
                "first": min(100, limit - len(found)),
            }
            if cursor:
                params["after"] = cursor
            try:
                data = self._get("/clips", params)
            except TwitchAPIError as exc:
                raise type(exc)(
                    f"Twitch Helix page {page + 1} for {official_name!r} "
                    f"failed: {exc}"
                ) from exc
            clips = data.get("data") or []
            if not clips:
                break
            for item in clips:
                try:
                    clip = self._parse_item(item, official_name, game_id)
                except (KeyError, TypeError, ValueError) as exc:
                    log.warning("Skipping malformed Helix clip: %s", exc)
                    continue
                if cutoff is not None:
                    created = _parse_iso(clip.created_at)
                    if created is not None and created < cutoff:
                        continue
                if clip.id not in found:
                    found[clip.id] = clip
            cursor = (data.get("pagination") or {}).get("cursor")
            if not cursor:
                break
            if len(found) < limit:
                time.sleep(0.2)
        result = list(found.values())[:limit]
        log.info("Helix: fetched %d clip(s) for %r (period=%s)",
                 len(result), official_name, period)
        return result

    @staticmethod
    def _parse_item(item: Dict[str, Any], game_name: str,
                    game_id: str) -> TwitchClip:
        clip_id = str(item["id"]).strip()
        if not clip_id:
            raise ValueError("empty clip id")
        return TwitchClip(
            id=clip_id,
            title=str(item.get("title") or ""),
            game_name=game_name,
            broadcaster_name=str(item.get("broadcaster_name") or ""),
            url=str(item.get("url") or ""),
            created_at=str(item.get("created_at") or ""),
            duration=float(item.get("duration") or 0),
            view_count=int(item.get("view_count") or 0),
            thumbnail_url=str(item.get("thumbnail_url") or ""),
            game_id=game_id,
        )

    # ------------------------------------------------------------------ health
    def check_health(self) -> ProviderHealth:
        try:
            self._ensure_token()
            data = self._get("/games", {"name": "Counter-Strike"})
            games = data.get("data") or []
            if not games:
                return ProviderHealth(
                    ok=False, provider=self.name,
                    detail="authenticated, but game lookup returned no data",
                )
            return ProviderHealth(ok=True, provider=self.name,
                                  detail="OAuth + Helix OK")
        except TwitchAPIError as exc:
            return ProviderHealth(ok=False, provider=self.name,
                                  detail=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            return ProviderHealth(ok=False, provider=self.name,
                                  detail=f"{type(exc).__name__}: {exc}")

    def health_check(self) -> bool:
        return self.check_health().ok


def _retry_after(response: Any, default: float) -> float:
    value = (getattr(response, "headers", {}) or {}).get("Retry-After")
    try:
        wait = float(value)
    except (TypeError, ValueError):
        return max(0.5, min(default, 30.0))
    return max(0.5, min(wait, 30.0))


def _parse_iso(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
