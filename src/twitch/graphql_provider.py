"""Temporary Twitch web GraphQL provider (no own app credentials required).

This is the default provider because the project owner cannot currently
register a Twitch application (2FA).  It uses the public Twitch web Client-ID
and the same GraphQL operations the website uses.  It must never be treated
as an official application credential, and the pipeline must keep working
without TWITCH_CLIENT_ID/SECRET.
"""
from __future__ import annotations

import time
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

GQL_URL = "https://gql.twitch.tv/gql"
DEFAULT_WEB_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"

SEARCH_GAME = """query SearchCategory($query: String!) {
  searchCategories(query: $query, first: 5) {
    edges {
      node {
        id
        name
      }
    }
  }
}"""

GAME_CLIPS_BY_ID = """query ClipsCards__Game($id: ID!, $limit: Int, $cursor: Cursor, $criteria: GameClipsInput) {
  game(id: $id) {
    id
    name
    clips(first: $limit, after: $cursor, criteria: $criteria) {
      pageInfo { hasNextPage }
      edges {
        cursor
        node {
          id
          slug
          title
          url
          thumbnailURL
          viewCount
          createdAt
          durationSeconds
          broadcaster { displayName login }
        }
      }
    }
  }
}"""

PERIOD_MAP = {
    "day": "LAST_DAY",
    "week": "LAST_WEEK",
    "month": "LAST_MONTH",
    "all": "ALL_TIME",
}


class TwitchGraphQLProvider(TwitchProvider):
    def __init__(self, client_id: str = DEFAULT_WEB_CLIENT_ID,
                 timeout: float = 20.0, max_pages: int = 20):
        self._client_id = client_id or DEFAULT_WEB_CLIENT_ID
        self._timeout = timeout
        self._max_pages = max(1, int(max_pages))
        self._session = requests.Session()
        self._session.headers.update({
            "Client-ID": self._client_id,
            "Content-Type": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Origin": "https://www.twitch.tv",
            "Referer": "https://www.twitch.tv/",
        })
        self._health_cache: Optional[Tuple[float, ProviderHealth]] = None

    @property
    def name(self) -> str:
        return "graphql"

    # ------------------------------------------------------------------ health
    def check_health(self) -> ProviderHealth:
        now = time.time()
        if self._health_cache and now - self._health_cache[0] < 300:
            return self._health_cache[1]
        try:
            search = self._query(SEARCH_GAME, {"query": "Counter-Strike"})
            edges = ((search.get("searchCategories") or {}).get("edges") or [])
            if not edges:
                raise TwitchSchemaError(
                    "searchCategories returned no results for a known game"
                )
            game_id = (edges[0].get("node") or {}).get("id")
            if not game_id:
                raise TwitchSchemaError("searchCategories node has no id")
            payload = self._query(GAME_CLIPS_BY_ID, {
                "id": game_id,
                "limit": 1,
                "cursor": None,
                "criteria": {"period": "LAST_WEEK"},
            })
            game = payload.get("game")
            if game is None or not isinstance(game.get("clips"), dict):
                raise TwitchSchemaError(
                    "game.clips is missing; Twitch GraphQL schema may have changed"
                )
            health = ProviderHealth(
                ok=True, provider=self.name,
                detail=f"search+clips OK (game id {game_id})",
            )
        except TwitchAPIError as exc:
            health = ProviderHealth(ok=False, provider=self.name,
                                    detail=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - health must not crash startup
            health = ProviderHealth(ok=False, provider=self.name,
                                    detail=f"{type(exc).__name__}: {exc}")
        self._health_cache = (now, health)
        return health

    def health_check(self) -> bool:
        return self.check_health().ok

    # ------------------------------------------------------------------- query
    def _query(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Optional[str] = None
        last_status: Optional[int] = None
        for attempt in range(3):
            try:
                response = self._session.post(
                    GQL_URL,
                    json={"query": query, "variables": variables},
                    timeout=self._timeout,
                )
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                log.warning("Twitch GraphQL attempt %d failed: %s",
                            attempt + 1, exc)
                time.sleep(1.5 ** attempt)
                continue

            status = response.status_code
            last_status = status
            if status == 200:
                try:
                    payload = response.json()
                except ValueError:
                    last_error = "invalid JSON in response"
                    time.sleep(1.0 + attempt)
                    continue
                if not isinstance(payload, dict):
                    raise TwitchSchemaError(
                        f"Twitch GraphQL returned {type(payload).__name__}, "
                        "expected object"
                    )
                errors = payload.get("errors")
                if errors:
                    message = str(errors[0] if errors else "unknown GraphQL error")
                    raise TwitchSchemaError(
                        f"Twitch GraphQL errors: {message[:500]}"
                    )
                data = payload.get("data")
                if not isinstance(data, dict):
                    raise TwitchSchemaError(
                        "Twitch GraphQL response has no data object"
                    )
                return data
            if status in (401, 403):
                raise TwitchAuthError(
                    f"Twitch rejected the public Client-ID (HTTP {status}): "
                    f"{response.text[:200]}"
                )
            if status == 429:
                wait = _retry_after(response, default=2.0 ** attempt)
                last_error = f"rate limited (HTTP 429), waiting {wait:.1f}s"
                log.warning("Twitch rate limit: waiting %.1fs", wait)
                time.sleep(wait)
                continue
            if 500 <= status < 600:
                wait = _retry_after(response, default=2.0 ** attempt)
                last_error = f"server error HTTP {status}"
                time.sleep(wait)
                continue
            raise TwitchProviderError(
                f"Twitch HTTP {status}: {response.text[:300]}"
            )

        if last_status == 429:
            raise TwitchRateLimitError(
                f"Twitch rate limit persisted after retries ({last_error})"
            )
        raise TwitchProviderError(
            f"Twitch GraphQL request failed after retries: {last_error}"
        )

    # -------------------------------------------------------------------- game
    def _resolve_game(self, game_name: str) -> Tuple[str, str]:
        query_name = normalize_game_name(game_name) or game_name
        data = self._query(SEARCH_GAME, {"query": query_name})
        edges = ((data.get("searchCategories") or {}).get("edges") or [])
        candidates: List[Dict[str, Any]] = []
        for edge in edges:
            node = (edge or {}).get("node") or {}
            if node.get("id") and node.get("name"):
                candidates.append(node)
        if not candidates:
            raise TwitchGameNotFound(
                f"Game not found on Twitch: {game_name!r}. "
                "Check the game name or try its official Twitch category name."
            )
        key = query_name.casefold()
        chosen = next(
            (node for node in candidates if str(node["name"]).casefold() == key),
            candidates[0],
        )
        return str(chosen["id"]), str(chosen["name"])

    # ------------------------------------------------------------------- clips
    def get_clips(self, game_name: str, limit: int, period: str = "week",
                  max_pages: int = 20) -> List[TwitchClip]:
        limit = int(limit)
        if limit <= 0:
            return []
        game_id, official_name = self._resolve_game(game_name)
        log.info("Resolved game %r to %r (id=%s)", game_name, official_name,
                 game_id)
        gql_period = PERIOD_MAP.get(period)
        if gql_period is None:
            raise ValueError(f"Unknown period: {period!r}")
        pages = max(1, min(int(max_pages or self._max_pages), self._max_pages))
        found: Dict[str, TwitchClip] = {}
        cursor: Optional[str] = None
        malformed = 0

        for page in range(pages):
            if len(found) >= limit:
                break
            remaining = limit - len(found)
            try:
                data = self._query(GAME_CLIPS_BY_ID, {
                    "id": game_id,
                    "limit": min(100, remaining),
                    "cursor": cursor,
                    "criteria": {"period": gql_period},
                })
            except TwitchAPIError as exc:
                raise type(exc)(
                    f"Twitch page {page + 1} for {official_name!r} failed: {exc}"
                ) from exc

            game = data.get("game")
            if game is None:
                raise TwitchSchemaError(
                    "game is null in clips response; the category may no longer "
                    "exist or the GraphQL schema changed"
                )
            clips_data = game.get("clips")
            if not isinstance(clips_data, dict):
                raise TwitchSchemaError(
                    "game.clips is missing in the response; Twitch GraphQL "
                    "schema may have changed"
                )
            edges = clips_data.get("edges") or []
            if not edges:
                break

            last_cursor = None
            for edge in edges:
                last_cursor = (edge or {}).get("cursor") or last_cursor
                node = (edge or {}).get("node")
                if not node:
                    malformed += 1
                    continue
                try:
                    clip = self._parse_node(node, official_name, game_id)
                except (KeyError, TypeError, ValueError) as exc:
                    malformed += 1
                    log.warning("Skipping malformed Twitch clip node: %s", exc)
                    continue
                if clip.id and clip.id not in found:
                    found[clip.id] = clip

            if len(found) >= limit:
                break
            if not (clips_data.get("pageInfo") or {}).get("hasNextPage"):
                break
            if not last_cursor:
                log.warning("hasNextPage is true but cursor is empty; stopping")
                break
            cursor = last_cursor
            time.sleep(0.35)

        result = list(found.values())[:limit]
        if malformed and not result:
            raise TwitchSchemaError(
                f"All {malformed} clip nodes were malformed; "
                "Twitch GraphQL schema may have changed"
            )
        log.info("GraphQL: fetched %d clip(s) for %r (period=%s)",
                 len(result), official_name, period)
        return result

    @staticmethod
    def _parse_node(node: Dict[str, Any], game_name: str,
                    game_id: str) -> TwitchClip:
        clip_id = str(node["id"]).strip()
        if not clip_id:
            raise ValueError("empty clip id")
        broadcaster = node.get("broadcaster") or {}
        url = str(node.get("url") or "")
        slug = str(node.get("slug") or "")
        if not url and slug:
            url = f"https://clips.twitch.tv/{slug}"
        return TwitchClip(
            id=clip_id,
            title=str(node.get("title") or ""),
            game_name=game_name,
            broadcaster_name=str(broadcaster.get("displayName")
                                 or broadcaster.get("login") or ""),
            url=url,
            created_at=str(node.get("createdAt") or ""),
            duration=float(node.get("durationSeconds") or 0),
            view_count=int(node.get("viewCount") or 0),
            thumbnail_url=str(node.get("thumbnailURL") or ""),
            game_id=game_id,
            slug=slug,
        )


def _retry_after(response: Any, default: float) -> float:
    value = (getattr(response, "headers", {}) or {}).get("Retry-After")
    try:
        wait = float(value)
    except (TypeError, ValueError):
        return max(0.5, min(default, 30.0))
    return max(0.5, min(wait, 30.0))
