"""Temporary Twitch web GraphQL provider (no 2FA required)."""
import time
from typing import Any, Dict, List, Optional

import requests

from ..logging_setup import get_logger
from ..models import TwitchClip
from ..utils import normalize_game_name
from .base import TwitchProvider

log = get_logger(__name__)

GQL_URL = "https://gql.twitch.tv/gql"
DEFAULT_WEB_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"

# 1. Запрос для поиска ID игры по названию
SEARCH_GAME = """query SearchCategory($query: String!) {
  searchCategories(query: $query, first: 1) {
    edges {
      node {
        id
        name
      }
    }
  }
}"""

# 2. Запрос клипов по ID игры
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

PERIOD_MAP = {"day": "LAST_DAY", "week": "LAST_WEEK",
              "month": "LAST_MONTH", "all": "ALL_TIME"}


class TwitchProviderError(RuntimeError):
    pass


class TwitchGraphQLProvider(TwitchProvider):
    def __init__(self, client_id: str = DEFAULT_WEB_CLIENT_ID, timeout: float = 20.0):
        self._client_id = client_id or DEFAULT_WEB_CLIENT_ID
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Client-ID": self._client_id,
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        })

    @property
    def name(self) -> str:
        return "graphql"

    def health_check(self) -> bool:
        try:
            r = self._session.post(
                GQL_URL, json={"query": "{ __typename }"}, timeout=self._timeout,
            )
            return r.status_code == 200
        except Exception:
            return False

    def _query(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        last_err: Optional[str] = None
        for attempt in range(3):
            try:
                r = self._session.post(
                    GQL_URL,
                    json={"query": query, "variables": variables},
                    timeout=self._timeout,
                )
            except requests.RequestException as e:
                last_err = f"network: {e}"
                time.sleep(1.5 ** attempt)
                continue
            if r.status_code == 200:
                try:
                    payload = r.json()
                except ValueError:
                    last_err = "invalid JSON in response"
                    continue
                if "errors" in payload:
                    raise TwitchProviderError(f"GraphQL errors: {payload['errors']!r}")
                return payload.get("data") or {}
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After", 2 ** attempt))
                time.sleep(wait)
                continue
            raise TwitchProviderError(f"Twitch HTTP {r.status_code}: {r.text[:200]}")
        raise TwitchProviderError(f"Twitch GraphQL: exhausted retries ({last_err})")

    def get_clips(self, game_name: str, limit: int, period: str = "week",
                  max_pages: int = 10) -> List[TwitchClip]:
        game_name = normalize_game_name(game_name)
        
        # 1. Ищем ID игры
        search_data = self._query(SEARCH_GAME, {"query": game_name})
        edges = search_data.get("searchCategories", {}).get("edges", [])
        if not edges:
            log.warning("Game not found via search: %s", game_name)
            return []
        
        game_id = edges[0]["node"]["id"]
        official_game_name = edges[0]["node"]["name"]
        log.info("Resolved game '%s' to ID %s", official_game_name, game_id)

        # 2. Получаем клипы по ID
        gql_period = PERIOD_MAP.get(period, "LAST_WEEK")
        clips: List[TwitchClip] = []
        cursor = None

        for page in range(max_pages):
            if len(clips) >= limit:
                break
            try:
                data = self._query(GAME_CLIPS_BY_ID, {
                    "id": game_id,
                    "limit": min(100, limit - len(clips)),
                    "cursor": cursor,
                    "criteria": {"period": gql_period},
                })
            except TwitchProviderError as e:
                log.error("Page %d failed: %s", page, e)
                break

            g = data.get("game")
            if not g or not g.get("clips"):
                break

            clip_data = g.get("clips")
            edges = clip_data.get("edges", [])
            if not edges:
                break

            for edge in edges:
                n = edge["node"]
                try:
                    clips.append(TwitchClip(
                        id=n["id"],
                        title=n.get("title") or "",
                        game_name=official_game_name,
                        broadcaster_name=(n.get("broadcaster") or {}).get("displayName") or "",
                        url=n.get("url") or f"https://clips.twitch.tv/{n.get('slug')}",
                        created_at=n.get("createdAt") or "",
                        duration=float(n.get("durationSeconds") or 0),
                        view_count=int(n.get("viewCount") or 0),
                        thumbnail_url=n.get("thumbnailURL") or "",
                    ))
                except Exception as e:
                    log.warning("Skipping malformed clip node: %s", e)

            if not clip_data.get("pageInfo", {}).get("hasNextPage"):
                break
            cursor = edges[-1]["cursor"]
            time.sleep(0.4)
            
        log.info("GraphQL: fetched %d clips for '%s'", len(clips), game_name)
        return clips[:limit]
