from datetime import datetime, timedelta, timezone

import pytest

from src.twitch.errors import (TwitchGameNotFound, TwitchProviderError,
                              TwitchSchemaError)
from src.twitch.graphql_provider import TwitchGraphQLProvider
from src.twitch.helix_provider import TwitchHelixProvider


class FakeResponse:
    def __init__(self, status=200, payload=None, text="", headers=None):
        self.status_code = status
        self._payload = payload
        self.text = text or (str(payload) if payload is not None else "")
        self.headers = headers or {}

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, posts=None, gets=None):
        self.headers = {}
        self.posts = list(posts or [])
        self.gets = list(gets or [])
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if not self.posts:
            raise AssertionError(f"Unexpected POST {url}")
        return self.posts.pop(0)

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if not self.gets:
            raise AssertionError(f"Unexpected GET {url}")
        return self.gets.pop(0)


def gql(payload):
    return FakeResponse(200, {"data": payload})


def search_payload(game_id="1", name="Game"):
    return {"searchCategories": {"edges": [{"node": {"id": game_id,
                                                     "name": name}}]}}


def clip_node(clip_id, cursor="cursor"):
    return {
        "cursor": cursor,
        "node": {
            "id": clip_id, "slug": f"slug-{clip_id}", "title": f"title-{clip_id}",
            "url": f"https://twitch.tv/clip/{clip_id}", "viewCount": 10,
            "createdAt": "2026-01-01T00:00:00Z", "durationSeconds": 12,
            "broadcaster": {"displayName": "streamer", "login": "streamer"},
        },
    }


def clips_payload(edges, has_next=False):
    return {
        "game": {
            "id": "1", "name": "Game",
            "clips": {
                "pageInfo": {"hasNextPage": has_next},
                "edges": edges,
            },
        }
    }


def graphql_provider(monkeypatch, posts):
    monkeypatch.setattr("src.twitch.graphql_provider.time.sleep", lambda _: None)
    provider = TwitchGraphQLProvider(timeout=5)
    provider._session = FakeSession(posts=posts)
    return provider


def test_graphql_pagination(monkeypatch):
    provider = graphql_provider(monkeypatch, [
        gql(search_payload()),
        gql(clips_payload([clip_node("a", "c1"), clip_node("b", "c2")],
                          has_next=True)),
        gql(clips_payload([clip_node("c", "c3")], has_next=False)),
    ])
    clips = provider.get_clips("Game", 3, "week")
    assert [clip.id for clip in clips] == ["a", "b", "c"]
    assert clips[0].game_id == "1"
    assert clips[0].url.endswith("/a")
    assert len(provider._session.calls) == 3
    second_page_vars = provider._session.calls[2][2]["json"]["variables"]
    assert second_page_vars["cursor"] == "c2"
    assert second_page_vars["criteria"]["period"] == "LAST_WEEK"


def test_graphql_deduplicates(monkeypatch):
    provider = graphql_provider(monkeypatch, [
        gql(search_payload()),
        gql(clips_payload([clip_node("a"), clip_node("a")], has_next=False)),
    ])
    clips = provider.get_clips("Game", 5, "week")
    assert [clip.id for clip in clips] == ["a"]


def test_graphql_game_not_found(monkeypatch):
    provider = graphql_provider(monkeypatch, [
        gql({"searchCategories": {"edges": []}}),
    ])
    with pytest.raises(TwitchGameNotFound):
        provider.get_clips("Unknown Game", 1)


def test_graphql_schema_change(monkeypatch):
    provider = graphql_provider(monkeypatch, [
        gql(search_payload()),
        gql({"game": None}),
    ])
    with pytest.raises(TwitchSchemaError):
        provider.get_clips("Game", 1)


def test_graphql_empty_game_is_not_an_error(monkeypatch):
    provider = graphql_provider(monkeypatch, [
        gql(search_payload()),
        gql(clips_payload([], has_next=False)),
    ])
    assert provider.get_clips("Game", 5) == []


def test_graphql_rate_limit_retry(monkeypatch):
    provider = graphql_provider(monkeypatch, [
        FakeResponse(429, {"error": "rate"}, headers={"Retry-After": "0"}),
        gql(search_payload()),
        gql(clips_payload([clip_node("a")], has_next=False)),
    ])
    clips = provider.get_clips("Game", 1)
    assert [clip.id for clip in clips] == ["a"]


def test_graphql_service_error_is_retried(monkeypatch):
    provider = graphql_provider(monkeypatch, [
        gql(search_payload()),
        FakeResponse(200, {"errors": [{"message": "service error",
                                       "path": ["game", "clips"]}]}),
        gql(clips_payload([clip_node("a")], has_next=False)),
    ])
    clips = provider.get_clips("Game", 1)
    assert [clip.id for clip in clips] == ["a"]


def test_graphql_persistent_service_error_is_provider_error(monkeypatch):
    service_error = FakeResponse(200, {"errors": [{
        "message": "service error", "path": ["game", "clips"]}]})
    provider = graphql_provider(monkeypatch, [
        gql(search_payload()),
        service_error, service_error, service_error,
    ])
    with pytest.raises(TwitchProviderError):
        provider.get_clips("Game", 1)



def test_graphql_page_error_is_not_silently_partial(monkeypatch):
    provider = graphql_provider(monkeypatch, [
        gql(search_payload()),
        FakeResponse(500, {"error": "boom"}),
        FakeResponse(500, {"error": "boom"}),
        FakeResponse(500, {"error": "boom"}),
    ])
    with pytest.raises(TwitchProviderError):
        provider.get_clips("Game", 5)


def test_graphql_health_check(monkeypatch):
    provider = graphql_provider(monkeypatch, [
        gql(search_payload("32399", "Counter-Strike")),
        gql(clips_payload([], has_next=False)),
    ])
    health = provider.check_health()
    assert health.ok is True
    assert "OK" in health.detail


def test_graphql_health_check_auth_failure(monkeypatch):
    provider = graphql_provider(monkeypatch, [
        FakeResponse(401, {"error": "bad client id"}),
    ])
    health = provider.check_health()
    assert health.ok is False
    assert "401" in health.detail or "rejected" in health.detail


# --------------------------------------------------------------------- Helix
def token_response(token="token"):
    return FakeResponse(200, {"access_token": token, "expires_in": 3600})


def helix_provider(monkeypatch, posts, gets):
    monkeypatch.setattr("src.twitch.helix_provider.time.sleep", lambda _: None)
    provider = TwitchHelixProvider("id", "secret", timeout=5)
    provider._session = FakeSession(posts=posts, gets=gets)
    return provider


def test_helix_pagination_and_period(monkeypatch):
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    old = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat().replace(
        "+00:00", "Z")
    provider = helix_provider(
        monkeypatch,
        posts=[token_response()],
        gets=[
            FakeResponse(200, {"data": [{"id": "game-1",
                                         "name": "Counter-Strike"}]}),
            FakeResponse(200, {"data": [
                {"id": "old", "title": "old", "url": "u", "created_at": old,
                 "duration": 5, "view_count": 1, "broadcaster_name": "b"},
                {"id": "new", "title": "new", "url": "u2", "created_at": now,
                 "duration": 5, "view_count": 2, "broadcaster_name": "b"},
            ], "pagination": {"cursor": "next"}}),
        ],
    )
    clips = provider.get_clips("Counter-Strike 2", 1, "week")
    assert [clip.id for clip in clips] == ["new"]
    assert clips[0].game_id == "game-1"


def test_helix_refreshes_token_on_401(monkeypatch):
    provider = helix_provider(
        monkeypatch,
        posts=[token_response("t1"), token_response("t2")],
        gets=[
            FakeResponse(401, {"error": "expired"}),
            FakeResponse(200, {"data": [{"id": "game-1",
                                         "name": "Game"}]}),
        ],
    )
    game_id, name = provider._resolve_game("Game")
    assert (game_id, name) == ("game-1", "Game")
    assert provider._token == "t2"


def test_helix_game_search_fallback(monkeypatch):
    provider = helix_provider(
        monkeypatch,
        posts=[token_response()],
        gets=[
            FakeResponse(200, {"data": []}),
            FakeResponse(200, {"data": [{"id": "cat-1",
                                         "name": "Counter-Strike"}]}),
        ],
    )
    assert provider._resolve_game("Counter-Strike 2") == (
        "cat-1", "Counter-Strike")


def test_helix_health(monkeypatch):
    provider = helix_provider(
        monkeypatch,
        posts=[token_response()],
        gets=[FakeResponse(200, {"data": [{"id": "1",
                                           "name": "Counter-Strike"}]})],
    )
    health = provider.check_health()
    assert health.ok is True
