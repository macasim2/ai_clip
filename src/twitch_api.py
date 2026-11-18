import requests

PUBLIC_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
GQL_URL = "https://gql.twitch.tv/gql"


class TwitchAPIError(Exception):
    pass


class TwitchAPI:
    def __init__(self, client_id: str = None, client_secret: str = None):
        pass

    def _post(self, payload: dict):
        headers = {
            "Client-ID": PUBLIC_CLIENT_ID,
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.post(GQL_URL, json=payload, headers=headers, timeout=30)
        if not response.ok:
            raise TwitchAPIError(
                f"Ошибка GraphQL Twitch: {response.status_code} {response.text}"
            )
        return response.json()

    def get_game_id(self, game_name: str):
        query = """
        query($query: String!) {
          searchCategories(query: $query, first: 1) {
            edges {
              node {
                id
                name
              }
            }
          }
        }
        """
        data = self._post({"query": query, "variables": {"query": game_name}})
        edges = data.get("data", {}).get("searchCategories", {}).get("edges", [])
        if not edges:
            return None
        
        game = edges[0]["node"]
        print(f"-> Найдена игра в Twitch: {game['name']} (ID: {game['id']})")
        return game["id"]

    def get_clips(self, game_id: str, first: int = 20):
        # ИЗМЕНЕНИЕ: period: ALL_TIME вместо LAST_MONTH
        query = """
        query($gameId: ID!, $first: Int!) {
          game(id: $gameId) {
            clips(first: $first, criteria: { period: ALL_TIME }) {
              edges {
                node {
                  id
                  slug
                  title
                  viewCount
                  createdAt
                  durationSeconds
                  broadcaster { displayName }
                  thumbnailURL(width: 320, height: 180)
                }
              }
            }
          }
        }
        """
        variables = {"gameId": game_id, "first": max(1, min(first, 100))}
        data = self._post({"query": query, "variables": variables})

        game = data.get("data", {}).get("game")
        if not game:
            raise TwitchAPIError(f"Игра не найдена по ID: {game_id}")

        # ИСПРАВЛЕНИЕ ОШИБКИ: Безопасная проверка на null
        clips_data = game.get("clips")
        if not clips_data:
            return [] # Если клипов нет, возвращаем пустой список, а не падаем

        edges = clips_data.get("edges", [])

        clips = []
        for edge in edges:
            node = edge["node"]
            slug = node.get("slug")
            url = f"https://clips.twitch.tv/{slug}" if slug else ""
            clips.append({
                "id": node["id"],
                "game_id": game_id,
                "broadcaster_name": node["broadcaster"]["displayName"],
                "title": node["title"],
                "url": url,
                "thumbnail_url": node["thumbnailURL"],
                "view_count": node["viewCount"],
                "created_at": node["createdAt"],
                "duration": node["durationSeconds"],
            })
        return clips
