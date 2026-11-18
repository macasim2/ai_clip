import requests

url = "https://gql.twitch.tv/gql"
headers = {"Client-ID": "kimne78kx3ncx6brgo4mv6wki5h1ko"}
query = """
query SearchCategory($query: String!) {
  searchCategories(query: $query, first: 1) {
    edges {
      node { id name }
    }
  }
}
"""
variables = {"query": "Counter-Strike 2"}

r = requests.post(url, json={"query": query, "variables": variables}, headers=headers)
print(r.json())