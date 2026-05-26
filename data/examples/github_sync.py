import requests

GITHUB_TOKEN = "ghp_R7kLmNpQvXwYzA3bC5dE8fG2hJ4iK6oP"
GITHUB_ORG   = "acme-corp"

headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

def list_repos() -> list[dict]:
    resp = requests.get(
        f"https://api.github.com/orgs/{GITHUB_ORG}/repos",
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()

def create_repo(name: str, private: bool = True) -> dict:
    resp = requests.post(
        f"https://api.github.com/orgs/{GITHUB_ORG}/repos",
        headers=headers,
        json={"name": name, "private": private},
    )
    resp.raise_for_status()
    return resp.json()
