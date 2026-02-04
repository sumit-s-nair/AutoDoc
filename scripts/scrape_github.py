"""
AutoDoc GitHub Scraper
======================
Scrapes high-quality README.md files and repository structures from GitHub.

Features:
- Queries repos by stars, activity, and language
- Extracts README content and file structure
- Handles rate limiting with exponential backoff
- Supports GitHub Personal Access Token for higher limits

Usage:
    python scrape_github.py [--stars 500] [--limit 1000] [--output data/raw/github]
    
Environment:
    GITHUB_TOKEN: Personal access token for higher rate limits
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Generator

import requests
from tqdm import tqdm


class GitHubScraper:
    """Scrapes GitHub repositories for documentation training data."""
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the scraper.
        
        Args:
            token: GitHub Personal Access Token (optional, but recommended)
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.session = requests.Session()
        
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
            self.rate_limit = 5000
            print("✓ Using authenticated requests (5000 req/hour)")
        else:
            self.rate_limit = 60
            print("⚠ No token provided (60 req/hour). Set GITHUB_TOKEN for faster scraping.")
        
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        self.request_count = 0
        self.last_request_time = None
    
    def _rate_limit_wait(self):
        """Handle rate limiting with exponential backoff."""
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            min_interval = 3600 / self.rate_limit  # Spread requests over an hour
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """
        Make a rate-limited request to GitHub API.
        
        Args:
            endpoint: API endpoint (e.g., /search/repositories)
            params: Query parameters
        
        Returns:
            JSON response or None on failure
        """
        self._rate_limit_wait()
        
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 403:
                    # Rate limited - wait and retry
                    reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                    wait_time = max(reset_time - time.time(), 60)
                    print(f"\n⏳ Rate limited. Waiting {wait_time:.0f}s...")
                    time.sleep(wait_time)
                elif response.status_code == 404:
                    return None
                else:
                    print(f"\n⚠ Request failed: {response.status_code}")
                    time.sleep(2 ** attempt)
                    
            except requests.RequestException as e:
                print(f"\n⚠ Request error: {e}")
                time.sleep(2 ** attempt)
        
        return None
    
    def search_repos(
        self,
        min_stars: int = 500,
        languages: list = None,
        limit: int = 100
    ) -> Generator[dict, None, None]:
        """
        Search for repositories matching criteria.
        
        Args:
            min_stars: Minimum star count
            languages: List of programming languages
            limit: Maximum repos to return
        
        Yields:
            Repository metadata
        """
        languages = languages or ["python", "javascript", "typescript", "go", "rust"]
        
        for language in languages:
            query = f"stars:>={min_stars} language:{language} pushed:>{datetime.now() - timedelta(days=730):%Y-%m-%d}"
            
            page = 1
            repos_found = 0
            
            while repos_found < limit // len(languages):
                params = {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 30,
                    "page": page
                }
                
                result = self._request("/search/repositories", params)
                
                if not result or "items" not in result:
                    break
                
                for repo in result["items"]:
                    yield repo
                    repos_found += 1
                    
                    if repos_found >= limit // len(languages):
                        break
                
                if len(result["items"]) < 30:
                    break
                    
                page += 1
    
    def get_readme(self, owner: str, repo: str) -> Optional[str]:
        """
        Get README content for a repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
        
        Returns:
            README content or None
        """
        result = self._request(f"/repos/{owner}/{repo}/readme")
        
        if result and "content" in result:
            import base64
            try:
                content = base64.b64decode(result["content"]).decode("utf-8")
                return content
            except Exception:
                return None
        
        return None
    
    def get_file_tree(self, owner: str, repo: str, max_depth: int = 2) -> Optional[list]:
        """
        Get file tree for a repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            max_depth: Maximum directory depth
        
        Returns:
            List of file paths or None
        """
        result = self._request(f"/repos/{owner}/{repo}/git/trees/HEAD", {"recursive": "1"})
        
        if result and "tree" in result:
            files = []
            for item in result["tree"]:
                # Filter by depth
                depth = item["path"].count("/")
                if depth <= max_depth:
                    files.append({
                        "path": item["path"],
                        "type": item["type"],
                        "size": item.get("size", 0)
                    })
            return files
        
        return None
    
    def scrape_repo(self, repo: dict) -> Optional[dict]:
        """
        Scrape all relevant data from a repository.
        
        Args:
            repo: Repository metadata from search
        
        Returns:
            Scraped data or None
        """
        owner = repo["owner"]["login"]
        name = repo["name"]
        
        readme = self.get_readme(owner, name)
        
        if not readme or len(readme) < 500:
            return None
        
        file_tree = self.get_file_tree(owner, name)
        
        return {
            "repo_name": f"{owner}/{name}",
            "url": repo["html_url"],
            "stars": repo["stargazers_count"],
            "language": repo["language"],
            "description": repo.get("description", ""),
            "readme": readme,
            "file_tree": file_tree,
            "scraped_at": datetime.now().isoformat()
        }


def filter_quality(data: dict) -> bool:
    """
    Filter out low-quality entries.
    
    Args:
        data: Scraped repository data
    
    Returns:
        True if entry passes quality checks
    """
    readme = data["readme"]
    
    # Check for template-generated content
    template_markers = [
        "<!-- PROJECT SHIELDS -->",
        "{{cookiecutter",
        "REPLACE_ME",
        "TODO: Add",
    ]
    
    for marker in template_markers:
        if marker in readme:
            return False
    
    # Check for minimum content
    sections = readme.count("\n## ") + readme.count("\n# ")
    if sections < 2:
        return False
    
    # Check for code examples
    has_code = "```" in readme
    
    return has_code


def main():
    parser = argparse.ArgumentParser(
        description="Scrape GitHub repositories for documentation data"
    )
    parser.add_argument(
        "--stars",
        type=int,
        default=500,
        help="Minimum star count for repositories"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of repositories to scrape"
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="python,javascript,typescript",
        help="Comma-separated list of languages"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/github",
        help="Output directory"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="GitHub Personal Access Token"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    languages = [lang.strip() for lang in args.languages.split(",")]
    
    print("=" * 50)
    print("AutoDoc GitHub Scraper")
    print("=" * 50)
    print(f"Min stars: {args.stars}")
    print(f"Limit: {args.limit}")
    print(f"Languages: {languages}")
    print(f"Output: {output_dir.absolute()}")
    
    scraper = GitHubScraper(token=args.token)
    
    # Scrape repositories
    scraped_data = []
    failed = 0
    filtered = 0
    
    print("\n📦 Searching and scraping repositories...")
    
    repos = list(scraper.search_repos(
        min_stars=args.stars,
        languages=languages,
        limit=args.limit
    ))
    
    for repo in tqdm(repos, desc="Scraping"):
        data = scraper.scrape_repo(repo)
        
        if data is None:
            failed += 1
            continue
        
        if not filter_quality(data):
            filtered += 1
            continue
        
        scraped_data.append(data)
    
    # Save results
    output_file = output_dir / f"repos_{datetime.now():%Y%m%d_%H%M%S}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("\n" + "=" * 50)
    print("Scraping Summary")
    print("=" * 50)
    print(f"  Repositories found: {len(repos)}")
    print(f"  Successfully scraped: {len(scraped_data)}")
    print(f"  Failed to scrape: {failed}")
    print(f"  Filtered (low quality): {filtered}")
    print(f"\n  Output: {output_file}")
    
    print("\n✅ Scraping complete!")


if __name__ == "__main__":
    main()
