"""
Web Search Module for QuantumLeap
Provides web search and content fetching capabilities
"""

import asyncio
import re
from typing import Optional
from urllib.parse import quote_plus, urlparse

import httpx
from bs4 import BeautifulSoup


class WebSearcher:
    """Web search and content fetching"""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    
    async def search_duckduckgo(self, query: str, max_results: int = 5) -> list[dict]:
        """Search DuckDuckGo and return results"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers, follow_redirects=True) as client:
                # DuckDuckGo HTML search
                url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
                response = await client.get(url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, "html.parser")
                results = []
                
                # Parse search results
                for result in soup.select(".result")[:max_results]:
                    title_elem = result.select_one(".result__title")
                    snippet_elem = result.select_one(".result__snippet")
                    url_elem = result.select_one(".result__url")
                    
                    if title_elem and url_elem:
                        title = title_elem.get_text(strip=True)
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        result_url = url_elem.get("href", "")
                        
                        # Clean up DuckDuckGo redirect URL
                        if result_url.startswith("//duckduckgo.com/l/?"):
                            continue
                        
                        results.append({
                            "title": title,
                            "url": result_url,
                            "snippet": snippet
                        })
                
                return results
        except Exception as e:
            print(f"[WebSearch] DuckDuckGo search failed: {e}")
            return []
    
    async def fetch_url(self, url: str, max_length: int = 5000) -> dict:
        """Fetch and extract content from a URL"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Get title
                title = soup.title.string if soup.title else ""
                
                # Extract main content
                # Try common content containers
                content = None
                for selector in ["article", "main", ".content", "#content", ".post", ".entry-content"]:
                    content_elem = soup.select_one(selector)
                    if content_elem:
                        content = content_elem.get_text(separator="\n", strip=True)
                        break
                
                # Fallback to body
                if not content:
                    body = soup.body
                    if body:
                        content = body.get_text(separator="\n", strip=True)
                    else:
                        content = soup.get_text(separator="\n", strip=True)
                
                # Clean up whitespace
                content = re.sub(r'\n\s*\n+', '\n\n', content)
                content = content[:max_length]
                
                return {
                    "url": url,
                    "title": title,
                    "content": content,
                    "length": len(content)
                }
        except Exception as e:
            return {
                "url": url,
                "error": str(e),
                "content": ""
            }
    
    async def search_and_fetch(self, query: str, num_results: int = 3, max_content_length: int = 3000) -> dict:
        """Search and fetch content from top results"""
        # Search
        results = await self.search_duckduckgo(query, max_results=num_results)
        
        if not results:
            return {
                "query": query,
                "results": [],
                "error": "No search results found"
            }
        
        # Fetch content from top results
        fetch_tasks = []
        for result in results[:num_results]:
            fetch_tasks.append(self.fetch_url(result["url"], max_length=max_content_length))
        
        contents = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # Combine results with content
        for i, result in enumerate(results[:num_results]):
            if i < len(contents) and not isinstance(contents[i], Exception):
                result["content"] = contents[i].get("content", "")
                result["fetch_error"] = contents[i].get("error")
        
        return {
            "query": query,
            "results": results[:num_results],
            "num_results": len(results)
        }


# Global instance
web_searcher = WebSearcher()
