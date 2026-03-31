# Web Search Feature

QuantumLeap includes built-in web search capabilities powered by DuckDuckGo. Models can search the web and fetch content from URLs to answer questions with up-to-date information.

## Features

- ✅ **DuckDuckGo Search**: Privacy-focused web search
- ✅ **Content Extraction**: Automatic extraction of main content from web pages
- ✅ **Clean Text**: Removes scripts, styles, navigation, and ads
- ✅ **Async**: Non-blocking search and fetch operations
- ✅ **Configurable**: Control number of results and content length

## API Endpoints

### Search the Web

```bash
curl -X POST http://localhost:11435/api/web/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest AI research 2026",
    "num_results": 3,
    "max_content_length": 3000
  }'
```

**Response**:
```json
{
  "query": "latest AI research 2026",
  "num_results": 3,
  "results": [
    {
      "title": "AI Research Breakthroughs 2026",
      "url": "https://example.com/ai-2026",
      "snippet": "Recent advances in...",
      "content": "Full article content here..."
    }
  ]
}
```

### Fetch URL Content

```bash
curl "http://localhost:11435/api/web/fetch?url=https://example.com&max_length=5000"
```

**Response**:
```json
{
  "url": "https://example.com",
  "title": "Example Domain",
  "content": "Extracted text content...",
  "length": 1234
}
```

## Using with Models

### System Prompt with Web Search Tool

When chatting with a model, you can include a system prompt that enables web search:

```json
{
  "model": "your-model",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant with access to web search. When you need current information, you can search the web using the search_web tool. Format: [SEARCH: query]. You will receive search results with content from top pages."
    },
    {
      "role": "user",
      "content": "What are the latest developments in quantum computing?"
    }
  ]
}
```

### Example Workflow

1. **User asks a question** requiring current information
2. **Model responds** with `[SEARCH: quantum computing 2026]`
3. **Client detects** the search request and calls `/api/web/search`
4. **Client sends results** back to model in a new message
5. **Model synthesizes** answer from search results

### Python Example

```python
import httpx
import json

async def chat_with_web_search(question: str):
    base_url = "http://localhost:11435"
    
    # Initial chat request
    response = await httpx.post(f"{base_url}/api/chat", json={
        "model": "your-model",
        "messages": [
            {
                "role": "system",
                "content": "You have access to web search. Use [SEARCH: query] when you need current info."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    })
    
    assistant_message = response.json()["message"]["content"]
    
    # Check if model wants to search
    if "[SEARCH:" in assistant_message:
        # Extract search query
        query = assistant_message.split("[SEARCH:")[1].split("]")[0].strip()
        
        # Perform search
        search_response = await httpx.post(f"{base_url}/api/web/search", json={
            "query": query,
            "num_results": 3,
            "max_content_length": 2000
        })
        
        search_results = search_response.json()
        
        # Format results for model
        results_text = "\n\n".join([
            f"**{r['title']}**\n{r['url']}\n{r['content'][:500]}..."
            for r in search_results["results"]
        ])
        
        # Send results back to model
        final_response = await httpx.post(f"{base_url}/api/chat", json={
            "model": "your-model",
            "messages": [
                {
                    "role": "system",
                    "content": "You have access to web search."
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": assistant_message
                },
                {
                    "role": "user",
                    "content": f"Search results for '{query}':\n\n{results_text}"
                }
            ]
        })
        
        return final_response.json()["message"]["content"]
    
    return assistant_message
```

## Configuration

### Timeout

Default timeout is 10 seconds per request. Modify in `api/web_search.py`:

```python
web_searcher = WebSearcher(timeout=15)  # 15 seconds
```

### User Agent

The searcher uses a standard browser user agent. Modify in `api/web_search.py` if needed.

### Content Extraction

The module tries these selectors in order:
1. `<article>`
2. `<main>`
3. `.content`, `#content`
4. `.post`, `.entry-content`
5. `<body>` (fallback)

## Privacy

- Uses DuckDuckGo (no tracking)
- No API keys required
- No search history stored
- Direct HTTP requests (no third-party services)

## Limitations

- **Rate limiting**: DuckDuckGo may rate-limit excessive requests
- **Content extraction**: Some sites may not extract cleanly
- **JavaScript**: Dynamic content requiring JS won't be captured
- **Paywalls**: Content behind paywalls won't be accessible

## Troubleshooting

### "No search results found"

- Query may be too specific
- DuckDuckGo may be temporarily unavailable
- Network connectivity issues

### "Failed to fetch URL"

- Site may block automated requests
- Site may be down
- Timeout (increase in config)

### Content is garbled

- Site uses heavy JavaScript (not supported)
- Site has unusual HTML structure
- Try fetching a different source

## Examples

### Current Events

```bash
curl -X POST http://localhost:11435/api/web/search \
  -H "Content-Type: application/json" \
  -d '{"query": "tech news today", "num_results": 5}'
```

### Research Papers

```bash
curl -X POST http://localhost:11435/api/web/search \
  -H "Content-Type: application/json" \
  -d '{"query": "arxiv quantum computing 2026", "num_results": 3}'
```

### Documentation

```bash
curl "http://localhost:11435/api/web/fetch?url=https://docs.python.org/3/library/asyncio.html&max_length=10000"
```

## Security Notes

- Web search is **enabled by default** when using `scripts/start.sh`
- No authentication required (local server)
- Consider adding rate limiting for production
- Validate URLs before fetching in production environments

## Future Enhancements

Potential improvements:
- [ ] Multiple search engines (Google, Bing)
- [ ] Image search
- [ ] PDF extraction
- [ ] Search result caching
- [ ] Rate limiting per client
- [ ] Authentication for public deployments
