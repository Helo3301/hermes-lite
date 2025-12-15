"""
General-Purpose Web Crawler for HERMES-Lite

A scopeable web crawler that can crawl any whitelisted domain with focus
on specific subjects/topics. Designed to be flexible for different knowledge
domains (wikis, documentation sites, etc.)

Features:
- Subject scoping via URL patterns and content keywords
- Link following with depth control
- Rate limiting to be respectful
- HTML to markdown conversion
- Integration with Hermes ingestion pipeline
"""
import re
import time
import logging
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse, urljoin
from collections import deque

import httpx
from bs4 import BeautifulSoup

from .crawler import validate_url_against_whitelist, DomainNotWhitelistedError

logger = logging.getLogger(__name__)

# Rate limiting between requests (seconds)
DEFAULT_RATE_LIMIT = 2.0

# Maximum pages to crawl in one session
DEFAULT_MAX_PAGES = 100

# Maximum crawl depth from seed URLs
DEFAULT_MAX_DEPTH = 3


@dataclass
class SubjectScope:
    """
    Defines the scope for crawling a specific subject within a domain.

    Attributes:
        name: Human-readable name for the subject (e.g., "redstone")
        url_patterns: Regex patterns that URLs must match to be crawled
        url_exclude_patterns: Regex patterns for URLs to skip
        content_keywords: Keywords that should appear in page content
        seed_urls: Starting URLs for the crawl
        collection_name: Hermes collection to store documents in
    """
    name: str
    url_patterns: list[str] = field(default_factory=list)
    url_exclude_patterns: list[str] = field(default_factory=list)
    content_keywords: list[str] = field(default_factory=list)
    seed_urls: list[str] = field(default_factory=list)
    collection_name: str = "web-crawl"

    def matches_url(self, url: str) -> bool:
        """Check if a URL matches this subject's scope."""
        # Check exclusions first
        for pattern in self.url_exclude_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False

        # If no inclusion patterns, accept all non-excluded URLs
        if not self.url_patterns:
            return True

        # Check if URL matches any inclusion pattern
        for pattern in self.url_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        return False

    def matches_content(self, text: str) -> bool:
        """Check if content is relevant to this subject."""
        if not self.content_keywords:
            return True

        text_lower = text.lower()
        # Require at least one keyword to be present
        return any(kw.lower() in text_lower for kw in self.content_keywords)


@dataclass
class CrawledPage:
    """Represents a crawled web page."""
    url: str
    title: str
    markdown: str
    links: list[str]
    crawled_at: datetime
    depth: int


# Pre-defined subject scopes for common use cases
MINECRAFT_REDSTONE_SCOPE = SubjectScope(
    name="minecraft-redstone",
    url_patterns=[
        r'/w/Redstone',
        r'/w/.*[Rr]edstone.*',
        r'/w/Lever',
        r'/w/Button',
        r'/w/Pressure_Plate',
        r'/w/Tripwire',
        r'/w/Observer',
        r'/w/Piston',
        r'/w/Sticky_Piston',
        r'/w/Dropper',
        r'/w/Dispenser',
        r'/w/Hopper',
        r'/w/Comparator',
        r'/w/Repeater',
        r'/w/Target',
        r'/w/Daylight_Detector',
        r'/w/Trapped_Chest',
        r'/w/TNT',
        r'/w/Note_Block',
        r'/w/Logic_circuit',
        r'/w/Clock_circuit',
        r'/w/Memory_circuit',
        r'/w/Pulse_circuit',
        r'/w/Transmission_circuit',
        r'/w/Mechanics/',
        r'/w/Tutorials/.*[Rr]edstone',
        r'/w/Tutorials/.*[Cc]ircuit',
        r'/w/Tutorials/.*[Ll]ogic',
    ],
    url_exclude_patterns=[
        r'/w/File:',
        r'/w/Category:',
        r'/w/Template:',
        r'/w/User:',
        r'/w/Talk:',
        r'/w/Special:',
        r'\?',  # Query parameters (usually non-content pages)
        r'#',   # Anchors
        r'/w/.*\(.*Edition\)',  # Edition-specific variants (we want main articles)
    ],
    content_keywords=[
        'redstone', 'circuit', 'signal', 'power', 'lever', 'torch',
        'repeater', 'comparator', 'piston', 'dust', 'wire', 'pulse',
        'clock', 'logic', 'gate', 'AND', 'OR', 'NOT', 'XOR', 'latch',
        'flip-flop', 'memory', 'transmission', 'mechanism'
    ],
    seed_urls=[
        'https://minecraft.wiki/w/Redstone',
        'https://minecraft.wiki/w/Redstone_Dust',
        'https://minecraft.wiki/w/Redstone_Torch',
        'https://minecraft.wiki/w/Redstone_Repeater',
        'https://minecraft.wiki/w/Redstone_Comparator',
        'https://minecraft.wiki/w/Redstone_mechanics',
        'https://minecraft.wiki/w/Redstone_components',
        'https://minecraft.wiki/w/Logic_circuit',
        'https://minecraft.wiki/w/Clock_circuit',
        'https://minecraft.wiki/w/Memory_circuit',
        'https://minecraft.wiki/w/Tutorials/Redstone_computers',
    ],
    collection_name="minecraft-redstone"
)


def html_to_markdown(html: str, base_url: str = "") -> tuple[str, str, list[str]]:
    """
    Convert HTML to markdown and extract links.

    Returns:
        Tuple of (title, markdown_content, list_of_links)
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Extract title
    title_tag = soup.find('title')
    title = title_tag.get_text().strip() if title_tag else "Untitled"
    # Clean up wiki-style titles
    title = re.sub(r'\s*[-–|].*[Ww]iki.*$', '', title).strip()

    # Remove unwanted elements
    for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header',
                                   'aside', 'form', 'iframe', 'noscript']):
        element.decompose()

    # Remove wiki-specific noise
    for selector in ['.mw-editsection', '.navbox', '.infobox', '.sidebar',
                     '.mbox', '.ambox', '.reference', '.reflist', '.toc',
                     '#mw-navigation', '#footer', '.noprint', '.mw-indicators']:
        for element in soup.select(selector):
            element.decompose()

    # Find main content area (wiki-specific)
    content = soup.find('div', {'id': 'mw-content-text'})
    if not content:
        content = soup.find('main') or soup.find('article') or soup.body or soup

    # Extract links before converting to text
    links = []
    for a in content.find_all('a', href=True):
        href = a['href']
        # Convert relative URLs to absolute
        if href.startswith('/'):
            href = urljoin(base_url, href)
        if href.startswith('http'):
            links.append(href)

    # Convert to markdown-like text
    markdown_lines = []

    for element in content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'pre', 'code', 'blockquote']):
        text = element.get_text().strip()
        if not text:
            continue

        if element.name == 'h1':
            markdown_lines.append(f"\n# {text}\n")
        elif element.name == 'h2':
            markdown_lines.append(f"\n## {text}\n")
        elif element.name == 'h3':
            markdown_lines.append(f"\n### {text}\n")
        elif element.name == 'h4':
            markdown_lines.append(f"\n#### {text}\n")
        elif element.name in ['h5', 'h6']:
            markdown_lines.append(f"\n##### {text}\n")
        elif element.name == 'li':
            markdown_lines.append(f"- {text}")
        elif element.name == 'pre' or element.name == 'code':
            markdown_lines.append(f"\n```\n{text}\n```\n")
        elif element.name == 'blockquote':
            markdown_lines.append(f"\n> {text}\n")
        else:
            markdown_lines.append(f"\n{text}\n")

    markdown = '\n'.join(markdown_lines)

    # Clean up excessive whitespace
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)
    markdown = markdown.strip()

    return title, markdown, links


class WebCrawler:
    """
    General-purpose web crawler with subject scoping.

    Can crawl any whitelisted domain while focusing on specific topics
    defined by URL patterns and content keywords.
    """

    def __init__(
        self,
        db,
        embed_client,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_depth: int = DEFAULT_MAX_DEPTH
    ):
        """
        Initialize the web crawler.

        Args:
            db: Database instance with whitelist and document methods
            embed_client: Embedding client for vectorizing content
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            rate_limit: Seconds between requests
            max_pages: Maximum pages to crawl
            max_depth: Maximum link depth from seed URLs
        """
        self.db = db
        self.embed_client = embed_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rate_limit = rate_limit
        self.max_pages = max_pages
        self.max_depth = max_depth

        self.client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={
                'User-Agent': 'HERMES-Lite/1.0 (Knowledge Base Crawler; +https://github.com/hermes-lite)'
            }
        )
        self.last_request_time = 0

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch a page from a whitelisted URL.

        Args:
            url: URL to fetch

        Returns:
            HTML content or None if failed
        """
        # Validate against whitelist
        try:
            validate_url_against_whitelist(url, self.db)
        except DomainNotWhitelistedError as e:
            logger.warning(f"Skipping non-whitelisted URL: {url}")
            return None

        self._rate_limit_wait()

        try:
            response = self.client.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def crawl_subject(
        self,
        scope: SubjectScope,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> list[CrawledPage]:
        """
        Crawl pages for a specific subject scope.

        Args:
            scope: SubjectScope defining what to crawl
            progress_callback: Optional callback(current, total, url) for progress updates

        Returns:
            List of successfully crawled pages
        """
        crawled_pages = []
        visited_urls = set()

        # Queue: (url, depth)
        queue = deque((url, 0) for url in scope.seed_urls)

        pages_crawled = 0

        while queue and pages_crawled < self.max_pages:
            url, depth = queue.popleft()

            # Normalize URL
            url = url.split('#')[0]  # Remove anchors

            if url in visited_urls:
                continue
            visited_urls.add(url)

            # Check if URL matches scope
            if not scope.matches_url(url):
                logger.debug(f"URL doesn't match scope: {url}")
                continue

            # Progress update
            if progress_callback:
                progress_callback(pages_crawled, self.max_pages, url)

            logger.info(f"Crawling [{pages_crawled + 1}/{self.max_pages}] depth={depth}: {url}")

            # Fetch page
            html = self.fetch_page(url)
            if not html:
                continue

            # Convert to markdown
            title, markdown, links = html_to_markdown(html, url)

            # Check content relevance
            if not scope.matches_content(markdown):
                logger.debug(f"Content doesn't match scope keywords: {url}")
                continue

            # Store crawled page
            page = CrawledPage(
                url=url,
                title=title,
                markdown=markdown,
                links=links,
                crawled_at=datetime.now(),
                depth=depth
            )
            crawled_pages.append(page)
            pages_crawled += 1

            # Add links to queue if within depth limit
            if depth < self.max_depth:
                for link in links:
                    if link not in visited_urls and scope.matches_url(link):
                        queue.append((link, depth + 1))

        logger.info(f"Crawled {len(crawled_pages)} pages for subject '{scope.name}'")
        return crawled_pages

    def ingest_pages(
        self,
        pages: list[CrawledPage],
        collection_name: str,
        source_name: str = "web"
    ) -> list[int]:
        """
        Ingest crawled pages into Hermes.

        Args:
            pages: List of CrawledPage objects
            collection_name: Collection to store documents in
            source_name: Source identifier for metadata

        Returns:
            List of document IDs for successfully ingested pages
        """
        from .ingest import chunk_recursive, clean_text, count_tokens
        from .database import compute_hash

        # Get or create collection
        collection_id = self.db.get_collection_id(collection_name)
        if not collection_id:
            collection_id = self.db.create_collection(
                collection_name,
                f"Web crawl collection: {collection_name}"
            )

        doc_ids = []

        for i, page in enumerate(pages):
            logger.info(f"Ingesting [{i + 1}/{len(pages)}]: {page.title}")

            try:
                # Clean the markdown
                clean_md = clean_text(page.markdown)

                # Check for duplicates
                content_hash = compute_hash(clean_md)
                if self.db.document_exists(content_hash):
                    logger.info(f"Skipping duplicate: {page.title}")
                    continue

                # Add metadata header
                full_content = f"""# {page.title}

**Source:** {page.url}
**Crawled:** {page.crawled_at.isoformat()}

---

{clean_md}
"""

                # Chunk the content
                chunks = chunk_recursive(
                    full_content,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )

                if not chunks:
                    logger.warning(f"No chunks generated for: {page.title}")
                    continue

                # Create filename from title
                safe_title = re.sub(r'[^\w\s-]', '', page.title)[:50]
                filename = f"{source_name}_{safe_title}.md"

                # Store document
                doc_id = self.db.insert_document(
                    filename=filename,
                    source_path=None,
                    doc_type='web',
                    clean_md=full_content,
                    content_hash=content_hash,
                    collection_id=collection_id,
                    source_url=page.url
                )

                # Embed chunks
                chunk_texts = [c.text for c in chunks]
                embeddings = self.embed_client.embed_batch(chunk_texts)

                # Store chunks with embeddings
                chunks_data = []
                for j, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunks_data.append({
                        'doc_id': doc_id,
                        'chunk_index': j,
                        'content': chunk.text,
                        'embedding': embedding,
                        'token_count': chunk.tokens,
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char
                    })

                self.db.insert_chunks_batch(chunks_data)
                doc_ids.append(doc_id)

                logger.info(f"Ingested: {page.title} ({len(chunks)} chunks)")

            except Exception as e:
                logger.error(f"Failed to ingest '{page.title}': {e}")
                continue

        return doc_ids

    def crawl_and_ingest(
        self,
        scope: SubjectScope,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> dict:
        """
        Crawl a subject and ingest all pages.

        Args:
            scope: SubjectScope defining what to crawl
            progress_callback: Optional progress callback

        Returns:
            Summary dict with crawl/ingest statistics
        """
        start_time = datetime.now()

        # Crawl pages
        pages = self.crawl_subject(scope, progress_callback)

        # Ingest pages
        doc_ids = self.ingest_pages(pages, scope.collection_name, scope.name)

        elapsed = (datetime.now() - start_time).total_seconds()

        return {
            'subject': scope.name,
            'collection': scope.collection_name,
            'pages_crawled': len(pages),
            'documents_ingested': len(doc_ids),
            'document_ids': doc_ids,
            'elapsed_seconds': elapsed,
            'crawled_urls': [p.url for p in pages]
        }

    def close(self):
        """Close the HTTP client."""
        self.client.close()


def create_subject_scope(
    name: str,
    domain: str,
    url_patterns: list[str],
    seed_urls: list[str],
    content_keywords: list[str] = None,
    url_exclude_patterns: list[str] = None,
    collection_name: str = None
) -> SubjectScope:
    """
    Factory function to create a SubjectScope.

    Args:
        name: Subject name
        domain: Base domain (for reference)
        url_patterns: URL patterns to include
        seed_urls: Starting URLs
        content_keywords: Optional content filter keywords
        url_exclude_patterns: Optional URL patterns to exclude
        collection_name: Optional collection name (defaults to name)

    Returns:
        Configured SubjectScope
    """
    return SubjectScope(
        name=name,
        url_patterns=url_patterns,
        url_exclude_patterns=url_exclude_patterns or [
            r'/File:', r'/Category:', r'/Template:', r'/User:',
            r'/Talk:', r'/Special:', r'\?', r'#'
        ],
        content_keywords=content_keywords or [],
        seed_urls=seed_urls,
        collection_name=collection_name or name
    )
