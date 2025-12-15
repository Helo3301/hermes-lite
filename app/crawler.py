"""
arXiv Paper Crawler for HERMES-Lite

This module fetches papers from arXiv and other whitelisted sources,
validates URLs against the domain whitelist, and ingests them into
the HERMES-Lite knowledge base.

Security: Only downloads from whitelisted domains to prevent malicious content.
"""
import re
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import httpx

logger = logging.getLogger(__name__)

# arXiv API base URL
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Rate limiting: 5 seconds between requests for stability (arXiv minimum is 3)
RATE_LIMIT_SECONDS = 5

# Maximum file size to download (50MB)
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024


@dataclass
class Paper:
    """Represents an arXiv paper."""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: datetime
    updated: datetime
    categories: list[str]
    pdf_url: str
    arxiv_url: str


class DomainNotWhitelistedError(Exception):
    """Raised when attempting to download from a non-whitelisted domain."""
    pass


class FileTooLargeError(Exception):
    """Raised when a file exceeds the maximum allowed size."""
    pass


def extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    parsed = urlparse(url)
    return parsed.netloc.lower()


def validate_url_against_whitelist(url: str, db) -> bool:
    """
    Check if a URL's domain is in the whitelist.

    Args:
        url: The URL to validate
        db: Database instance with whitelist methods

    Returns:
        True if whitelisted, raises DomainNotWhitelistedError otherwise
    """
    domain = extract_domain(url)
    if not db.is_domain_whitelisted(domain):
        raise DomainNotWhitelistedError(
            f"Domain '{domain}' is not whitelisted. URL: {url}"
        )
    return True


class ArxivCrawler:
    """
    Crawler for fetching papers from arXiv.

    Uses the arXiv API to search for papers and download PDFs.
    All URLs are validated against the domain whitelist before downloading.
    """

    def __init__(self, db, embed_client, chunk_size: int = 512, chunk_overlap: int = 64):
        """
        Initialize the crawler.

        Args:
            db: Database instance
            embed_client: Embedding client for vectorizing documents
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.db = db
        self.embed_client = embed_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client = httpx.Client(timeout=60.0, follow_redirects=True)
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_SECONDS:
            time.sleep(RATE_LIMIT_SECONDS - elapsed)
        self.last_request_time = time.time()

    def search_arxiv(
        self,
        query: str,
        categories: list[str] = None,
        max_results: int = 10,
        days_back: int = 7
    ) -> list[Paper]:
        """
        Search arXiv for papers matching the query.

        Args:
            query: Search query (title, abstract, etc.)
            categories: List of arXiv categories (e.g., ['cs.AI', 'cs.LG'])
            max_results: Maximum number of papers to return
            days_back: Only return papers from the last N days

        Returns:
            List of Paper objects
        """
        self._rate_limit()

        # Build the search query
        search_parts = []

        if query:
            # Search in title and abstract
            search_parts.append(f'all:"{query}"')

        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            search_parts.append(f"({cat_query})")

        search_query = " AND ".join(search_parts) if search_parts else "cat:cs.AI"

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        logger.info(f"Searching arXiv: {search_query}")

        try:
            response = self.client.get(ARXIV_API_URL, params=params)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"arXiv API request failed: {e}")
            return []

        # Parse the Atom feed response
        papers = self._parse_arxiv_response(response.text, start_date)
        logger.info(f"Found {len(papers)} papers")

        return papers

    def _parse_arxiv_response(self, xml_text: str, min_date: datetime) -> list[Paper]:
        """Parse arXiv API Atom feed response."""
        papers = []

        # Define namespaces
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv response: {e}")
            return []

        for entry in root.findall('atom:entry', ns):
            try:
                # Extract paper ID from the id URL
                id_url = entry.find('atom:id', ns).text
                arxiv_id = id_url.split('/abs/')[-1]

                # Get title (remove newlines)
                title = entry.find('atom:title', ns).text
                title = ' '.join(title.split())

                # Get authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns).text
                    authors.append(name)

                # Get abstract
                abstract = entry.find('atom:summary', ns).text
                abstract = ' '.join(abstract.split())

                # Get dates
                published_str = entry.find('atom:published', ns).text
                updated_str = entry.find('atom:updated', ns).text
                published = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                updated = datetime.fromisoformat(updated_str.replace('Z', '+00:00'))

                # Filter by date
                if published.replace(tzinfo=None) < min_date:
                    continue

                # Get categories
                categories = []
                for category in entry.findall('arxiv:primary_category', ns):
                    categories.append(category.get('term'))
                for category in entry.findall('atom:category', ns):
                    cat_term = category.get('term')
                    if cat_term not in categories:
                        categories.append(cat_term)

                # Get PDF link
                pdf_url = None
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'pdf':
                        pdf_url = link.get('href')
                        break

                if not pdf_url:
                    # Construct PDF URL from ID
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

                papers.append(Paper(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    published=published,
                    updated=updated,
                    categories=categories,
                    pdf_url=pdf_url,
                    arxiv_url=f"https://arxiv.org/abs/{arxiv_id}"
                ))

            except Exception as e:
                logger.warning(f"Failed to parse entry: {e}")
                continue

        return papers

    def download_pdf(self, url: str) -> bytes:
        """
        Download a PDF from a whitelisted URL.

        Args:
            url: URL of the PDF to download

        Returns:
            PDF content as bytes

        Raises:
            DomainNotWhitelistedError: If the domain is not whitelisted
            FileTooLargeError: If the file exceeds MAX_FILE_SIZE_BYTES
        """
        # SECURITY: Validate URL against whitelist
        validate_url_against_whitelist(url, self.db)

        self._rate_limit()

        logger.info(f"Downloading PDF: {url}")

        # First, do a HEAD request to check file size
        try:
            head_response = self.client.head(url)
            content_length = head_response.headers.get('content-length')
            if content_length and int(content_length) > MAX_FILE_SIZE_BYTES:
                raise FileTooLargeError(
                    f"File size {int(content_length)} exceeds max {MAX_FILE_SIZE_BYTES}"
                )
        except httpx.HTTPError:
            pass  # Some servers don't support HEAD, continue with GET

        # Download the PDF
        response = self.client.get(url)
        response.raise_for_status()

        # Check actual size
        if len(response.content) > MAX_FILE_SIZE_BYTES:
            raise FileTooLargeError(
                f"File size {len(response.content)} exceeds max {MAX_FILE_SIZE_BYTES}"
            )

        return response.content

    def ingest_paper(
        self,
        paper: Paper,
        collection_name: str = "ai-papers"
    ) -> Optional[int]:
        """
        Download and ingest a paper into the knowledge base.

        Args:
            paper: Paper object to ingest
            collection_name: Name of the collection to add the paper to

        Returns:
            Document ID if successful, None if failed or already exists
        """
        from .ingest import ingest_document
        from .database import compute_hash

        # Get collection ID
        collection_id = self.db.get_collection_id(collection_name)
        if not collection_id:
            collection_id = self.db.create_collection(
                collection_name,
                f"Auto-created collection for {collection_name}"
            )

        try:
            # Download the PDF
            pdf_content = self.download_pdf(paper.pdf_url)

            # Check if already ingested (by URL or content hash)
            content_hash = compute_hash(pdf_content.decode('latin-1'))
            if self.db.document_exists(content_hash):
                logger.info(f"Paper already ingested: {paper.title}")
                return None

            # Create a filename from the title
            safe_title = re.sub(r'[^\w\s-]', '', paper.title)[:50]
            filename = f"{paper.arxiv_id}_{safe_title}.pdf"

            # Ingest the document
            file_path = Path(filename)
            clean_md, chunks = ingest_document(
                file_path,
                content=pdf_content,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

            # Prepend paper metadata to the markdown
            metadata_header = f"""# {paper.title}

**arXiv ID:** {paper.arxiv_id}
**Authors:** {', '.join(paper.authors)}
**Published:** {paper.published.strftime('%Y-%m-%d')}
**Categories:** {', '.join(paper.categories)}
**URL:** {paper.arxiv_url}

## Abstract

{paper.abstract}

---

"""
            clean_md = metadata_header + clean_md

            # Store document
            doc_id = self.db.insert_document(
                filename=filename,
                source_path=None,
                doc_type='pdf',
                clean_md=clean_md,
                content_hash=content_hash,
                collection_id=collection_id,
                source_url=paper.arxiv_url
            )

            # Embed and store chunks
            logger.info(f"Embedding {len(chunks)} chunks for: {paper.title}")
            chunk_texts = [c.text for c in chunks]
            embeddings = self.embed_client.embed_batch(chunk_texts)

            chunks_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunks_data.append({
                    'doc_id': doc_id,
                    'chunk_index': i,
                    'content': chunk.text,
                    'embedding': embedding,
                    'token_count': chunk.tokens,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char
                })

            self.db.insert_chunks_batch(chunks_data)

            logger.info(f"Successfully ingested: {paper.title} ({len(chunks)} chunks)")
            return doc_id

        except DomainNotWhitelistedError as e:
            logger.error(f"Security: {e}")
            return None
        except FileTooLargeError as e:
            logger.warning(f"Skipping large file: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to ingest paper '{paper.title}': {e}")
            return None

    def crawl_recent_papers(
        self,
        keywords: list[str],
        categories: list[str] = None,
        collection_name: str = "ai-papers",
        max_papers: int = 10,
        days_back: int = 7
    ) -> list[dict]:
        """
        Crawl recent papers matching keywords and ingest them.

        Args:
            keywords: List of keywords to search for
            categories: arXiv categories to search (default: cs.AI, cs.LG, cs.CL)
            collection_name: Collection to add papers to
            max_papers: Maximum number of papers to ingest
            days_back: How many days back to search

        Returns:
            List of ingested paper info dicts
        """
        if categories is None:
            categories = ['cs.AI', 'cs.LG', 'cs.CL']

        ingested = []
        seen_ids = set()

        for keyword in keywords:
            if len(ingested) >= max_papers:
                break

            papers = self.search_arxiv(
                query=keyword,
                categories=categories,
                max_results=max_papers - len(ingested),
                days_back=days_back
            )

            for paper in papers:
                if len(ingested) >= max_papers:
                    break

                # Skip duplicates within this crawl
                if paper.arxiv_id in seen_ids:
                    continue
                seen_ids.add(paper.arxiv_id)

                # Sequential processing: one paper at a time with delay
                logger.info(f"Processing paper {len(ingested) + 1}: {paper.arxiv_id}")
                doc_id = self.ingest_paper(paper, collection_name)
                if doc_id:
                    ingested.append({
                        'doc_id': doc_id,
                        'arxiv_id': paper.arxiv_id,
                        'title': paper.title,
                        'authors': paper.authors,
                        'abstract': paper.abstract,
                        'url': paper.arxiv_url,
                        'categories': paper.categories
                    })
                    logger.info(f"Successfully ingested: {paper.title[:50]}...")

                # Safety delay between papers (even if ingestion failed)
                time.sleep(RATE_LIMIT_SECONDS)

        return ingested

    def close(self):
        """Close the HTTP client."""
        self.client.close()


def run_daily_crawl(
    db,
    embed_client,
    config: dict
) -> list[dict]:
    """
    Run the daily paper crawl based on configuration.

    Args:
        db: Database instance
        embed_client: Embedding client
        config: Crawler configuration dict with keys:
            - keywords: List of search keywords
            - categories: List of arXiv categories
            - collection: Collection name
            - max_papers: Max papers per day
            - days_back: How far back to search

    Returns:
        List of ingested paper info
    """
    crawler = ArxivCrawler(db, embed_client)

    try:
        results = crawler.crawl_recent_papers(
            keywords=config.get('keywords', ['RAG', 'LLM', 'retrieval augmented']),
            categories=config.get('categories', ['cs.AI', 'cs.LG', 'cs.CL']),
            collection_name=config.get('collection', 'ai-papers'),
            max_papers=config.get('max_papers', 10),
            days_back=config.get('days_back', 7)
        )
        return results
    finally:
        crawler.close()
