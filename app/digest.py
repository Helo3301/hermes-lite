"""
Daily Digest Generator for HERMES-Lite

Generates summaries of newly ingested papers and sends notifications.
Can use a local LLM (via Ollama) or the Anthropic API for summarization.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional
import json

import httpx

logger = logging.getLogger(__name__)


def summarize_with_ollama(
    text: str,
    ollama_host: str = "http://localhost:11434",
    model: str = "llama3.2"
) -> str:
    """
    Generate a summary using a local Ollama model.

    Args:
        text: Text to summarize
        ollama_host: Ollama API host
        model: Model to use for summarization

    Returns:
        Summary string
    """
    prompt = f"""Summarize this research paper in 2-3 concise bullet points.
Focus on: what problem it solves, the key technique/method, and main results.
Be specific and technical but accessible.

Paper:
{text[:8000]}

Summary (2-3 bullet points):"""

    try:
        response = httpx.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 500
                }
            },
            timeout=120.0
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama summarization failed: {e}")
        return ""


def summarize_with_anthropic(
    text: str,
    api_key: str,
    model: str = "claude-3-haiku-20240307"
) -> str:
    """
    Generate a summary using the Anthropic API.

    Args:
        text: Text to summarize
        api_key: Anthropic API key
        model: Model to use

    Returns:
        Summary string
    """
    prompt = f"""Summarize this research paper in 2-3 concise bullet points.
Focus on: what problem it solves, the key technique/method, and main results.
Be specific and technical but accessible.

Paper:
{text[:8000]}"""

    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": model,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=60.0
        )
        response.raise_for_status()
        content = response.json().get("content", [])
        if content:
            return content[0].get("text", "").strip()
        return ""
    except Exception as e:
        logger.error(f"Anthropic summarization failed: {e}")
        return ""


class DigestGenerator:
    """Generates daily digests of newly ingested papers."""

    def __init__(
        self,
        db,
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "llama3.2",
        anthropic_api_key: Optional[str] = None,
        use_local: bool = True
    ):
        """
        Initialize the digest generator.

        Args:
            db: Database instance
            ollama_host: Ollama API host
            ollama_model: Ollama model for summarization
            anthropic_api_key: Optional Anthropic API key
            use_local: If True, use Ollama; if False, use Anthropic
        """
        self.db = db
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        self.anthropic_api_key = anthropic_api_key
        self.use_local = use_local

    def summarize(self, text: str) -> str:
        """Generate a summary of the given text."""
        if self.use_local:
            return summarize_with_ollama(text, self.ollama_host, self.ollama_model)
        elif self.anthropic_api_key:
            return summarize_with_anthropic(text, self.anthropic_api_key)
        else:
            # Fallback: just return the abstract/first part
            return text[:500] + "..."

    def get_recent_papers(
        self,
        collection_name: str = "ai-papers",
        hours_back: int = 24
    ) -> list[dict]:
        """
        Get papers ingested in the last N hours.

        Args:
            collection_name: Collection to query
            hours_back: How many hours back to look

        Returns:
            List of document dicts
        """
        collection_id = self.db.get_collection_id(collection_name)
        if not collection_id:
            return []

        cutoff = datetime.now() - timedelta(hours=hours_back)
        cutoff_str = cutoff.strftime('%Y-%m-%d %H:%M:%S')

        cursor = self.db.conn.execute(
            """SELECT id, filename, source_url, ingested_at, clean_md
               FROM documents
               WHERE collection_id = ? AND ingested_at >= ?
               ORDER BY ingested_at DESC""",
            (collection_id, cutoff_str)
        )

        return [dict(row) for row in cursor]

    def generate_digest(
        self,
        collection_name: str = "ai-papers",
        hours_back: int = 24,
        summarize_papers: bool = True
    ) -> dict:
        """
        Generate a digest of recently ingested papers.

        Args:
            collection_name: Collection to generate digest for
            hours_back: How many hours back to include
            summarize_papers: Whether to generate summaries (slower)

        Returns:
            Digest dict with papers and summaries
        """
        papers = self.get_recent_papers(collection_name, hours_back)

        if not papers:
            return {
                "generated_at": datetime.now().isoformat(),
                "collection": collection_name,
                "hours_back": hours_back,
                "paper_count": 0,
                "papers": [],
                "summary": "No new papers in this period."
            }

        digest_papers = []
        for paper in papers:
            paper_info = {
                "id": paper["id"],
                "filename": paper["filename"],
                "url": paper["source_url"],
                "ingested_at": paper["ingested_at"]
            }

            # Extract title from markdown (first # heading)
            clean_md = paper.get("clean_md", "")
            lines = clean_md.split('\n')
            title = "Untitled"
            for line in lines:
                if line.startswith('# '):
                    title = line[2:].strip()
                    break
            paper_info["title"] = title

            # Extract abstract if present
            abstract = ""
            in_abstract = False
            for line in lines:
                if line.strip() == "## Abstract":
                    in_abstract = True
                    continue
                if in_abstract:
                    if line.startswith("##") or line.startswith("---"):
                        break
                    abstract += line + "\n"
            paper_info["abstract"] = abstract.strip()

            # Generate summary if requested
            if summarize_papers and clean_md:
                logger.info(f"Summarizing: {title[:50]}...")
                paper_info["summary"] = self.summarize(clean_md)
            else:
                paper_info["summary"] = abstract[:300] + "..." if len(abstract) > 300 else abstract

            digest_papers.append(paper_info)

        return {
            "generated_at": datetime.now().isoformat(),
            "collection": collection_name,
            "hours_back": hours_back,
            "paper_count": len(digest_papers),
            "papers": digest_papers,
            "summary": f"Found {len(digest_papers)} new papers in the last {hours_back} hours."
        }

    def format_digest_markdown(self, digest: dict) -> str:
        """
        Format a digest as markdown for display or email.

        Args:
            digest: Digest dict from generate_digest()

        Returns:
            Markdown formatted string
        """
        md = f"""# AI Papers Digest

**Generated:** {digest['generated_at']}
**Collection:** {digest['collection']}
**Period:** Last {digest['hours_back']} hours
**Papers:** {digest['paper_count']}

---

"""
        if not digest['papers']:
            md += "*No new papers in this period.*\n"
            return md

        for i, paper in enumerate(digest['papers'], 1):
            md += f"""## {i}. {paper['title']}

**URL:** {paper.get('url', 'N/A')}
**Ingested:** {paper['ingested_at']}

### Summary
{paper.get('summary', 'No summary available.')}

---

"""
        return md

    def format_digest_html(self, digest: dict) -> str:
        """
        Format a digest as HTML for email.

        Args:
            digest: Digest dict from generate_digest()

        Returns:
            HTML formatted string
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #1a1a2e; }}
        h2 {{ color: #00d4ff; margin-top: 30px; }}
        .meta {{ color: #666; font-size: 14px; margin-bottom: 20px; }}
        .paper {{ background: #f5f5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .paper-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
        .paper-url {{ font-size: 14px; color: #0066cc; }}
        .summary {{ margin-top: 10px; line-height: 1.6; }}
        hr {{ border: none; border-top: 1px solid #ddd; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>AI Papers Digest</h1>
    <div class="meta">
        <strong>Generated:</strong> {digest['generated_at']}<br>
        <strong>Collection:</strong> {digest['collection']}<br>
        <strong>Period:</strong> Last {digest['hours_back']} hours<br>
        <strong>Papers:</strong> {digest['paper_count']}
    </div>
    <hr>
"""
        if not digest['papers']:
            html += "<p><em>No new papers in this period.</em></p>"
        else:
            for i, paper in enumerate(digest['papers'], 1):
                url = paper.get('url', '#')
                html += f"""
    <div class="paper">
        <div class="paper-title">{i}. {paper['title']}</div>
        <div class="paper-url"><a href="{url}">{url}</a></div>
        <div class="summary"><strong>Summary:</strong><br>{paper.get('summary', 'No summary available.')}</div>
    </div>
"""

        html += """
</body>
</html>
"""
        return html


def send_digest_email(
    digest_html: str,
    to_email: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    from_email: str = None
):
    """
    Send the digest via email.

    Note: Requires smtplib which is in the standard library.
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    if from_email is None:
        from_email = smtp_user

    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"AI Papers Digest - {datetime.now().strftime('%Y-%m-%d')}"
    msg['From'] = from_email
    msg['To'] = to_email

    html_part = MIMEText(digest_html, 'html')
    msg.attach(html_part)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(from_email, to_email, msg.as_string())
        logger.info(f"Digest email sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send digest email: {e}")
        return False


def send_digest_discord(
    digest: dict,
    webhook_url: str
):
    """
    Send the digest to a Discord webhook.

    Args:
        digest: Digest dict
        webhook_url: Discord webhook URL
    """
    # Format for Discord (limited to 2000 chars per message)
    content = f"**AI Papers Digest** - {digest['paper_count']} new papers\n\n"

    for paper in digest['papers'][:5]:  # Limit to 5 papers per message
        content += f"**{paper['title'][:100]}**\n"
        content += f"{paper.get('url', 'N/A')}\n"
        summary = paper.get('summary', '')[:200]
        if summary:
            content += f"_{summary}_\n"
        content += "\n"

    if len(digest['papers']) > 5:
        content += f"_...and {len(digest['papers']) - 5} more papers_\n"

    try:
        response = httpx.post(
            webhook_url,
            json={"content": content[:2000]},
            timeout=10.0
        )
        response.raise_for_status()
        logger.info("Digest sent to Discord")
        return True
    except Exception as e:
        logger.error(f"Failed to send Discord notification: {e}")
        return False
