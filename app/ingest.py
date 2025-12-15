"""Document ingestion pipeline: convert, clean, chunk."""
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import logging

import tiktoken

logger = logging.getLogger(__name__)

# Tokenizer for counting tokens
tokenizer = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """A chunk of text from a document."""
    text: str
    tokens: int
    start_char: int
    end_char: int


# Cleaning patterns for common document artifacts
CLEANING_PATTERNS = [
    # Page numbers
    (r'^\s*\d+\s*$', '', re.MULTILINE),
    (r'Page \d+ of \d+', '', 0),

    # Copyright notices
    (r'©.*?\d{4}.*?(?:rights reserved)?', '', re.IGNORECASE),
    (r'Copyright.*?\d{4}', '', re.IGNORECASE),

    # Excessive whitespace
    (r'\n{3,}', '\n\n', 0),
    (r'[ \t]+', ' ', 0),

    # Broken hyphenation (word-\nword -> word)
    (r'(\w+)-\n(\w+)', r'\1\2', 0),

    # Smart quotes to straight
    (r'[\u201c\u201d\u201e]', '"', 0),  # " " „
    (r'[\u2018\u2019\u201a]', "'", 0),  # ' ' ‚

    # Common PDF artifacts
    (r'\x0c', '\n', 0),  # Form feed
    (r'[\x00-\x08\x0b\x0e-\x1f]', '', 0),  # Control chars
]


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text))


def convert_to_markdown(file_path: Path, content: Optional[bytes] = None) -> str:
    """Convert document to markdown using pandoc."""
    suffix = file_path.suffix.lower()

    # Map extensions to pandoc formats
    format_map = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'doc',
        '.xlsx': 'xlsx',
        '.xls': 'xls',
        '.pptx': 'pptx',
        '.html': 'html',
        '.htm': 'html',
        '.epub': 'epub',
        '.odt': 'odt',
        '.rtf': 'rtf',
        '.md': None,  # Already markdown
        '.txt': None,  # Plain text
    }

    if suffix not in format_map:
        raise ValueError(f"Unsupported file type: {suffix}")

    # If already markdown or text, just read it
    if format_map[suffix] is None:
        if content:
            return content.decode('utf-8', errors='replace')
        return file_path.read_text(errors='replace')

    # Use temp file if content provided
    if content:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        file_to_convert = tmp_path
    else:
        file_to_convert = file_path
        tmp_path = None

    try:
        # Try pandoc first
        result = subprocess.run(
            ['pandoc', '-f', format_map[suffix], '-t', 'markdown', '--wrap=none', str(file_to_convert)],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout

        # Fallback for PDFs: use pdftotext
        if suffix == '.pdf':
            logger.info("Pandoc failed for PDF, trying pdftotext")
            result = subprocess.run(
                ['pdftotext', '-layout', str(file_to_convert), '-'],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                return result.stdout

        raise RuntimeError(f"Conversion failed: {result.stderr}")

    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


def clean_text(text: str) -> str:
    """Clean text using regex patterns."""
    cleaned = text

    for pattern, replacement, flags in CLEANING_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=flags)

    # Strip leading/trailing whitespace from lines
    lines = [line.strip() for line in cleaned.split('\n')]
    cleaned = '\n'.join(lines)

    # Final whitespace cleanup
    cleaned = cleaned.strip()

    return cleaned


def chunk_recursive(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_chunk_size: int = 100
) -> list[Chunk]:
    """
    Recursively chunk text, trying to preserve semantic boundaries.

    Strategy:
    1. Try to split on paragraph breaks (\\n\\n)
    2. If chunk too large, split on line breaks (\\n)
    3. If still too large, split on sentence boundaries (. )
    4. Last resort: split on token boundary
    """
    if not text.strip():
        return []

    chunks = []
    current_pos = 0

    # Split into paragraphs first
    paragraphs = re.split(r'\n\n+', text)

    current_chunk_parts = []
    current_chunk_tokens = 0
    chunk_start = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = count_tokens(para)

        # If single paragraph exceeds chunk size, split it further
        if para_tokens > chunk_size:
            # First, flush current chunk if any
            if current_chunk_parts:
                chunk_text = '\n\n'.join(current_chunk_parts)
                if count_tokens(chunk_text) >= min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        tokens=current_chunk_tokens,
                        start_char=chunk_start,
                        end_char=chunk_start + len(chunk_text)
                    ))
                current_chunk_parts = []
                current_chunk_tokens = 0

            # Split large paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_chunk_parts = []
            sent_chunk_tokens = 0
            sent_start = current_pos

            for sent in sentences:
                sent_tokens = count_tokens(sent)

                if sent_chunk_tokens + sent_tokens > chunk_size and sent_chunk_parts:
                    chunk_text = ' '.join(sent_chunk_parts)
                    if count_tokens(chunk_text) >= min_chunk_size:
                        chunks.append(Chunk(
                            text=chunk_text,
                            tokens=sent_chunk_tokens,
                            start_char=sent_start,
                            end_char=sent_start + len(chunk_text)
                        ))
                    # Overlap: keep some sentences
                    overlap_tokens = 0
                    overlap_parts = []
                    for s in reversed(sent_chunk_parts):
                        s_tokens = count_tokens(s)
                        if overlap_tokens + s_tokens > chunk_overlap:
                            break
                        overlap_parts.insert(0, s)
                        overlap_tokens += s_tokens

                    sent_chunk_parts = overlap_parts
                    sent_chunk_tokens = overlap_tokens
                    sent_start = current_pos

                sent_chunk_parts.append(sent)
                sent_chunk_tokens += sent_tokens
                current_pos += len(sent) + 1

            # Add remaining sentences
            if sent_chunk_parts:
                chunk_text = ' '.join(sent_chunk_parts)
                if count_tokens(chunk_text) >= min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        tokens=sent_chunk_tokens,
                        start_char=sent_start,
                        end_char=sent_start + len(chunk_text)
                    ))

            chunk_start = current_pos
            continue

        # Check if adding this paragraph exceeds chunk size
        if current_chunk_tokens + para_tokens > chunk_size and current_chunk_parts:
            # Emit current chunk
            chunk_text = '\n\n'.join(current_chunk_parts)
            if count_tokens(chunk_text) >= min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    tokens=current_chunk_tokens,
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_text)
                ))

            # Overlap: keep last paragraph(s) up to overlap tokens
            overlap_tokens = 0
            overlap_parts = []
            for p in reversed(current_chunk_parts):
                p_tokens = count_tokens(p)
                if overlap_tokens + p_tokens > chunk_overlap:
                    break
                overlap_parts.insert(0, p)
                overlap_tokens += p_tokens

            current_chunk_parts = overlap_parts
            current_chunk_tokens = overlap_tokens
            chunk_start = current_pos

        current_chunk_parts.append(para)
        current_chunk_tokens += para_tokens
        current_pos += len(para) + 2  # +2 for \n\n

    # Don't forget the last chunk
    if current_chunk_parts:
        chunk_text = '\n\n'.join(current_chunk_parts)
        if count_tokens(chunk_text) >= min_chunk_size:
            chunks.append(Chunk(
                text=chunk_text,
                tokens=current_chunk_tokens,
                start_char=chunk_start,
                end_char=chunk_start + len(chunk_text)
            ))

    return chunks


def ingest_document(
    file_path: Path,
    content: Optional[bytes] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_chunk_size: int = 100
) -> tuple[str, list[Chunk]]:
    """
    Full ingestion pipeline: convert -> clean -> chunk.

    Returns:
        Tuple of (cleaned_markdown, list_of_chunks)
    """
    logger.info(f"Converting {file_path.name}")
    raw_md = convert_to_markdown(file_path, content)

    logger.info(f"Cleaning text ({len(raw_md)} chars)")
    clean_md = clean_text(raw_md)

    logger.info(f"Chunking ({len(clean_md)} chars)")
    chunks = chunk_recursive(clean_md, chunk_size, chunk_overlap, min_chunk_size)

    logger.info(f"Created {len(chunks)} chunks")
    return clean_md, chunks
