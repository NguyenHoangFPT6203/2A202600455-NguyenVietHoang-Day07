from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        
        # Split on sentence boundaries: ". ", "! ", "? ", and ".\n"
        # Use regex to preserve delimiters and split
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= self.max_sentences_per_chunk:
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk = []
        
        # Add remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
        
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        
        # If text fits within chunk_size, no need to split further
        if len(current_text) <= self.chunk_size:
            return [current_text]
        
        # If no separators left, return the text as is (force a chunk)
        if not remaining_separators:
            return [current_text]
        
        # Try the first separator
        separator = remaining_separators[0]
        remaining = remaining_separators[1:]
        
        # If separator is empty, fall back to next separator
        if not separator:
            return self._split(current_text, remaining)
        
        # Split by the current separator
        parts = current_text.split(separator)
        
        # Check if any part is larger than chunk_size
        good_splits = all(len(part) <= self.chunk_size for part in parts)
        
        if good_splits:
            # All parts fit, recurse on each part separately
            chunks = []
            for part in parts:
                sub_chunks = self._split(part, self.separators)
                chunks.extend(sub_chunks)
            return chunks
        else:
            # Current separator didn't work well, try the next one
            return self._split(current_text, remaining)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_product = _dot(vec_a, vec_b)
    
    # Compute magnitudes
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(x * x for x in vec_b))
    
    # Guard against zero magnitude
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    return dot_product / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        """Compare fixed-size, sentence-based, and recursive chunking strategies."""
        result = {}
        
        # Fixed-size chunking
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=0)
        fixed_chunks = fixed_chunker.chunk(text)
        result['fixed_size'] = {
            'count': len(fixed_chunks),
            'avg_length': sum(len(c) for c in fixed_chunks) / len(fixed_chunks) if fixed_chunks else 0,
            'chunks': fixed_chunks
        }
        
        # Sentence-based chunking (approximate max_sentences based on chunk_size)
        # Assume average sentence is ~50 chars, so max_sentences = chunk_size // 50
        max_sentences = max(1, chunk_size // 50)
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=max_sentences)
        sentence_chunks = sentence_chunker.chunk(text)
        result['by_sentences'] = {
            'count': len(sentence_chunks),
            'avg_length': sum(len(c) for c in sentence_chunks) / len(sentence_chunks) if sentence_chunks else 0,
            'chunks': sentence_chunks
        }
        
        # Recursive chunking
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)
        recursive_chunks = recursive_chunker.chunk(text)
        result['recursive'] = {
            'count': len(recursive_chunks),
            'avg_length': sum(len(c) for c in recursive_chunks) / len(recursive_chunks) if recursive_chunks else 0,
            'chunks': recursive_chunks
        }
        
        return result
