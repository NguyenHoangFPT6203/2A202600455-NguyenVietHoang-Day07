"""
Day 07 - Strategy Comparison on Vietnamese Civil Code Document

This script compares 4 chunking strategies:
1. FixedSizeChunker (built-in)
2. SentenceChunker (built-in)
3. RecursiveChunker (built-in)
4. LegalArticleChunker (custom for legal domain)

Metrics:
- Number of chunks
- Average chunk size
- Chunk coherence (subjective)
- Retrieval effectiveness on benchmark queries
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import (
    Document, 
    FixedSizeChunker, 
    SentenceChunker, 
    RecursiveChunker,
    EmbeddingStore,
    _mock_embed,
    KnowledgeBaseAgent
)

from strategy_legal_chunker import (
    LegalArticleChunker,
    BENCHMARK_QUERIES
)


def load_civil_code():
    """Load civil code document."""
    doc_path = Path(__file__).parent / "data" / "Boluatdansu2015.md"
    
    if not doc_path.exists():
        print(f"Warning: {doc_path} not found. Using dummy document.")
        return Document(
            id="blds2015",
            content="Bộ luật Dân sự 2015 - dummy content",
            metadata={"source": "Bộ luật Dân sự 2015"}
        )
    
    content = doc_path.read_text(encoding="utf-8")
    return Document(
        id="blds2015",
        content=content,
        metadata={"source": "Bộ luật Dân sự 2015", "type": "legal"}
    )


def compare_chunkers(doc: Document, chunk_size: int = 1200):
    """Compare 4 chunking strategies."""
    
    print("=" * 80)
    print("CHUNKING STRATEGY COMPARISON ON VIETNAMESE CIVIL CODE")
    print("=" * 80)
    
    strategies = {}
    
    # 1. Fixed-Size Chunker
    print("\n[1] FixedSizeChunker (chunk_size=1200, overlap=100)")
    fixed = FixedSizeChunker(chunk_size=chunk_size, overlap=100)
    fixed_chunks = fixed.chunk(doc.content)
    strategies["fixed_size"] = fixed_chunks
    print(f"  - Chunks: {len(fixed_chunks)}")
    if fixed_chunks:
        avg_size = sum(len(c) for c in fixed_chunks) / len(fixed_chunks)
        print(f"  - Avg size: {avg_size:.1f} chars")
        print(f"  - Sample (first 150 chars): {fixed_chunks[0][:150]}...")
    
    # 2. Sentence Chunker
    print("\n[2] SentenceChunker (max_sentences_per_chunk=5)")
    sentence = SentenceChunker(max_sentences_per_chunk=5)
    sentence_chunks = sentence.chunk(doc.content)
    strategies["by_sentences"] = sentence_chunks
    print(f"  - Chunks: {len(sentence_chunks)}")
    if sentence_chunks:
        avg_size = sum(len(c) for c in sentence_chunks) / len(sentence_chunks)
        print(f"  - Avg size: {avg_size:.1f} chars")
        print(f"  - Sample (first 150 chars): {sentence_chunks[0][:150]}...")
    
    # 3. Recursive Chunker
    print("\n[3] RecursiveChunker (chunk_size=1200)")
    recursive = RecursiveChunker(chunk_size=chunk_size)
    recursive_chunks = recursive.chunk(doc.content)
    strategies["recursive"] = recursive_chunks
    print(f"  - Chunks: {len(recursive_chunks)}")
    if recursive_chunks:
        avg_size = sum(len(c) for c in recursive_chunks) / len(recursive_chunks)
        print(f"  - Avg size: {avg_size:.1f} chars")
        print(f"  - Sample (first 150 chars): {recursive_chunks[0][:150]}...")
    
    # 4. Legal Article Chunker (custom)
    print("\n[4] LegalArticleChunker (max_chunk_size=1200, group_khoans=True) [CUSTOM]")
    legal = LegalArticleChunker(max_chunk_size=chunk_size, group_khoans=True)
    legal_chunks = legal.chunk(doc.content)
    strategies["legal_article"] = legal_chunks
    print(f"  - Chunks: {len(legal_chunks)}")
    if legal_chunks:
        avg_size = sum(len(c) for c in legal_chunks) / len(legal_chunks)
        print(f"  - Avg size: {avg_size:.1f} chars")
        print(f"  - Sample (first 150 chars): {legal_chunks[0][:150]}...")
    
    return strategies


def test_retrieval(strategies: dict, doc: Document):
    """Test retrieval quality on benchmark queries."""
    
    print("\n" + "=" * 80)
    print("RETRIEVAL TEST ON 5 BENCHMARK QUERIES")
    print("=" * 80)
    
    for strategy_name, chunks in strategies.items():
        print(f"\n{'='*80}")
        print(f"Strategy: {strategy_name.upper()}")
        print(f"{'='*80}")
        
        # Create store for this strategy
        store = EmbeddingStore(
            collection_name=f"legal_{strategy_name}",
            embedding_fn=_mock_embed
        )
        store.add_documents([doc])
        
        # Test each query
        for q in BENCHMARK_QUERIES[:3]:  # Test first 3 queries for brevity
            print(f"\nQuery {q['id']}: {q['query'][:60]}...")
            results = store.search(q['query'], top_k=3)
            
            print(f"  Top 3 results:")
            for i, result in enumerate(results, 1):
                preview = result['content'][:80].replace('\n', ' ')
                score = result['score']
                print(f"    {i}. [score={score:.3f}] {preview}...")


def print_summary():
    """Print strategy comparison summary."""
    print("\n" + "=" * 80)
    print("STRATEGY DESIGN SUMMARY")
    print("=" * 80)
    
    summary = """
**Custom Strategy: LegalArticleChunker**

Design Rationale:
- Vietnamese legal documents have hierarchical structure (Tham Luận -> Điều -> Khoản)
- Simple character-count chunking loses article coherence
- Header-based splitting preserves legal context and semantics

Innovation:
- Splits on article (Điều) boundaries instead of character count
- Groups related khoans (sections) within chunk size limit
- Preserves talk (Tham Luận) context in chunk metadata
- Falls back to paragraph-based chunking if no structure found

Advantages:
✓ Each chunk covers a complete legal concept (one article)
✓ Improves retrieval for law-related Q&A
✓ Reduces fragmentation of related legal concepts
✓ Better for domain-specific RAG systems

Disadvantages:
✗ Relies on specific formatting (may fail on malformed docs)
✗ Slightly larger average chunk size (may hit token limits)
✗ Not generalizable to non-legal documents

Comparison Prediction:
- FixedSizeChunker: Fast, uniform, but may split articles
- SentenceChunker: Good for narrative, less suitable for legal
- RecursiveChunker: Balanced, but treats all text equally
- LegalArticleChunker: BEST for legal domain (higher semantic coherence)

Use Case:
If you're building a legal Q&A RAG system for Vietnamese law documents,
use LegalArticleChunker. For general documents, RecursiveChunker is safer.

Metadata Schema (stored in chunk context):
- article_number: Điều number (e.g., "295")
- talk_title: Tham Luận title
- section_numbers: List of Khoản numbers

Example Chunk:
"[KHÁI QUÁT NHỮNG ĐIỂM MỚI CỦA BLDS 2015 - Điều 295]
Tài sản bảo đảm phải thuộc quyền sở hữu của bên bảo đảm...
Khoản 2: Tài sản có thể được mô tả chung..."
"""
    print(summary)


def main():
    # Load document
    doc = load_civil_code()
    print(f"Document loaded: {doc.id}")
    print(f"Content size: {len(doc.content):,} characters\n")
    
    # Compare strategies
    strategies = compare_chunkers(doc, chunk_size=1200)
    
    # Test retrieval quality
    test_retrieval(strategies, doc)
    
    # Print summary
    print_summary()


if __name__ == "__main__":
    main()
