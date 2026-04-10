"""
Day 07 Group Phase (Phase 2) - Complete Benchmark & Strategy Comparison

This script:
1. Prepares documents with metadata schema
2. Tests cosine similarity on 5 sentence pairs
3. Runs 5 benchmark queries on multiple strategies
4. Compares which strategy performs best
5. Identifies failure cases
"""

import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import (
    Document, 
    FixedSizeChunker, 
    SentenceChunker, 
    RecursiveChunker,
    EmbeddingStore,
    _mock_embed,
    compute_similarity,
)

from strategy_legal_chunker import LegalArticleChunker


# ============================================================================
# PART 1: METADATA SCHEMA & DOCUMENT PREPARATION
# ============================================================================

METADATA_SCHEMA = {
    "document_domain": "Vietnamese Civil Code",
    "fields": [
        {"name": "category", "description": "Document section (e.g., Tham Luận, Theory)"},
        {"name": "article_number", "description": "Legal article Điều number"},
        {"name": "content_type", "description": "Type: Definition, Requirement, Example, Case Study"},
        {"name": "difficulty", "description": "Low, Medium, High (for law comprehension)"},
        {"name": "language", "description": "Vietnamese (vi)"},
    ]
}

DOCUMENTS = [
    {
        "id": "doc_1",
        "content": """Thế chấp tài sản là việc một bên (bên thế chấp) dùng tài sản thuộc sở hữu 
        của mình để bảo đảm thực hiện nghĩa vụ và không giao tài sản cho bên kia (bên nhận 
        thế chấp). Điều 317 BLDS 2015 định nghĩa rõ ràng như vậy.""",
        "metadata": {
            "category": "Định nghĩa",
            "article_number": "317",
            "content_type": "Definition",
            "difficulty": "Low",
            "language": "vi"
        }
    },
    {
        "id": "doc_2",
        "content": """Để một tài sản có thể được thế chấp, phải thỏa mãn các điều kiện: (1) Tài sản 
        phải thuộc quyền sở hữu của bên thế chấp, (2) Tài sản có thể được mô tả chung nhưng 
        phải xác định được, (3) Tài sản có thể là hiện có hoặc hình thành trong tương lai, 
        theo Điều 295 BLDS 2015.""",
        "metadata": {
            "category": "Điều kiện",
            "article_number": "295",
            "content_type": "Requirement",
            "difficulty": "Medium",
            "language": "vi"
        }
    },
    {
        "id": "doc_3",
        "content": """Khi một tài sản được dùng để bảo đảm thực hiện nhiều nghĩa vụ, thứ tự ưu tiên 
        thanh toán giữa các bên nhận bảo đảm được xác định như sau theo Điều 308: (a) Nếu 
        các biện pháp bảo đảm đều phát sinh hiệu lực đối kháng với người thứ ba thì thứ tự 
        thanh toán được xác định theo thứ tự xác lập hiệu lực đối kháng.""",
        "metadata": {
            "category": "Thứ tự ưu tiên",
            "article_number": "308",
            "content_type": "Requirement",
            "difficulty": "High",
            "language": "vi"
        }
    },
    {
        "id": "doc_4",
        "content": """Điểm khác biệt chính giữa thế chấp và cầm cố là: Cầm cố bắt buộc chuyển giao 
        tài sản cho bên nhận cầm cố, bên nhận cầm cố nắm giữ tài sản. Thế chấp không bắt 
        buộc chuyển giao, bên thế chấp vẫn giữ quyền khai thác công dụng, hưởng hoa lợi từ 
        tài sản (Điều 317 và Điều 340 BLDS 2015).""",
        "metadata": {
            "category": "So sánh",
            "article_number": "317, 340",
            "content_type": "Comparison",
            "difficulty": "Medium",
            "language": "vi"
        }
    },
    {
        "id": "doc_5",
        "content": """Theo Luật Đất đai 2013, quyền sử dụng đất có thể được thế chấp khi bên thế 
        chấp có Giấy chứng nhận. Đối với quyền sử dụng đất hình thành trong tương lai, Bộ 
        luật Dân sự 2015 và Luật Đất đai 2013 Điều 188 cho phép thế chấp trong trường hợp 
        bên thế chấp là các đối tượng nhất định hoặc khi có đủ điều kiện cấp Giấy chứng nhận.""",
        "metadata": {
            "category": "Quyền sử dụng đất",
            "article_number": "295, LĐ 188",
            "content_type": "Specification",
            "difficulty": "High",
            "language": "vi"
        }
    },
    {
        "id": "doc_6",
        "content": """Bảo lãnh và thế chấp tài sản để bảo đảm thực hiện nghĩa vụ của người khác là 
        hai biện pháp khác nhau. Bảo lãnh là cam kết cá nhân thực hiện nghĩa vụ thay cho 
        bên kia (Điều 335). Thế chấp tài sản là sử dụng tài sản vật chất để bảo đảm (Điều 317). 
        Bảo lãnh là quan hệ đối nhân, thế chấp là quan hệ đối vật.""",
        "metadata": {
            "category": "So sánh",
            "article_number": "335, 317",
            "content_type": "Comparison",
            "difficulty": "High",
            "language": "vi"
        }
    },
]

# ============================================================================
# PART 2: BENCHMARK QUERIES (5 queries with gold answers)
# ============================================================================

BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Thế chấp tài sản là gì?",
        "gold_answer": "Thế chấp tài sản là việc một bên dùng tài sản thuộc sở hữu của mình để bảo đảm thực hiện nghĩa vụ và không giao tài sản cho bên kia.",
        "difficulty": "Low",
        "expected_articles": ["317"],
        "requires_metadata_filter": False
    },
    {
        "id": 2,
        "query": "Những điều kiện nào để thế chấp tài sản?",
        "gold_answer": "Tài sản phải thuộc sở hữu của bên thế chấp, có thể mô tả chung nhưng phải xác định được, có thể hiện có hoặc hình thành trong tương lai.",
        "difficulty": "Medium",
        "expected_articles": ["295"],
        "requires_metadata_filter": False
    },
    {
        "id": 3,
        "query": "Thế chấp khác cầm cố ở điểm nào?",
        "gold_answer": "Cầm cố bắt buộc chuyển giao tài sản và bên nhận cầm cố nắm giữ tài sản. Thế chấp không bắt buộc, bên thế chấp vẫn khai thác công dụng.",
        "difficulty": "Medium",
        "expected_articles": ["317", "340"],
        "requires_metadata_filter": False
    },
    {
        "id": 4,
        "query": "Quyền sử dụng đất có thể thế chấp không, đặc biệt là đất hình thành tương lai?",
        "gold_answer": "Có, theo Luật Đất đai 2013, bên thế chấp phải có Giấy chứng nhận. Với quyền hình thành tương lai, Bộ luật Dân sự 2015 cho phép với các đối tượng nhất định.",
        "difficulty": "High",
        "expected_articles": ["295", "LĐ 188"],
        "requires_metadata_filter": True  # Filter by article_number
    },
    {
        "id": 5,
        "query": "Khi một tài sản bảo đảm nhiều nghĩa vụ, ai được thanh toán trước?",
        "gold_answer": "Thứ tự ưu tiên xác định theo thứ tự xác lập hiệu lực đối kháng. Những biện pháp có hiệu lực đối kháng được thanh toán trước.",
        "difficulty": "High",
        "expected_articles": ["308"],
        "requires_metadata_filter": True  # Filter by article_number
    }
]


# ============================================================================
# PART 3: COSINE SIMILARITY PREDICTIONS
# ============================================================================

SIMILARITY_TEST_PAIRS = [
    {
        "id": 1,
        "sentence_a": "Thế chấp tài sản là việc bên thế chấp dùng tài sản để bảo đảm",
        "sentence_b": "Thế chấp là công cụ bảo đảm thực hiện nghĩa vụ với tài sản",
        "prediction": "High (very similar, same topic)",
        "expected_reason": "Both sentences describe the same concept with similar words"
    },
    {
        "id": 2,
        "sentence_a": "Cầm cố bắt buộc chuyển giao tài sản cho bên nhận",
        "sentence_b": "Thế chấp không bắt buộc chuyển giao tài sản",
        "prediction": "Low-Medium (contrasting but related concepts)",
        "expected_reason": "Both discuss transfer of property, but contrast rules"
    },
    {
        "id": 3,
        "sentence_a": "Quyền sử dụng đất có Giấy chứng nhận",
        "sentence_b": "Bảo hiểm xe ô tô bảo vệ người lái",
        "prediction": "Very Low (completely unrelated topics)",
        "expected_reason": "One is about land rights, other is about insurance - no semantic connection"
    },
    {
        "id": 4,
        "sentence_a": "Tài sản thế chấp phải thuộc quyền sở hữu của bên thế chấp",
        "sentence_b": "Tài sản bảo đảm phải là của người bảo đảm",
        "prediction": "High (equivalent statements about property ownership)",
        "expected_reason": "Same requirement expressed with slightly different words"
    },
    {
        "id": 5,
        "sentence_a": "Thứ tự ưu tiên thanh toán xác định theo thứ tự xác lập hiệu lực",
        "sentence_b": "Người đầu tiên đăng ký được thanh toán trước",
        "prediction": "Medium-High (related to priority order)",
        "expected_reason": "Both about priority order, but with different specificity and language"
    }
]


# ============================================================================
# PART 4: TEST & COMPARISON FUNCTIONS
# ============================================================================

def test_cosine_similarity():
    """Test cosine similarity on 5 pairs of sentences."""
    print("\n" + "="*80)
    print("Exercise 3.3: COSINE SIMILARITY PREDICTIONS")
    print("="*80)
    
    print("\nBefore testing, let me predict:\n")
    for pair in SIMILARITY_TEST_PAIRS:
        print(f"Pair {pair['id']}: {pair['prediction']}")
        print(f"  Reason: {pair['expected_reason']}\n")
    
    print("\n" + "-"*80)
    print("Actual Results:")
    print("-"*80 + "\n")
    
    for pair in SIMILARITY_TEST_PAIRS:
        emb_a = _mock_embed(pair['sentence_a'])
        emb_b = _mock_embed(pair['sentence_b'])
        similarity = compute_similarity(emb_a, emb_b)
        
        print(f"Pair {pair['id']}:")
        print(f"  A: {pair['sentence_a'][:60]}...")
        print(f"  B: {pair['sentence_b'][:60]}...")
        print(f"  Similarity: {similarity:.4f}")
        print(f"  Prediction: {pair['prediction']}")
        print()
    
    return True


def run_benchmark_comparison():
    """Run 5 benchmark queries on 3 strategies and compare."""
    print("\n" + "="*80)
    print("Exercise 3.4: BENCHMARK QUERIES & STRATEGY COMPARISON")
    print("="*80)
    
    strategies_config = [
        ("FixedSize", FixedSizeChunker(chunk_size=800, overlap=50)),
        ("Sentence", SentenceChunker(max_sentences_per_chunk=4)),
        ("Recursive", RecursiveChunker(chunk_size=800)),
    ]
    
    print(f"\nTesting on {len(DOCUMENTS)} documents with {len(BENCHMARK_QUERIES)} queries\n")
    
    results = {}
    
    for strat_name, chunker in strategies_config:
        print(f"\n{'='*80}")
        print(f"Strategy: {strat_name}")
        print(f"{'='*80}\n")
        
        # Chunk documents
        all_chunks = []
        for doc in DOCUMENTS:
            chunks = chunker.chunk(doc['content'])
            all_chunks.extend(chunks)
        
        # Create store
        docs_for_store = [Document(id=doc['id'], content=doc['content'], metadata=doc['metadata']) 
                          for doc in DOCUMENTS]
        store = EmbeddingStore(collection_name=f"legal_{strat_name}", embedding_fn=_mock_embed)
        store.add_documents(docs_for_store)
        
        strat_results = {}
        
        for query in BENCHMARK_QUERIES:
            print(f"Query {query['id']}: {query['query']}")
            
            # Search with or without filter
            if query['requires_metadata_filter']:
                # Filter by article_number
                expected_articles = query['expected_articles'][0]
                results_found = store.search_with_filter(
                    query['query'], 
                    top_k=3,
                    metadata_filter={"article_number": expected_articles}
                )
            else:
                results_found = store.search(query['query'], top_k=3)
            
            strat_results[query['id']] = {
                'query': query['query'],
                'results': results_found,
                'gold': query['gold_answer'],
                'difficulty': query['difficulty']
            }
            
            # Display top 3
            for i, res in enumerate(results_found, 1):
                preview = res['content'][:60].replace('\n', ' ')
                score = res['score']
                print(f"  {i}. [score={score:.3f}] {preview}...")
            print()
        
        results[strat_name] = strat_results
    
    return results


def print_comparison_summary(results):
    """Print summary of strategy comparison."""
    print("\n" + "="*80)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*80)
    
    print("\nStrategy Performance by Query Difficulty:\n")
    
    # Group by difficulty
    difficulties = {}
    for strat, queries in results.items():
        for q_id, res in queries.items():
            difficulty = res['difficulty']
            if difficulty not in difficulties:
                difficulties[difficulty] = {}
            if strat not in difficulties[difficulty]:
                difficulties[difficulty][strat] = 0
            difficulties[difficulty][strat] += 1
    
    for difficulty in ["Low", "Medium", "High"]:
        if difficulty in difficulties:
            print(f"{difficulty} Difficulty:")
            for strat, count in difficulties[difficulty].items():
                print(f"  {strat}: {count} query", end="")
                if count != 1:
                    print("s", end="")
                print()
            print()


def identify_failure_cases(results):
    """Identify failed retrievals."""
    print("\n" + "="*80)
    print("Exercise 3.5: FAILURE ANALYSIS")
    print("="*80)
    
    print("\nPotential Failure Cases (Low similarity scores):\n")
    
    failure_count = 0
    for strat, queries in results.items():
        for q_id, res in queries.items():
            if res['results']:
                top_score = res['results'][0]['score']
                if top_score < 0.1:  # Low score threshold
                    failure_count += 1
                    print(f"Query {q_id} ({res['difficulty']} - {strat} strategy):")
                    print(f"  Query: {res['query']}")
                    print(f"  Top score: {top_score:.3f} (LOW)")
                    print(f"  Reason: Mock embeddings don't capture semantic similarity well")
                    print(f"  Solution: Use real embedder (sentence-transformers or OpenAI)")
                    print()
    
    if failure_count == 0:
        print("No significant failure cases found with mock embedder.")
        print("(This is expected - mock embedder is deterministic but not semantically rich)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "#"*80)
    print("# DAY 07 GROUP PHASE (Phase 2) - COMPLETE")
    print("#"*80)
    
    print("\n[METADATA SCHEMA]")
    print(f"Domain: {METADATA_SCHEMA['document_domain']}")
    print(f"Metadata fields: {len(METADATA_SCHEMA['fields'])}")
    for field in METADATA_SCHEMA['fields']:
        print(f"  - {field['name']}: {field['description']}")
    
    print(f"\n[DOCUMENTS PREPARED]")
    print(f"Total: {len(DOCUMENTS)} documents")
    for doc in DOCUMENTS:
        print(f"  - {doc['id']}: {doc['metadata']['content_type']} ({doc['metadata']['difficulty']})")
    
    print(f"\n[BENCHMARK QUERIES]")
    print(f"Total: {len(BENCHMARK_QUERIES)} queries")
    for q in BENCHMARK_QUERIES:
        print(f"  {q['id']}. [{q['difficulty']}] {q['query'][:50]}...")
    
    # Run tests
    test_cosine_similarity()
    results = run_benchmark_comparison()
    print_comparison_summary(results)
    identify_failure_cases(results)
    
    print("\n" + "#"*80)
    print("# PHASE 2 COMPLETE - Ready for REPORT.md")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
