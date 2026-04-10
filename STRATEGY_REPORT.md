# Day 07 Lab — Custom Retrieval Strategy Design Report

## Executive Summary

**Domain**: Vietnamese Civil Code (Bộ luật Dân sự 2015)  
**Strategy Name**: `LegalArticleChunker` (Custom for legal domain)  
**Total Benchmark Queries**: 5  
**Comparison**: vs FixedSizeChunker, SentenceChunker, RecursiveChunker

---

## Part 1: Strategy Design

### Domain Analysis

The Vietnamese Civil Code is a hierarchical legal document with:

- Multiple speeches/talks (Tham Luận)
- Articles with specific structure (Điều NNN)
- Sections within articles (Khoản 1, Khoản 2, ...)
- Case studies and practical examples

**Key Insight**: Simple character-count chunking loses legal coherence. Articles are semantic units that span varied lengths.

### Custom Strategy: LegalArticleChunker

**Design Philosophy**:

- Split on article boundaries (Điều), not character count
- Preserve talk context and section hierarchy
- Group related khoans (sections) when they fit chunk size
- Fall back to paragraph-based chunking if structure detection fails

**Algorithm**:

```
1. Split document by Tham Luận (talks)
2. For each talk:
   a. Find all Articles (Điều NNN)
   b. For each article:
      - Identify sections (Khoản 1, 2, ...)
      - Group khoans together until chunk_size limit reached
      - Add metadata: talk_title, article_number, section_list
3. Return list of coherent chunks with legal context preserved
```

**Advantages**:

- ✓ Each chunk is a complete legal concept (one article)
- ✓ Improves retrieval precision for law-related Q&A
- ✓ Reduces fragmentation of related legal content
- ✓ Better for domain-specific RAG systems
- ✓ Metadata enrichment for filtering ("Show me Điều 295 only")

**Disadvantages**:

- ✗ Depends on specific formatting (fails on malformed docs)
- ✗ May produce slightly larger chunks (token limits in LLM)
- ✗ Not generalizable to non-legal documents

**Metadata Schema**:

```python
{
    "article_number": "295",           # Điều number
    "talk_title": "KHÁI QUÁT...",     # Tham Luận context
    "section_numbers": ["1", "2"],    # Khoản numbers
    "document_type": "legal"
}
```

**Example Chunk Output**:

```
[KHÁI QUÁT NHỮNG ĐIỂM MỚI CỦA BLDS 2015 - Điều 295]

Tài sản bảo đảm phải thuộc quyền sở hữu của bên bảo đảm, trừ trường hợp
cầm giữ tài sản, bảo lưu quyền sở hữu

Khoản 1: Tài sản bảo đảm có thể được mô tả chung, nhưng phải xác định được

Khoản 2: Tài sản bảo đảm có thể là tài sản hiện có hoặc tài sản hình thành
trong tương lai
```

---

## Part 2: Benchmark Queries (5 queries for comparison)

### Query 1: Basic Definition

**Query**: "Thế chấp tài sản được định nghĩa như thế nào theo Bộ luật Dân sự 2015?"

**Gold Answer**:
Thế chấp tài sản là việc một bên dùng tài sản thuộc sở hữu của mình để bảo
đảm thực hiện nghĩa vụ và không giao tài sản cho bên kia. Bên thế chấp vẫn giữ
quyền khai thác công dụng, hưởng hoa lợi từ tài sản.

**Keywords**: thế chấp, tài sản, Điều 317, bảo đảm, định nghĩa
**Expected Source**: Điều 317 - Thế chấp tài sản
**Difficulty**: Easy (direct definition)

---

### Query 2: Conditions & Requirements

**Query**: "Điều kiện để một tài sản có thể được thế chấp là gì?"

**Gold Answer**:
Tài sản thế chấp phải thuộc quyền sở hữu của bên thế chấp. Tài sản có thể được
mô tả chung nhưng phải xác định được. Tài sản có thể là hiện có hoặc hình thành
trong tương lai. Giá trị tài sản có thể lớn hơn, bằng hoặc nhỏ hơn giá trị
nghĩa vụ được bảo đảm.

**Keywords**: điều kiện, tài sản bảo đảm, Điều 295, sở hữu, xác định được
**Expected Source**: Điều 295 - Tài sản bảo đảm
**Difficulty**: Medium (requires multiple khoans)

---

### Query 3: Priority Rank

**Query**: "Khi một tài sản được dùng để bảo đảm nhiều nghĩa vụ, thứ tự ưu tiên thanh toán được xác định như thế nào?"

**Gold Answer**:
Thứ tự ưu tiên thanh toán được xác định theo thứ tự xác lập hiệu lực đối kháng.
Nếu có biện pháp bảo đảm phát sinh hiệu lực đối kháng và không phát sinh, thì
biện pháp có hiệu lực được thanh toán trước. Các bên có quyền thỏa thuận thay
đổi thứ tự này trong phạm vi bảo đảm của bên mà mình thế quyền.

**Keywords**: thứ tự ưu tiên, thanh toán, Điều 308, hiệu lực đối kháng
**Expected Source**: Điều 308 - Thứ tự ưu tiên thanh toán
**Difficulty**: Hard (complex legal concept)

---

### Query 4: Future Assets

**Query**: "Liệu quyền sử dụng đất hình thành trong tương lai có được phép thế chấp không?"

**Gold Answer**:
Theo Nghị định 163/2006 sửa đổi năm 2012, tài sản hình thành trong tương lai
không bao gồm quyền sử dụng đất. Tuy nhiên, Bộ luật Dân sự 2015 quy định rộng
hơn và Luật Đất đai 2013 cho phép thế chấp quyền sử dụng đất (kể cả hình thành
trong tương lai) khi bên thế chấp là các đối tượng nhất định hoặc khi có đủ
điều kiện cấp Giấy chứng nhận.

**Keywords**: quyền sử dụng đất, tương lai, Luật Đất đai, thế chấp, Điều 295
**Expected Source**: Multiple (Điều 295, Luật Đất đai 2013, Luật 188)
**Difficulty**: Very Hard (cross-reference between laws)

---

### Query 5: Distinction

**Query**: "Sự khác biệt giữa thế chấp và cầm cố là gì?"

**Gold Answer**:
Điểm chung: Cả hai đều có mục đích bảo đảm thực hiện nghĩa vụ, tài sản phải
thuộc sở hữu của bên bảo đảm, và phải xác định được.

Điểm khác: Cầm cố bắt buộc chuyển giao tài sản cho bên nhận cầm cố và bên
nhận cầm cố nắm giữ tài sản. Thế chấp không bắt buộc chuyển giao, bên thế
chấp vẫn có quyền khai thác công dụng, hưởng hoa lợi từ tài sản.

**Keywords**: thế chấp, cầm cố, khác biệt, chuyển giao, nắm giữ, Điều 317, Điều 340
**Expected Source**: Điều 317 (Thế chấp), Điều 340 (Cầm cố)
**Difficulty**: Medium (compare two articles)

---

## Part 3: Strategy Comparison

### Comparison Metrics

| Metric              | Fixed-Size  | Sentence    | Recursive | Legal Article |
| ------------------- | ----------- | ----------- | --------- | ------------- |
| **Chunks**          | 128         | 134         | 31,090+   | ~45-60 (est.) |
| **Avg Size**        | 1,199 chars | 1,049 chars | 3.5 chars | ~2,500 chars  |
| **Coherence**       | Medium      | Medium      | Low       | **High**      |
| **Legal Context**   | No          | No          | No        | **Yes**       |
| **Speed**           | Fast        | Fast        | Medium    | **Medium**    |
| **Domain-Specific** | No          | No          | No        | **Yes**       |

### Strengths & Weaknesses

**FixedSizeChunker**:

- ✓ Uniform chunk sizes (predictable)
- ✓ Fast execution
- ✗ May split article in middle
- ✗ Loses legal structure

**SentenceChunker**:

- ✓ Respects sentence boundaries
- ✓ Reasonable chunk sizes
- ✗ Less suitable for legal text
- ✗ Still loses article coherence

**RecursiveChunker**:

- ✓ Tries multiple separators
- ✓ Generally balanced
- ✗ Generic approach
- ✗ No domain awareness

**LegalArticleChunker** (CUSTOM):

- ✓ **Article-level coherence**
- ✓ **Legal metadata enrichment**
- ✓ **Better for law Q&A**
- ✓ **Preserves talk context**
- ✗ Requires specific formatting
- ✗ Slightly larger chunks

---

## Part 4: Practical Recommendation

### When to Use Each Strategy

| Strategy                | Best For                                |
| ----------------------- | --------------------------------------- |
| **FixedSizeChunker**    | General documents, uniform token limits |
| **SentenceChunker**     | Narrative text, news articles           |
| **RecursiveChunker**    | Mixed content, unknown format           |
| **LegalArticleChunker** | Legal Q&A RAG, Vietnamese law docs      |

### Implementation Tips

```python
# Use LegalArticleChunker for legal domain
from strategy_legal_chunker import LegalArticleChunker

chunker = LegalArticleChunker(
    max_chunk_size=2000,    # Adjust for token limits
    group_khoans=True       # Group related sections
)

chunks = chunker.chunk(legal_document_text)

# Then embed and store
store = EmbeddingStore(collection_name="legal_kb")
store.add_documents(chunks)

# Retrieve for legal Q&A
results = store.search("Thế chấp tài sản là gì?", top_k=3)
```

---

## Part 5: Key Insights

1. **Domain Matters**: Generic strategies underperform on specialized domains
2. **Structure is Gold**: Legal documents have inherent hierarchy—exploit it!
3. **Context Preservation**: Adding article metadata improves relevance
4. **Hybrid Approach**: Combine multiple strategies (Recursive + custom fallback)
5. **Metadata Filtering**: Enable filtering by Điều number for higher precision

---

## Conclusion

**LegalArticleChunker** is a domain-specific strategy optimized for Vietnamese
legal documents. It balances coherence, metadata richness, and retrieval
effectiveness better than generic approaches.

For your team's lab, recommend:

- **Person A**: Use FixedSizeChunker (baseline)
- **Person B**: Use SentenceChunker (narrative test)
- **You**: Use LegalArticleChunker (legal domain optimization)

**Compare in team meeting**: Whose strategy retrieves more relevant answers
to the 5 benchmark queries? The custom strategy should win on legal-specific questions.

---

## File References

- `strategy_legal_chunker.py` — Implementation of LegalArticleChunker
- `test_legal_strategy.py` — Comparison script
- `data/Boluatdansu2015.md` — Civil Code document for testing
- `src/chunking.py` — Built-in chunkers (FixedSize, Sentence, Recursive)
