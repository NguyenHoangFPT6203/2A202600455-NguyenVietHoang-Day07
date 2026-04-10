"""
Day 07 - Custom Chunking Strategy for Vietnamese Legal Documents

Strategy: LegalArticleChunker
Domain: Vietnamese Civil Code (Bộ luật Dân sự 2015)
Author: @user

Rationale:
Legal documents have unique structure with Articles (Điều), Sections (Khoản),
and nested hierarchies. A header-based chunking strategy that respects article
boundaries preserves legal context and ensures coherent chunks for RAG retrieval.

This strategy is optimized for:
- Legal Q&A retrieval (e.g., "What are rights of thế chấp?")
- Cross-article references
- Consistent article-level semantics
"""

import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class LegalChunk:
    """Represents a chunk from legal document with metadata."""
    content: str
    article_number: Optional[str] = None
    talk_title: Optional[str] = None
    section_numbers: list[str] = None
    
    def __post_init__(self):
        if self.section_numbers is None:
            self.section_numbers = []


class LegalArticleChunker:
    """
    Custom chunking strategy for Vietnamese legal documents.
    
    Splits on article (Điều) boundaries and preserves:
    - Article number and title
    - Section (Khoản) structure
    - Tham Luận (talk) context
    
    This ensures legal coherence and improves retrieval for law-related queries.
    """
    
    def __init__(self, max_chunk_size: int = 1500, group_khoans: bool = True):
        """
        Args:
            max_chunk_size: Maximum chars per chunk (soft limit)
            group_khoans: If True, group multiple khoans if they fit in chunk_size
        """
        self.max_chunk_size = max_chunk_size
        self.group_khoans = group_khoans
    
    def chunk(self, text: str) -> list[str]:
        """Split legal text into chunks respecting article structure."""
        if not text:
            return []
        
        # Split into tham luan sections
        tham_luans = self._split_by_tham_luan(text)
        
        chunks = []
        for tham_luan_title, tham_luan_content in tham_luans:
            # Split each tham luan by articles
            dieu_chunks = self._split_by_articles(tham_luan_content, tham_luan_title)
            chunks.extend(dieu_chunks)
        
        return chunks
    
    def _split_by_tham_luan(self, text: str) -> list[tuple[str, str]]:
        """Extract tham luan (talk) sections."""
        # Pattern: "THAM LUẬN" followed by title
        pattern = r'^(THAM LUẬN.*?)(?=^THAM LUẬN|^ĐIỀU KIỆN|$)'
        matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
        
        tham_luans = []
        for match in matches:
            section = match.group(0)
            # Extract title (first line after THAM LUẬN)
            lines = section.split('\n')
            title = lines[0] if len(lines) > 0 else "Unknown Talk"
            tham_luans.append((title, section))
        
        # If no tham luans found, treat entire text as one section
        if not tham_luans:
            tham_luans = [("Bộ Luật Dân Sự 2015", text)]
        
        return tham_luans
    
    def _split_by_articles(self, text: str, talk_title: str) -> list[str]:
        """Split text by articles (Điều) while preserving sections (Khoản)."""
        chunks = []
        
        # Pattern: "Điều NNN" or "Điều số NNN"
        # Split on article boundaries
        dieu_pattern = r'(Điều\s+(?:số\s+)?\d+.*?)(?=Điều\s+(?:số\s+)?\d+|THAM LUẬN|ĐIỀU KIỆN|$)'
        articles = re.findall(dieu_pattern, text, re.DOTALL)
        
        if not articles:
            # No structured articles found, use recursive fallback
            return self._chunky_paragraphs(text, talk_title)
        
        for article_text in articles:
            article_text = article_text.strip()
            if not article_text:
                continue
            
            # Extract article number
            match = re.search(r'Điều\s+(?:số\s+)?(\d+)', article_text)
            article_num = match.group(1) if match else "Unknown"
            
            # Group khoans (sections) if they fit
            if self.group_khoans:
                khoans = self._split_by_khoans(article_text)
                current_chunk = ""
                
                for khoan in khoans:
                    if len(current_chunk) + len(khoan) <= self.max_chunk_size:
                        current_chunk += khoan + "\n"
                    else:
                        if current_chunk.strip():
                            chunks.append(self._add_context(current_chunk.strip(), talk_title, article_num))
                        current_chunk = khoan + "\n"
                
                if current_chunk.strip():
                    chunks.append(self._add_context(current_chunk.strip(), talk_title, article_num))
            else:
                # Don't group, each khoan is a chunk
                khoans = self._split_by_khoans(article_text)
                for khoan in khoans:
                    if khoan.strip():
                        chunks.append(self._add_context(khoan.strip(), talk_title, article_num))
        
        return chunks if chunks else self._chunky_paragraphs(text, talk_title)
    
    def _split_by_khoans(self, article_text: str) -> list[str]:
        """Split article into khoans (sections)."""
        # Pattern: "Khoản 1", "Khoản 2", etc.
        khoan_pattern = r'(Khoản\s+\d+.*?)(?=Khoản\s+\d+|$)'
        khoans = re.findall(khoan_pattern, article_text, re.DOTALL)
        
        if khoans:
            return khoans
        
        # If no khoans, return full article
        return [article_text]
    
    def _chunky_paragraphs(self, text: str, talk_title: str) -> list[str]:
        """Fallback: split by paragraphs if no structure found."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current = ""
        
        for para in paragraphs:
            if len(current) + len(para) <= self.max_chunk_size:
                current += para + "\n\n"
            else:
                if current.strip():
                    chunks.append(self._add_context(current.strip(), talk_title))
                current = para + "\n\n"
        
        if current.strip():
            chunks.append(self._add_context(current.strip(), talk_title))
        
        return chunks
    
    def _add_context(self, chunk: str, talk_title: str, article_num: Optional[str] = None) -> str:
        """Add context header to chunk."""
        if article_num:
            return f"[{talk_title} - Điều {article_num}]\n{chunk}"
        else:
            return f"[{talk_title}]\n{chunk}"


# Benchmark queries for legal document retrieval
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Thế chấp tài sản được định nghĩa như thế nào theo Bộ luật Dân sự 2015?",
        "gold_answer": "Thế chấp tài sản là việc một bên dùng tài sản thuộc sở hữu của mình để bảo đảm thực hiện nghĩa vụ và không giao tài sản cho bên kia. Bên thế chấp vẫn giữ quyền khai thác công dụng, hưởng hoa lợi từ tài sản.",
        "keywords": ["thế chấp", "tài sản", "Điều 317", "bảo đảm"]
    },
    {
        "id": 2,
        "query": "Điều kiện để một tài sản có thể được thế chấp là gì?",
        "gold_answer": "Tài sản thế chấp phải thuộc quyền sở hữu của bên thế chấp, có thể được mô tả chung nhưng phải xác định được, có thể là tài sản hiện có hoặc hình thành trong tương lai, và giá trị có thể lớn hơn, bằng hoặc nhỏ hơn giá trị nghĩa vụ được bảo đảm.",
        "keywords": ["điều kiện", "tài sản bảo đảm", "Điều 295", "sở hữu"]
    },
    {
        "id": 3,
        "query": "Khi một tài sản được dùng để bảo đảm nhiều nghĩa vụ, thứ tự ưu tiên thanh toán được xác định như thế nào?",
        "gold_answer": "Thứ tự ưu tiên thanh toán được xác định theo thứ tự xác lập hiệu lực đối kháng, hoặc nếu có biện pháp bảo đảm phát sinh hiệu lực đối kháng và không phát sinh, thì biện pháp có hiệu lực được thanh toán trước. Các bên có quyền thỏa thuận thay đổi thứ tự này.",
        "keywords": ["thứ tự ưu tiên", "thanh toán", "Điều 308", "hiệu lực đối kháng"]
    },
    {
        "id": 4,
        "query": "Liệu quyền sử dụng đất hình thành trong tương lai có được phép thế chấp không?",
        "gold_answer": "Theo Nghị định 163/2006 sửa đổi năm 2012, tài sản hình thành trong tương lai không bao gồm quyền sử dụng đất. Tuy nhiên, Bộ luật Dân sự 2015 quy định rộng hơn, cho phép thế chấp quyền sử dụng đất khi thoả mãn các điều kiện của Luật Đất đai, bao gồm cả quyền sử dụng đất hình thành trong tương lai trong một số trường hợp.",
        "keywords": ["quyền sử dụng đất", "tương lai", "Luật Đất đai", "thế chấp"]
    },
    {
        "id": 5,
        "query": "Sự khác biệt giữa thế chấp và cầm cố là gì?",
        "gold_answer": "Điểm chung: cả hai đều bảo đảm thực hiện nghĩa vụ và tài sản phải thuộc sở hữu của bên bảo đảm. Điểm khác: cầm cố bắt buộc chuyển giao tài sản cho bên nhận cầm cố, trong khi thế chấp không bắt buộc, bên thế chấp vẫn có quyền khai thác tài sản.",
        "keywords": ["thế chấp", "cầm cố", "khác biệt", "chuyển giao"]
    }
]


def test_legal_chunker():
    """Test the legal chunker on a sample text."""
    sample = """
THAM LUẬN
KHÁI QUÁT NHỮNG ĐIỂM MỚI CỦA BLDS 2015

Điều 295: Tài sản bảo đảm
1. Tài sản bảo đảm phải thuộc quyền sở hữu của bên bảo đảm.
2. Tài sản có thể được mô tả chung.

Khoản 1: Quy định chung
Theo quy định.

Khoản 2: Trường hợp đặc biệt
Trong trường hợp khác.

Điều 296: Hiệu lực
1. Giao dịch bảo đảm có hiệu lực.
2. Phát sinh hiệu lực.
"""
    
    chunker = LegalArticleChunker(max_chunk_size=500, group_khoans=True)
    chunks = chunker.chunk(sample)
    
    print("=== Legal Article Chunker Test ===")
    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
        print()


if __name__ == "__main__":
    test_legal_chunker()
    
    print("\n=== Benchmark Queries ===")
    for q in BENCHMARK_QUERIES:
        print(f"\nQuery {q['id']}: {q['query']}")
        print(f"Keywords: {', '.join(q['keywords'])}")
