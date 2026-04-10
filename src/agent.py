from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # Retrieve top_k relevant chunks
        results = self.store.search(question, top_k=top_k)
        
        # Build context from retrieved chunks
        context_parts = []
        for result in results:
            context_parts.append(result["content"])
        
        context = "\n---\n".join(context_parts)
        
        # Build prompt with context and question
        prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Call LLM with prompt
        return self.llm_fn(prompt)
