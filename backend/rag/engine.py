import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from providers.factory import get_llm, get_streaming_llm
from vectorstore.chroma import get_vectorstore
from rag.prompts import RAG_PROMPT
from rag.retrieval import get_hybrid_retriever, multi_query_retrieve, hyde_retrieve
from rag.reranking import rerank_documents
from rag.postprocessing import remove_redundant, reorder_long_context
from models.schemas import Source
from config import settings

logger = logging.getLogger(__name__)


class RAGEngine:
    def _retrieve(self, question: str) -> list[Document]:
        """Configurable retrieval pipeline based on settings."""
        llm = get_llm()

        # Step 1: Retrieve documents
        if settings.use_multi_query:
            logger.info("Using multi-query retrieval")
            docs = multi_query_retrieve(question, llm)
        elif settings.use_hyde:
            logger.info("Using HyDE retrieval")
            docs = hyde_retrieve(question, llm)
        elif settings.use_hybrid_search:
            logger.info("Using hybrid (BM25 + vector) retrieval")
            retriever = get_hybrid_retriever()
            docs = retriever.invoke(question)
        else:
            logger.info("Using simple vector similarity search")
            vectorstore = get_vectorstore()
            results = vectorstore.similarity_search_with_relevance_scores(
                question, k=settings.retrieval_top_k
            )
            # Attach scores to metadata for simple search
            docs = []
            for doc, score in results:
                doc.metadata["relevance_score"] = score
                docs.append(doc)

        if not docs:
            return []

        # Step 2: Rerank if enabled
        if settings.use_reranking:
            logger.info("Reranking %d documents", len(docs))
            docs = rerank_documents(question, docs)

        # Step 3: Post-processing (always applied)
        docs = remove_redundant(docs)
        docs = reorder_long_context(docs)

        return docs

    def _build_context(self, docs: list[Document]) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source_file', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    def _build_sources(self, docs: list[Document]) -> list[Source]:
        return [
            Source(
                doc_name=doc.metadata.get("source_file", "unknown"),
                page=doc.metadata.get("page"),
                chunk_text=doc.page_content[:300],
                relevance_score=round(doc.metadata.get("relevance_score", 0.0), 4),
            )
            for doc in docs
        ]

    def query(self, question: str, use_hyde: bool = False) -> tuple[str, list[Source]]:
        llm = get_llm()
        docs = self._retrieve(question)

        if not docs:
            return "I don't have enough context to answer this question. Please upload relevant documents first.", []

        context = self._build_context(docs)
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        return answer, self._build_sources(docs)

    async def stream_query(self, question: str, use_hyde: bool = False):
        streaming_llm = get_streaming_llm()
        docs = self._retrieve(question)

        if not docs:
            yield {"type": "token", "content": "I don't have enough context to answer this question. Please upload relevant documents first."}
            yield {"type": "sources", "sources": []}
            return

        context = self._build_context(docs)
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        chain = prompt | streaming_llm

        async for chunk in chain.astream({"context": context, "question": question}):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                yield {"type": "token", "content": token}

        sources = self._build_sources(docs)
        yield {"type": "sources", "sources": [s.model_dump() for s in sources]}


rag_engine = RAGEngine()
