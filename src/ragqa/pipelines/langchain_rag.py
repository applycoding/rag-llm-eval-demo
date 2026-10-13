"""LangChain retrieval chain over the shared vector store and prompt file."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import ConfigDict

from ragqa.embeddings import Embedder
from ragqa.llm import LanguageModel
from ragqa.prompts import format_context
from ragqa.stores.base import VectorStore
from ragqa.types import RetrievedChunk


class StoreBackedRetriever(BaseRetriever):
    """LangChain retriever that embeds the question and queries the shared store."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedder: Any
    store: Any
    k: int = 4

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        embedding = self.embedder.embed_query(query)
        hits = self.store.query(embedding, self.k)
        return [_document(hit) for hit in hits]


class LangChainLLM(LLM):
    """Adapt the shared reader to a LangChain LLM."""

    backend: Any

    @property
    def _llm_type(self) -> str:
        return "ragqa"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        del stop, run_manager, kwargs
        return self.backend.generate(prompt)


class LangChainPipeline:
    name = "langchain"

    def __init__(self, embedder: Embedder, store: VectorStore, llm: LanguageModel, template: str) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = LangChainLLM(backend=llm)
        self._prompt = PromptTemplate.from_template(template)

    def answer(self, question: str, k: int) -> tuple[str, list[RetrievedChunk]]:
        captured: list[Document] = []

        def capture_context(documents: list[Document]) -> str:
            captured.extend(documents)
            return format_context([_chunk(document) for document in documents])

        retriever = StoreBackedRetriever(embedder=self._embedder, store=self._store, k=k)
        chain = (
            {
                "context": retriever | RunnableLambda(capture_context),
                "question": RunnablePassthrough(),
            }
            | self._prompt
            | self._llm
            | StrOutputParser()
        )
        text = str(chain.invoke(question))
        return text, [_chunk(document) for document in captured]


def _document(chunk: RetrievedChunk) -> Document:
    return Document(
        page_content=chunk.text,
        metadata={
            "id": chunk.id,
            "document_id": chunk.document_id,
            "title": chunk.title,
            "source": chunk.source,
            "score": chunk.score,
        },
    )


def _chunk(document: Document) -> RetrievedChunk:
    metadata = document.metadata
    return RetrievedChunk(
        id=str(metadata.get("id", "")),
        document_id=str(metadata.get("document_id", "")),
        title=str(metadata.get("title", "")),
        source=str(metadata.get("source", "")),
        text=document.page_content,
        score=float(metadata.get("score", 0.0)),
    )
