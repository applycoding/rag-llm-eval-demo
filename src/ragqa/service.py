"""Compose ingestion and the three answering pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import assert_never

from ragqa.chunking import chunk_document
from ragqa.config import Settings
from ragqa.embeddings import Embedder, build_embedder
from ragqa.ingestion import load_directory
from ragqa.llm import LanguageModel, build_llm
from ragqa.pipelines.langchain_rag import LangChainPipeline
from ragqa.pipelines.llamaindex_rag import LlamaIndexPipeline
from ragqa.pipelines.tfidf_rag import TfidfPipeline, TfidfRetriever
from ragqa.prompts import load_prompt
from ragqa.stores import build_store
from ragqa.stores.base import VectorStore
from ragqa.types import Answer, ChunkRecord, Document, IngestResult, PipelineName


class QAService:
    def __init__(
        self,
        *,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        llm: LanguageModel,
        template: str,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self._template = template
        self._tfidf = TfidfRetriever()
        self._langchain = LangChainPipeline(embedder, store, llm, template)
        self._llamaindex = LlamaIndexPipeline(embedder, store, llm, template)
        self._tfidf_pipeline = TfidfPipeline(self._tfidf, llm, template)
        self._refresh_tfidf()

    def ingest_directory(self, path: Path) -> IngestResult:
        return self.ingest_documents(load_directory(path))

    def ingest_documents(self, documents: list[Document]) -> IngestResult:
        chunks = []
        for document in documents:
            self.store.delete_document(document.document_id)
            chunks.extend(
                chunk_document(
                    document,
                    chunk_size=self.settings.chunk_size,
                    overlap=self.settings.chunk_overlap,
                )
            )
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        records = [
            ChunkRecord(
                id=chunk.id,
                document_id=chunk.document_id,
                title=chunk.title,
                source=chunk.source,
                text=chunk.text,
                embedding=vector,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.store.upsert(records)
        self._refresh_tfidf()
        return IngestResult(documents=len(documents), chunks=len(chunks))

    def query(self, question: str, *, pipeline: PipelineName = "langchain", k: int = 4) -> Answer:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("question must not be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        text, chunks = self._select(pipeline).answer(cleaned, k)
        return Answer(question=cleaned, text=text, pipeline=pipeline, chunks=chunks)

    def close(self) -> None:
        self.store.close()

    def _refresh_tfidf(self) -> None:
        self._tfidf.fit(self.store.list_chunks())

    def _select(self, pipeline: PipelineName) -> LangChainPipeline | LlamaIndexPipeline | TfidfPipeline:
        if pipeline == "langchain":
            return self._langchain
        if pipeline == "llamaindex":
            return self._llamaindex
        if pipeline == "tfidf":
            return self._tfidf_pipeline
        assert_never(pipeline)


def build_service(settings: Settings | None = None) -> QAService:
    settings = settings or Settings()
    embedder = build_embedder(settings)
    store = build_store(settings, embedder)
    llm = build_llm(settings)
    template = load_prompt(settings.prompt_path)
    return QAService(settings=settings, embedder=embedder, store=store, llm=llm, template=template)
