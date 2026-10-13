"""HTTP API for archive question answering."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ragqa.config import Settings
from ragqa.service import QAService, build_service
from ragqa.types import Document, PipelineName


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    pipeline: PipelineName = "langchain"
    k: int = Field(default=4, ge=1, le=20)


class SourceOut(BaseModel):
    id: str
    document_id: str
    title: str
    source: str
    score: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    pipeline: PipelineName
    sources: list[SourceOut]


class IngestDocument(BaseModel):
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    title: str | None = None
    source: str | None = None


class IngestRequest(BaseModel):
    documents: list[IngestDocument] | None = None
    directory: str | None = None


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    vector_store: str
    embedder: str
    llm: str
    chunks: int


def create_app(settings: Settings | None = None, service: QAService | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.service = service or build_service(settings)
        if settings.auto_ingest and app.state.service.store.count() == 0:
            app.state.service.ingest_directory(settings.corpus_dir)
        yield
        app.state.service.close()

    app = FastAPI(
        title="Meridian Archive QA",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(FileNotFoundError)
    async def missing_file(_request: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        current: QAService = app.state.service
        return HealthResponse(
            status="ok",
            vector_store=settings.vector_store,
            embedder=current.embedder.name,
            llm=current.llm.name,
            chunks=current.store.count(),
        )

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(body: IngestRequest) -> IngestResponse:
        current: QAService = app.state.service
        if body.documents:
            documents = [
                Document(
                    document_id=item.document_id,
                    title=item.title or item.document_id,
                    source=item.source or item.document_id,
                    text=item.text,
                )
                for item in body.documents
            ]
            result = current.ingest_documents(documents)
        else:
            directory = Path(body.directory) if body.directory else settings.corpus_dir
            result = current.ingest_directory(directory)
        return IngestResponse(documents=result.documents, chunks=result.chunks)

    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest) -> QueryResponse:
        current: QAService = app.state.service
        answer = current.query(body.question, pipeline=body.pipeline, k=body.k)
        return QueryResponse(
            question=answer.question,
            answer=answer.text,
            pipeline=answer.pipeline,
            sources=[
                SourceOut(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    source=chunk.source,
                    score=chunk.score,
                    text=chunk.text,
                )
                for chunk in answer.chunks
            ],
        )

    return app
