"""LlamaIndex retriever and prompt completion over the shared store and prompt file."""

from __future__ import annotations

from typing import Any

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms import CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from ragqa.embeddings import Embedder
from ragqa.llm import LanguageModel
from ragqa.prompts import format_context
from ragqa.stores.base import VectorStore
from ragqa.types import RetrievedChunk


class StoreRetriever(BaseRetriever):
    """LlamaIndex retriever that embeds the question and queries the shared store."""

    def __init__(self, embedder: Embedder, store: VectorStore, k: int) -> None:
        super().__init__()
        self._embedder = embedder
        self._store = store
        self._k = k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        embedding = self._embedder.embed_query(query_bundle.query_str)
        return [_node(chunk) for chunk in self._store.query(embedding, self._k)]


class LlamaIndexLLM(CustomLLM):
    """Adapt the shared reader to a LlamaIndex LLM."""

    backend: Any
    model_name: str = "ragqa"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=8192, num_output=512, model_name=self.model_name)

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        del formatted, kwargs
        return CompletionResponse(text=self.backend.generate(prompt))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        del formatted, kwargs
        text = self.backend.generate(prompt)
        yield CompletionResponse(text=text, delta=text)


class LlamaIndexPipeline:
    name = "llamaindex"

    def __init__(self, embedder: Embedder, store: VectorStore, llm: LanguageModel, template: str) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = LlamaIndexLLM(backend=llm)
        renamed = template.replace("{context}", "{context_str}").replace("{question}", "{query_str}")
        self._template = PromptTemplate(renamed)

    def answer(self, question: str, k: int) -> tuple[str, list[RetrievedChunk]]:
        nodes = StoreRetriever(self._embedder, self._store, k).retrieve(question)
        chunks = [_chunk(node) for node in nodes]
        prompt = self._template.format(
            context_str=format_context(chunks),
            query_str=question,
        )
        text = self._llm.complete(str(prompt)).text
        return text, chunks


def _node(chunk: RetrievedChunk) -> NodeWithScore:
    node = TextNode(
        text=chunk.text,
        id_=chunk.id,
        metadata={
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "title": chunk.title,
            "source": chunk.source,
        },
    )
    return NodeWithScore(node=node, score=chunk.score)


def _chunk(node_with_score: NodeWithScore) -> RetrievedChunk:
    node = node_with_score.node
    metadata = node.metadata or {}
    raw_text = getattr(node, "text", "")
    text = raw_text if isinstance(raw_text, str) else node.get_content()
    return RetrievedChunk(
        id=str(metadata.get("chunk_id", node.node_id)),
        document_id=str(metadata.get("document_id", "")),
        title=str(metadata.get("title", "")),
        source=str(metadata.get("source", "")),
        text=text,
        score=float(node_with_score.score or 0.0),
    )
