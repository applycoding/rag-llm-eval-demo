"""PostgreSQL store backed by the pgvector extension."""

from __future__ import annotations

import re
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from ragqa.types import ChunkRecord, RetrievedChunk

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


class PgVectorStore:
    def __init__(self, database_url: str, *, dim: int, table: str = "chunks") -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if not _IDENTIFIER.fullmatch(table):
            raise ValueError(f"Invalid table name: {table}")
        self._table = table
        self._dim = dim
        try:
            self._conn = psycopg.connect(database_url, autocommit=True)
        except psycopg.OperationalError as exc:
            raise RuntimeError("Could not connect to PostgreSQL for the pgvector store.") from exc
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        try:
            self.ensure_schema()
        except Exception:
            self._conn.close()
            raise

    def ensure_schema(self) -> None:
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ragqa_meta (
                table_name TEXT PRIMARY KEY,
                dim INTEGER NOT NULL
            )
            """
        )
        existing = self._conn.execute(
            "SELECT dim FROM ragqa_meta WHERE table_name = %s",
            (self._table,),
        ).fetchone()
        if existing is None:
            self._conn.execute(
                "INSERT INTO ragqa_meta (table_name, dim) VALUES (%s, %s)",
                (self._table, self._dim),
            )
        elif int(existing[0]) != self._dim:
            raise RuntimeError(
                f"pgvector index dimension is {existing[0]}, but the embedder produced {self._dim}. "
                "Drop that ragqa_meta row and the chunk table, then ingest again."
            )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector({self._dim}) NOT NULL
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self._table}_embedding_hnsw
            ON {self._table}
            USING hnsw (embedding vector_cosine_ops)
            """
        )
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self._table}_document_id
            ON {self._table} (document_id)
            """
        )

    def upsert(self, records: list[ChunkRecord]) -> None:
        if not records:
            return
        with self._conn.cursor() as cursor:
            cursor.executemany(
                f"""
                INSERT INTO {self._table} (id, document_id, title, source, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    title = EXCLUDED.title,
                    source = EXCLUDED.source,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
                """,
                [
                    (
                        record.id,
                        record.document_id,
                        record.title,
                        record.source,
                        record.text,
                        Vector(record.embedding),
                    )
                    for record in records
                ],
            )

    def query(self, embedding: list[float], k: int) -> list[RetrievedChunk]:
        if k < 1:
            return []
        rows = self._conn.execute(
            f"""
            SELECT id, document_id, title, source, content, 1 - (embedding <=> %s) AS score
            FROM {self._table}
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (Vector(embedding), Vector(embedding), k),
        ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def delete_document(self, document_id: str) -> None:
        self._conn.execute(f"DELETE FROM {self._table} WHERE document_id = %s", (document_id,))

    def list_chunks(self) -> list[RetrievedChunk]:
        rows = self._conn.execute(
            f"""
            SELECT id, document_id, title, source, content, 0.0 AS score
            FROM {self._table}
            ORDER BY id
            """
        ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def count(self) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()
        return int(row[0]) if row else 0

    def clear(self) -> None:
        self._conn.execute(f"DELETE FROM {self._table}")

    def close(self) -> None:
        self._conn.close()


def _row_to_chunk(row: tuple[Any, ...]) -> RetrievedChunk:
    return RetrievedChunk(
        id=row[0],
        document_id=row[1],
        title=row[2],
        source=row[3],
        text=row[4],
        score=float(row[5]),
    )
