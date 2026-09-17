"""vector_documents table

Revision ID: 0003_vector_documents
Revises: 0002_enable_pgvector
Create Date: 2026-09-17

One table for all three logical collections (research_questions_index,
knowledge_index, sub_questions_index), discriminated by `collection` --
see db/models.py::VectorDocument and memory/vector_backend.py::PgVectorBackend.
No existing production data to migrate.

embedding is vector(768): the app's settings.EMBEDDING_DIMENSIONS at the
time this migration was written (gemini-embedding-001, truncated from its
native 3072 dims). pgvector cannot build an HNSW/IVFFlat index above 2000
dimensions, so this value isn't just a config default -- it's baked into
the index type this migration creates. Changing EMBEDDING_DIMENSIONS later
requires a new migration to rebuild this column and its index.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0003_vector_documents"
down_revision: Union[str, Sequence[str], None] = "0002_enable_pgvector"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIMENSIONS = 768


def upgrade() -> None:
    op.create_table(
        "vector_documents",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("collection", sa.String(), nullable=False),
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("document", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_vector_documents")),
        sa.UniqueConstraint(
            "collection", "doc_id", name="uq_vector_documents_collection_doc_id"
        ),
    )
    op.create_index(
        "ix_vector_documents_collection", "vector_documents", ["collection"], unique=False
    )
    op.create_index(
        "ix_vector_documents_embedding_hnsw",
        "vector_documents",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_vector_documents_embedding_hnsw", table_name="vector_documents")
    op.drop_index("ix_vector_documents_collection", table_name="vector_documents")
    op.drop_table("vector_documents")
