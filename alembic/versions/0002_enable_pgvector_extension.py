"""enable pgvector extension

Revision ID: 0002_enable_pgvector
Revises: 0001_initial_schema
Create Date: 2026-09-17

Isolated as its own step, ahead of the migration that creates any pgvector
column (0003), so the extension unambiguously exists before it's needed --
running both in one migration works today but couples an environment-level
concern (extension availability) to a schema-level one.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0002_enable_pgvector"
down_revision: Union[str, Sequence[str], None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector;")
