"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-09-17

Reflects db/models.py's non-vector tables as they exist on main today.
There is no existing production data (SQLite/Chroma were never deployed
with real users), so no data migration is needed here or in the two
migrations that follow -- this is a schema-only cutover.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial_schema"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "budget_entries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("amount_usd", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_budget_entries")),
    )
    op.create_table(
        "jobs",
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("job_id", name=op.f("pk_jobs")),
    )
    op.create_table(
        "research_questions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("topic", sa.String(), nullable=False),
        sa.Column("sub_questions", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_research_questions")),
    )
    op.create_table(
        "search_cache_entries",
        sa.Column("query_key", sa.String(), nullable=False),
        sa.Column("results_json", sa.JSON(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("query_key", name=op.f("pk_search_cache_entries")),
    )
    op.create_table(
        "conclusions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("finding_ids", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["question_id"], ["research_questions.id"],
            name=op.f("fk_conclusions_question_id_research_questions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conclusions")),
    )
    op.create_table(
        "pipeline_runs",
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("question_id", sa.Integer(), nullable=True),
        sa.Column("topic", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["question_id"], ["research_questions.id"],
            name=op.f("fk_pipeline_runs_question_id_research_questions"),
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_pipeline_runs")),
    )
    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.Integer(), nullable=False),
        sa.Column("url", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("retrieved_at", sa.DateTime(), nullable=False),
        sa.Column("raw_content", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["question_id"], ["research_questions.id"],
            name=op.f("fk_sources_question_id_research_questions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sources")),
    )
    op.create_table(
        "findings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("category", sa.String(), nullable=True),
        sa.Column("sub_question", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["question_id"], ["research_questions.id"],
            name=op.f("fk_findings_question_id_research_questions"),
        ),
        sa.ForeignKeyConstraint(
            ["source_id"], ["sources.id"], name=op.f("fk_findings_source_id_sources"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_findings")),
    )
    op.create_table(
        "pipeline_stage_states",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("attempts", sa.JSON(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"], ["pipeline_runs.run_id"],
            name=op.f("fk_pipeline_stage_states_run_id_pipeline_runs"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_pipeline_stage_states")),
    )
    op.create_table(
        "claim_groundings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.Integer(), nullable=False),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("grounded", sa.Boolean(), nullable=False),
        sa.Column("supporting_finding_id", sa.Integer(), nullable=True),
        sa.Column("objection", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["question_id"], ["research_questions.id"],
            name=op.f("fk_claim_groundings_question_id_research_questions"),
        ),
        sa.ForeignKeyConstraint(
            ["supporting_finding_id"], ["findings.id"],
            name=op.f("fk_claim_groundings_supporting_finding_id_findings"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_claim_groundings")),
    )
    op.create_table(
        "contradictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("finding_a_id", sa.Integer(), nullable=False),
        sa.Column("finding_b_id", sa.Integer(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["finding_a_id"], ["findings.id"], name=op.f("fk_contradictions_finding_a_id_findings"),
        ),
        sa.ForeignKeyConstraint(
            ["finding_b_id"], ["findings.id"], name=op.f("fk_contradictions_finding_b_id_findings"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_contradictions")),
    )


def downgrade() -> None:
    op.drop_table("contradictions")
    op.drop_table("claim_groundings")
    op.drop_table("pipeline_stage_states")
    op.drop_table("findings")
    op.drop_table("sources")
    op.drop_table("pipeline_runs")
    op.drop_table("conclusions")
    op.drop_table("search_cache_entries")
    op.drop_table("research_questions")
    op.drop_table("jobs")
    op.drop_table("budget_entries")
