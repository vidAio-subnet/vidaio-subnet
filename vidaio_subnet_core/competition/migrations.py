"""Explicit, idempotent bootstrap for the competition SQLite schema."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Engine, inspect, text
from sqlalchemy.schema import CreateIndex, CreateTable

from .models import CompetitionBase, CompetitionSchemaMigration


SCHEMA_VERSION = 1
BASELINE_NAME = "initial_competition_schema"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def apply_competition_migrations(engine: Engine) -> None:
    """Create the current pre-production schema as one squashed baseline."""

    if engine.dialect.name != "sqlite":
        raise ValueError("competition persistence currently supports SQLite only")

    migration_table = CompetitionSchemaMigration.__table__
    with engine.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys=ON"))

        existing_tables = set(inspect(connection).get_table_names())
        competition_tables = {
            table.name
            for table in CompetitionBase.metadata.sorted_tables
            if table.name != migration_table.name
        }
        existing_competition_tables = existing_tables & competition_tables
        if (
            migration_table.name not in existing_tables
            and existing_competition_tables
        ):
            raise RuntimeError(
                "competition tables exist without the squashed schema baseline; "
                "recreate the pre-production SQLite database"
            )

        connection.execute(CreateTable(migration_table, if_not_exists=True))
        applied = connection.execute(
            text(
                "SELECT version, name FROM competition_schema_migrations "
                "ORDER BY version"
            )
        ).all()
        if applied:
            expected = [(SCHEMA_VERSION, BASELINE_NAME)]
            if [tuple(row) for row in applied] != expected:
                raise RuntimeError(
                    "competition database uses the retired pre-production "
                    "migration history; recreate it for the squashed schema baseline"
                )
            return

        for table in CompetitionBase.metadata.sorted_tables:
            if table.name == migration_table.name:
                continue
            connection.execute(CreateTable(table, if_not_exists=True))

        existing_indexes = {
            (table_name, index["name"])
            for table_name in inspect(connection).get_table_names()
            for index in inspect(connection).get_indexes(table_name)
        }
        for table in CompetitionBase.metadata.sorted_tables:
            for index in table.indexes:
                if (table.name, index.name) not in existing_indexes:
                    connection.execute(CreateIndex(index, if_not_exists=True))

        connection.execute(
            text(
                "INSERT INTO competition_schema_migrations"
                "(version, name, applied_at) "
                "VALUES (:version, :name, :applied_at)"
            ),
            {
                "version": SCHEMA_VERSION,
                "name": BASELINE_NAME,
                "applied_at": _utc_now(),
            },
        )
