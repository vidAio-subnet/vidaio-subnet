"""Explicit, idempotent bootstrap for the competition SQLite schema."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Engine, inspect, text
from sqlalchemy.schema import CreateIndex, CreateTable

from .models import CompetitionBase, CompetitionSchemaMigration


SCHEMA_VERSION = 2
BASELINE_NAME = "initial_competition_schema"
SANDBOX_PREFERENCES_MIGRATION_NAME = "contender_sandbox_preferences"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def apply_competition_migrations(engine: Engine) -> None:
    """Create or upgrade the competition-only schema."""

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
            applied_rows = [tuple(row) for row in applied]
            baseline = [(1, BASELINE_NAME)]
            current = [
                *baseline,
                (SCHEMA_VERSION, SANDBOX_PREFERENCES_MIGRATION_NAME),
            ]
            if applied_rows == baseline:
                columns = {
                    column["name"]
                    for column in inspect(connection).get_columns(
                        "contender_metadata"
                    )
                }
                if "sandbox_gpu" not in columns:
                    connection.execute(
                        text(
                            "ALTER TABLE contender_metadata "
                            "ADD COLUMN sandbox_gpu VARCHAR(64)"
                        )
                    )
                if "sandbox_cpus" not in columns:
                    connection.execute(
                        text(
                            "ALTER TABLE contender_metadata "
                            "ADD COLUMN sandbox_cpus INTEGER"
                        )
                    )
                connection.execute(
                    text(
                        "INSERT INTO competition_schema_migrations"
                        "(version, name, applied_at) "
                        "VALUES (:version, :name, :applied_at)"
                    ),
                    {
                        "version": SCHEMA_VERSION,
                        "name": SANDBOX_PREFERENCES_MIGRATION_NAME,
                        "applied_at": _utc_now(),
                    },
                )
                return
            if applied_rows != current:
                raise RuntimeError(
                    "competition database uses the retired pre-production "
                    "migration history; recreate it for the squashed schema baseline"
                )
            columns = {
                column["name"]
                for column in inspect(connection).get_columns("contender_metadata")
            }
            if not {"sandbox_gpu", "sandbox_cpus"} <= columns:
                raise RuntimeError(
                    "competition schema migration history is ahead of its columns"
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

        applied_at = _utc_now()
        connection.execute(
            text(
                "INSERT INTO competition_schema_migrations"
                "(version, name, applied_at) "
                "VALUES (:baseline_version, :baseline_name, :applied_at), "
                "(:version, :name, :applied_at)"
            ),
            {
                "baseline_version": 1,
                "baseline_name": BASELINE_NAME,
                "version": SCHEMA_VERSION,
                "name": SANDBOX_PREFERENCES_MIGRATION_NAME,
                "applied_at": applied_at,
            },
        )
