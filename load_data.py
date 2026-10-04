import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile

import polars as pl

ROOT_DIR = Path(__file__).resolve().parent
CSV_PATH = ROOT_DIR / "cell-count.csv"
DATABASE_PATH = ROOT_DIR / "cell_count.db"
CELL_POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]

SQL_SCHEMA_DEFINITIONS = """
CREATE TABLE subjects (
    subject   TEXT PRIMARY KEY NOT NULL CHECK (length(trim(subject)) > 0),
    project   TEXT NOT NULL CHECK (length(trim(project)) > 0),
    condition TEXT NOT NULL CHECK (length(trim(condition)) > 0),
    age       INTEGER NOT NULL CHECK (typeof(age) = 'integer' AND age >= 0),
    sex       TEXT NOT NULL CHECK (sex IN ('M', 'F')),
    treatment TEXT NOT NULL CHECK (length(trim(treatment)) > 0),
    -- NULL = not applicable: healthy controls are untreated
    response  TEXT CHECK (response IN ('yes', 'no')),
    CHECK ((treatment = 'none') = (response IS NULL))
);

CREATE TABLE samples (
    sample                     TEXT PRIMARY KEY NOT NULL CHECK (length(trim(sample)) > 0),
    subject                    TEXT NOT NULL REFERENCES subjects(subject),
    sample_type                TEXT NOT NULL CHECK (length(trim(sample_type)) > 0),
    time_from_treatment_start  INTEGER NOT NULL
                               CHECK (typeof(time_from_treatment_start) = 'integer')
);

CREATE TABLE cell_counts (
    sample     TEXT NOT NULL REFERENCES samples(sample),
    population TEXT NOT NULL CHECK (
        population IN ('b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte')
    ),
    count      INTEGER NOT NULL CHECK (typeof(count) = 'integer' AND count >= 0),
    PRIMARY KEY (sample, population)
);
"""


def load_data_and_create_db(
    file_path: str | Path,
    sqlite_database_path: str | Path,
    *,
    cell_populations: list[str],
) -> int:
    """Validate a CSV and atomically replace the normalized SQLite database."""
    file_path, sqlite_database_path = Path(file_path), Path(sqlite_database_path)

    if len(cell_populations) != 5 or set(cell_populations) != set(CELL_POPULATIONS):
        raise ValueError("cell_populations must contain each of the five populations once")

    subject_columns = ["subject", "project", "condition", "age", "sex", "treatment", "response"]
    sample_columns = ["sample", "subject", "sample_type", "time_from_treatment_start"]
    column_types = dict.fromkeys(subject_columns + sample_columns, pl.String)
    column_types.update(
        dict.fromkeys(["age", "time_from_treatment_start", *cell_populations], pl.Int64)
    )
    df = pl.read_csv(file_path, schema_overrides=column_types)
    missing_columns = set(column_types) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing_columns))}")

    # Break the wide table into three DataFrames corresponding to the three tables
    subjects = df.select(subject_columns).unique(maintain_order=True)
    if subjects.height != df["subject"].n_unique():
        raise ValueError("Subject metadata is inconsistent across samples")
    if df["sample"].n_unique() != df.height:
        raise ValueError("Sample IDs must be unique")

    samples = df.select(sample_columns)
    counts = df.unpivot(
        on=cell_populations,
        index="sample",
        variable_name="population",
        value_name="count",
    )

    # Build beside the destination so a failed import leaves the previous database intact.
    with NamedTemporaryFile(
        prefix=f".{sqlite_database_path.stem}.",
        suffix=".db",
        dir=sqlite_database_path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        with closing(sqlite3.connect(temporary_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(SQL_SCHEMA_DEFINITIONS)
            conn.executemany(
                "INSERT INTO subjects VALUES (?,?,?,?,?,?,?)", subjects.iter_rows()
            )
            conn.executemany("INSERT INTO samples VALUES (?,?,?,?)", samples.iter_rows())
            conn.executemany("INSERT INTO cell_counts VALUES (?,?,?)", counts.iter_rows())
            conn.commit()
        temporary_path.replace(sqlite_database_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Written {df.height} rows to {sqlite_database_path}")
    return df.height


def main() -> None:
    load_data_and_create_db(
        file_path=CSV_PATH,
        sqlite_database_path=DATABASE_PATH,
        cell_populations=CELL_POPULATIONS,
    )


if __name__ == "__main__":
    main()
