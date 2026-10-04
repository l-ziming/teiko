# %% Part 2 Initial Analysis - Data Overview

import sqlite3
from contextlib import closing
from math import isfinite, sqrt
from pathlib import Path

import polars as pl
from matplotlib.figure import Figure
from scipy import stats

from load_data import CELL_POPULATIONS

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_formatting("NOTHING")
pl.Config(tbl_rows=20, float_precision=2)


def get_cell_frequencies(sqlite_database_path: str | Path) -> pl.DataFrame:
    # Per-sample relative frequencies, with the subject and sample metadata
    with closing(sqlite3.connect(Path(sqlite_database_path))) as conn:
        df_counts = pl.read_database("SELECT * FROM cell_counts", conn)
        df_samples = pl.read_database("SELECT * FROM samples", conn)
        df_subjects = pl.read_database("SELECT * FROM subjects", conn)

    df_joined = (
        df_counts
        .join(df_samples, on="sample")
        .join(df_subjects, on="subject")
    )

    return (
        df_joined
        .with_columns(
            pl.col("count").sum().over("sample").alias("total_count")
        )
        .with_columns(
            # A sample with no cells has an undefined relative frequency.
            pl.when(pl.col("total_count") > 0)
            .then(pl.col("count") / pl.col("total_count") * 100)
            .otherwise(None)
            .alias("percentage")
        )
        .sort(["sample", "population"])
    )


def get_frequency_summary(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(["sample", "total_count", "population", "count", "percentage"])


# %% Part 3.1 Statistical Analysis
def get_response_cohort(
    df: pl.DataFrame,
    *,
    target_condition: str = "melanoma",
    target_treatment: str = "miraclib",
    target_sample_type: str = "PBMC",
    by_subject: bool = True,
) -> pl.DataFrame:
    # The samples whose treatment response is being compared
    cohort_df = df.filter(
        (pl.col("condition") == target_condition) &
        (pl.col("treatment") == target_treatment) &
        (pl.col("sample_type") == target_sample_type) &
        (pl.col("response").is_in(["yes", "no"])) &
        pl.col("percentage").is_finite()
    )

    # Average repeated samples first so each subject contributes one observation.
    if by_subject:
        cohort_df = (
            cohort_df
            .group_by(["subject", "population", "response"])
            .agg(pl.col("percentage").mean())
        )

    return cohort_df.sort(["population", "response", "subject"])


def perform_statistical_test(
    df: pl.DataFrame,
    *,
    target_condition: str = "melanoma",
    target_treatment: str = "miraclib",
    target_sample_type: str = "PBMC",
    alpha: float = 0.05,
    by_subject: bool = True,
) -> pl.DataFrame:
    """Two-sided Mann–Whitney U tests, with Benjamini–Hochberg correction."""
    analysis_df = get_response_cohort(
        df,
        target_condition=target_condition,
        target_treatment=target_treatment,
        target_sample_type=target_sample_type,
        by_subject=by_subject,
    )

    results = []

    for pop in CELL_POPULATIONS:
        pop_data = analysis_df.filter(pl.col("population") == pop)
        responders = pop_data.filter(pl.col("response") == "yes")["percentage"]
        non_responders = pop_data.filter(pl.col("response") == "no")["percentage"]
        n_yes, n_no = len(responders), len(non_responders)

        p_value, cohens_d = None, None
        if n_yes and n_no:
            p_value = float(stats.mannwhitneyu(
                responders.to_numpy(), non_responders.to_numpy(), alternative="two-sided"
            ).pvalue)
            if not isfinite(p_value):
                p_value = None

        if n_yes > 1 and n_no > 1:
            pooled_variance = (
                (n_yes - 1) * responders.var(ddof=1)
                + (n_no - 1) * non_responders.var(ddof=1)
            ) / (n_yes + n_no - 2)
            if pooled_variance > 0:
                # Positive means the population is more abundant in responders.
                cohens_d = (responders.mean() - non_responders.mean()) / sqrt(pooled_variance)

        results.append({
            "population": pop,
            "n_responders": n_yes,
            "n_non_responders": n_no,
            "median_responders": responders.median(),
            "median_non_responders": non_responders.median(),
            "cohens_d": cohens_d,
            "p_value": p_value,
        })

    # Only valid tests enter the correction; unavailable comparisons stay null.
    valid_indices = [i for i, row in enumerate(results) if row["p_value"] is not None]
    p_adjusted = [None] * len(results)
    if valid_indices:
        corrected = stats.false_discovery_control(
            [results[i]["p_value"] for i in valid_indices], method="bh"
        )
        for i, value in zip(valid_indices, corrected):
            p_adjusted[i] = float(value)

    df_results = pl.DataFrame(results, schema={
        "population": pl.String,
        "n_responders": pl.Int64,
        "n_non_responders": pl.Int64,
        "median_responders": pl.Float64,
        "median_non_responders": pl.Float64,
        "cohens_d": pl.Float64,
        "p_value": pl.Float64,
    })

    return (
        df_results
        .with_columns(pl.Series("p_adjusted", p_adjusted, dtype=pl.Float64))
        .with_columns(
            (pl.col("median_responders") - pl.col("median_non_responders")).alias("median_diff"),
            (pl.col("p_adjusted") < alpha).fill_null(False).alias("is_significant")
        )
        .select(
            "population", "n_responders", "n_non_responders",
            "median_responders", "median_non_responders", "median_diff",
            "cohens_d", "p_value", "p_adjusted", "is_significant"
        )
        .sort("p_adjusted", nulls_last=True)
    )


# %% Part 3.2 Plot
def plot_response_frequencies(df: pl.DataFrame, *, by_subject: bool = True) -> Figure:
    cohort = get_response_cohort(df, by_subject=by_subject)
    figure = Figure(figsize=(14, 4.5), layout="constrained")
    axes = figure.subplots(1, len(CELL_POPULATIONS), sharey=True)

    for ax, population in zip(axes, CELL_POPULATIONS):
        pop_data = cohort.filter(pl.col("population") == population)
        tick_labels = []
        for position, response, label, color in [
            (1, "yes", "Responders", "#297d9b"),
            (2, "no", "Non-responders", "#a6b2bd"),
        ]:
            values = pop_data.filter(pl.col("response") == response)["percentage"].to_list()
            tick_labels.append(f"{label}\n(n={len(values)})")
            if values:
                box = ax.boxplot(
                    [values], positions=[position], widths=0.6, patch_artist=True,
                    medianprops={"color": "#172c3c", "linewidth": 1.5},
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
                )
                box["boxes"][0].set_facecolor(color)

        ax.set_title(population.replace("_", " ").title())
        ax.set_xticks([1, 2], tick_labels, fontsize=8)
        ax.set_xlim(0.4, 2.6)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        if pop_data.is_empty():
            ax.text(0.5, 0.5, "No eligible data", ha="center", transform=ax.transAxes)

    axes[0].set_ylabel("Relative frequency (%)")
    unit = "subject averages" if by_subject else "samples"
    figure.suptitle(f"Melanoma, miraclib, PBMC ({unit})")
    return figure


# %% Part 4 Data Subset Analysis
def get_baseline_samples(sqlite_database_path: str | Path) -> pl.DataFrame:
    query = """
        SELECT samples.sample, samples.subject, subjects.project,
               subjects.response, subjects.sex, samples.sample_type,
               subjects.treatment, samples.time_from_treatment_start
        FROM samples
        JOIN subjects ON subjects.subject = samples.subject
        WHERE subjects.condition = 'melanoma'
          AND subjects.treatment = 'miraclib'
          AND samples.sample_type = 'PBMC'
          AND samples.time_from_treatment_start = 0
        ORDER BY samples.sample
    """
    with closing(sqlite3.connect(Path(sqlite_database_path))) as conn:
        return pl.read_database(query, conn, schema_overrides={
            "sample": pl.String, "subject": pl.String, "project": pl.String,
            "response": pl.String, "sex": pl.String, "sample_type": pl.String,
            "treatment": pl.String, "time_from_treatment_start": pl.Int64,
        })


def get_baseline_summary(samples: pl.DataFrame) -> dict[str, pl.DataFrame]:
    return {
        "project": samples.group_by("project").len(name="sample_count").sort("project"),
        "response": samples.group_by("response").agg(
            pl.col("subject").n_unique().alias("subject_count")
        ).sort("response"),
        "sex": samples.group_by("sex").agg(
            pl.col("subject").n_unique().alias("subject_count")
        ).sort("sex"),
    }


def get_average_baseline_b_cells(sqlite_database_path: str | Path) -> float | None:
    # All sample types and treatments are eligible for this specific question.
    query = """
        SELECT AVG(cell_counts.count)
        FROM cell_counts
        JOIN samples ON samples.sample = cell_counts.sample
        JOIN subjects ON subjects.subject = samples.subject
        WHERE subjects.condition = 'melanoma'
          AND subjects.sex = 'M'
          AND subjects.response = 'yes'
          AND samples.time_from_treatment_start = 0
          AND cell_counts.population = 'b_cell'
    """
    with closing(sqlite3.connect(Path(sqlite_database_path))) as conn:
        return conn.execute(query).fetchone()[0]


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent
    db_path = ROOT_DIR / "cell_count.db"
    cell_frequencies = get_cell_frequencies(db_path)
    summary = get_frequency_summary(cell_frequencies)
    frequency_totals = (
        cell_frequencies.filter(pl.col("total_count") > 0)
        .group_by("sample").agg(pl.col("percentage").sum())
    )
    if not frequency_totals.is_empty():
        assert (frequency_totals["percentage"] - 100).abs().max() < 1e-9, "percentages do not sum to 100"

    print(summary)
    print(f"{summary.height} rows across {summary['sample'].n_unique()} samples")
    print("Zero-total samples have null percentages and are excluded from comparisons.")

    with pl.Config(tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print("\nResponders vs non-responders, subject-level (primary):")
        print(perform_statistical_test(cell_frequencies))
        print("\nResponders vs non-responders, sample-level (repeated samples):")
        print(perform_statistical_test(cell_frequencies, by_subject=False))

    figure = plot_response_frequencies(cell_frequencies)
    figure.savefig(ROOT_DIR / "response_boxplots.png", dpi=150)
    print("\nSaved response_boxplots.png")

    baseline_samples = get_baseline_samples(db_path)
    print(f"\nBaseline melanoma / miraclib / PBMC: {baseline_samples.height} samples")
    print(baseline_samples)
    for name, table in get_baseline_summary(baseline_samples).items():
        print(f"\nBaseline counts by {name}:")
        print(table)

    average = get_average_baseline_b_cells(db_path)
    average_text = f"{average:.2f}" if average is not None else "No eligible samples"
    print(f"\nAverage baseline B cells, male melanoma responders (all treatments/types): {average_text}")
