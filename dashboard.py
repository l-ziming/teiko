import sqlite3

import matplotlib.pyplot as plt
import polars as pl
import streamlit as st

from analyze_data import (
    get_average_baseline_b_cells,
    get_baseline_samples,
    get_baseline_summary,
    get_cell_frequencies,
    get_frequency_summary,
    get_response_cohort,
    perform_statistical_test,
    plot_response_frequencies,
)
from load_data import DATABASE_PATH
from load_data import main as load_data


@st.cache_data
def read_dashboard_data(database_modified_ns: int) -> tuple[pl.DataFrame, pl.DataFrame, float | None]:
    # The modification time invalidates cached results after the database is reloaded.
    return (
        get_cell_frequencies(DATABASE_PATH),
        get_baseline_samples(DATABASE_PATH),
        get_average_baseline_b_cells(DATABASE_PATH),
    )


def main() -> None:
    st.set_page_config(page_title="Immune cell analysis", page_icon="🧬", layout="wide")
    st.title("Immune cell analysis")
    st.caption("Loblaw Bio")

    try:
        if not DATABASE_PATH.exists():
            load_data()
        frequencies, baseline_samples, average_b_cells = read_dashboard_data(
            DATABASE_PATH.stat().st_mtime_ns
        )
    except (OSError, sqlite3.Error, ValueError) as error:
        st.error(f"Could not load data: {error}")
        st.stop()

    overview, response_comparison, baseline = st.tabs(
        ["Overview", "Response", "Baseline"]
    )

    with overview:
        st.subheader("Sample frequencies")
        st.caption("Frequency (%) = population count / sample total × 100")
        filtered = frequencies
        with st.expander("Filter samples"):
            st.caption("Overview only.")
            columns = st.columns(3)
            filters = [
                ("condition", "Condition"),
                ("treatment", "Treatment"),
                ("sample_type", "Sample type"),
                ("project", "Project"),
                ("response", "Response"),
                ("sex", "Sex"),
                ("time_from_treatment_start", "Treatment day"),
            ]
            for index, (field, label) in enumerate(filters):
                options = frequencies[field].drop_nulls().unique().sort().to_list()
                if frequencies[field].null_count():
                    options.append("Not recorded")
                selected = columns[index % 3].selectbox(label, ["All", *options])
                if selected == "Not recorded":
                    filtered = filtered.filter(pl.col(field).is_null())
                elif selected != "All":
                    filtered = filtered.filter(pl.col(field) == selected)

        sample_count, subject_count, population_count = st.columns(3)
        sample_count.metric("Samples", f"{filtered['sample'].n_unique():,}")
        subject_count.metric("Subjects", f"{filtered['subject'].n_unique():,}")
        population_count.metric("Cell populations", filtered["population"].n_unique())

        summary = get_frequency_summary(filtered)
        if summary.is_empty():
            st.info("No matching samples.")
        else:
            st.dataframe(
                summary,
                hide_index=True,
                column_config={
                    "percentage": st.column_config.NumberColumn("percentage", format="%.2f%%"),
                },
            )
            st.caption(f"{summary.height:,} rows")

    with response_comparison:
        st.subheader("Responders vs. non-responders")
        st.caption("Melanoma, miraclib, PBMC, all time points")
        cohort = get_response_cohort(frequencies, by_subject=False)
        responders = cohort.filter(pl.col("response") == "yes")
        non_responders = cohort.filter(pl.col("response") == "no")
        responder_count, non_responder_count, sample_count = st.columns(3)
        responder_count.metric("Responders", responders["subject"].n_unique())
        non_responder_count.metric("Non-responders", non_responders["subject"].n_unique())
        sample_count.metric("PBMC samples", f"{cohort['sample'].n_unique():,}")

        analysis_unit = st.radio(
            "Analysis unit",
            ["Subject averages", "All samples (exploratory)"],
            horizontal=True,
        )
        by_subject = analysis_unit == "Subject averages"
        if by_subject:
            st.caption("One mean frequency per subject and population.")
        else:
            st.warning("Repeated patient samples are correlated; results are exploratory.")

        if responders.is_empty() or non_responders.is_empty():
            st.info("Both response groups are required.")
        else:
            figure = plot_response_frequencies(frequencies, by_subject=by_subject)
            st.pyplot(figure)
            plt.close(figure)

            results = perform_statistical_test(frequencies, by_subject=by_subject)
            significant = results.filter(pl.col("is_significant"))["population"].to_list()
            if significant:
                st.success("Significant after FDR correction: " + ", ".join(significant))
            else:
                st.info("No significant differences after FDR correction.")
            st.caption("Two-sided Mann–Whitney U; BH FDR (5 tests); adjusted p < 0.05; pp = percentage points")
            st.dataframe(
                results,
                hide_index=True,
                column_config={
                    "population": "Population",
                    "n_responders": "Responders (n)",
                    "n_non_responders": "Non-responders (n)",
                    "median_responders": st.column_config.NumberColumn("Responder median (%)", format="%.2f"),
                    "median_non_responders": st.column_config.NumberColumn("Non-responder median (%)", format="%.2f"),
                    "median_diff": st.column_config.NumberColumn("Median difference (pp)", format="%.2f"),
                    "cohens_d": st.column_config.NumberColumn("Cohen's d", format="%.3f"),
                    "p_value": st.column_config.NumberColumn("p-value", format="%.4g"),
                    "p_adjusted": st.column_config.NumberColumn("FDR-adjusted p", format="%.4g"),
                    "is_significant": "Significant",
                },
            )
            st.caption("Associations only; prediction needs baseline data and independent validation.")

    with baseline:
        st.subheader("Baseline samples")
        st.caption("Melanoma, miraclib, PBMC, day 0")
        sample_count, subject_count = st.columns(2)
        sample_count.metric("Baseline samples", f"{baseline_samples.height:,}")
        subject_count.metric("Subjects", f"{baseline_samples['subject'].n_unique():,}")

        if baseline_samples.is_empty():
            st.info("No baseline samples.")
        else:
            baseline_summary = get_baseline_summary(baseline_samples)
            for column, (key, table) in zip(st.columns(3), baseline_summary.items()):
                with column:
                    label = "Samples" if key == "project" else "Subjects"
                    st.markdown(f"**{label} by {key}**")
                    st.dataframe(table, hide_index=True)
            st.caption("Projects: samples. Response and sex: distinct subjects.")
            with st.expander("View baseline samples"):
                st.dataframe(baseline_samples, hide_index=True)

        st.divider()
        st.subheader("Mean baseline B cells")
        st.metric(
            "Melanoma, male responders, day 0",
            f"{average_b_cells:.2f}" if average_b_cells is not None else "No matches",
        )
        st.caption("Per sample, all sample types and treatments")


if __name__ == "__main__":
    main()
