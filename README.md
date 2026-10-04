# Immune cell analysis

SQLite stores the clinical trial data, Polars computes cell frequencies, and
Streamlit provides an interactive dashboard.

## Run

Python 3.14 or later is required.

```sh
uv sync
source .venv/bin/activate
python load_data.py
python analyze_data.py
streamlit run dashboard.py
```

`python load_data.py` reads `cell-count.csv` and creates `cell_count.db` in the
repository root without arguments. Run it again to reload the data. Paths are
relative to the script, so the loader also works from another working directory.
The dashboard initializes the database automatically if it is missing.

`analyze_data.py` prints the frequency summary, response comparisons, baseline
counts, and the requested B cell average. It also saves `response_boxplots.png`.
The dashboard provides filters and the full summary table. Download CSV files
using each table's toolbar.
