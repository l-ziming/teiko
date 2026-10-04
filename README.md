# Immune cell analysis

SQLite stores the clinical trial data, Polars computes cell frequencies, and
Streamlit provides an interactive dashboard.

Repository: [l-ziming/teiko](https://github.com/l-ziming/teiko).

## Run in GitHub Codespaces

Open this repository in Codespaces and run these commands from its root:

```sh
make setup
make pipeline
make dashboard
```

- `setup` prepares uv, Python 3.14, and the dependencies pinned in `uv.lock`.
- `pipeline` loads `cell-count.csv` into `cell_count.db`, then runs Parts 2–4.
- `dashboard` starts Streamlit on port 8501 using that database.

In Codespaces, open the **Ports** tab and select **Open in Browser** for port
8501. If it is not listed, select **Forward a Port** and enter `8501`.

## Dashboard

[Open the local dashboard](http://localhost:8501) after running `make dashboard`.
In Codespaces, use the forwarded port's browser link instead.

The dashboard includes sample frequencies, response boxplots and statistics,
baseline counts, and the B cell average. Download CSV files from each table's
toolbar.

## Outputs

`make pipeline` creates `cell_count.db` and `response_boxplots.png` in the
repository root, and prints the frequency table, statistical results, baseline
counts, and B cell average. Run it again to reproduce these outputs.

For male melanoma responders at day 0 across all sample and treatment types,
the mean B cell count is **10206.15**.

The loader also runs directly without arguments:

```sh
.venv/bin/python load_data.py
```
