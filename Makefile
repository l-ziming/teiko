UV := $(shell command -v uv 2>/dev/null || printf '%s' '$(CURDIR)/.tools/uv')

.PHONY: setup pipeline dashboard

setup:
	@test -x "$(UV)" || curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$(CURDIR)/.tools" sh
	"$(UV)" sync --locked --inexact

pipeline:
	.venv/bin/python load_data.py
	.venv/bin/python analyze_data.py

dashboard:
	.venv/bin/python -m streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
