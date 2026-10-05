# Q-Edge task runner. Install `just` with: uv tool install rust-just

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Run the fast test suite
test:
    uv run pytest -q -m "not slow"

# Run every test, including the slow 4K end-to-end tests
test-all:
    uv run pytest -q

# Lint, format check and type check
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src app

# Auto-format and auto-fix lint issues
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Test coverage report for the core package
cov:
    uv run pytest -q --cov=q_edge --cov-report=term-missing

# Launch the Streamlit app
app:
    uv run streamlit run app/streamlit_app.py

# Run the end-to-end demo and benchmark
demo:
    uv run python scripts/demo.py
