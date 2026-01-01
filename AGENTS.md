# Repository Guidelines

## Project Structure & Module Organization
- `src/roomeq/`: app entry (`app.py`), CLI shim (`__main__.py`), and packages for domains.
  - `ui/`: PyQt6 wizard (`ui/wizard.py`), page views under `ui/pages/`, and custom widgets in `ui/widgets/`.
  - `core/`: DSP, measurement, sweep generation, device I/O, EQ optimizer, and export to RME formats.
  - `utils/` and `session/`: shared helpers and state handling.
- `tests/`: pytest suite covering DSP, devices, sweeps, analysis, and export behavior.
- `pyproject.toml`: dependency list, tool configs (pytest, ruff, mypy), entry points; `README.md` for quick start.

## Setup, Build, and Local Run
- Create env and install dev deps: `python -m venv venv && source venv/bin/activate && pip install -e ".[dev]"`.
- Launch the app: `roomeq` or `python -m roomeq` (loads the PyQt wizard).
- Keep `pythonpath` as configured by pytest (`pythonpath = ["src"]`); run commands from repo root.

## Coding Style & Naming Conventions
- Python 3.11+, ruff with 100-char lines and `E,F,W,I,N,UP,B,C4` rules; run `ruff check src tests` before push.
- Type hints preferred; mypy configured to warn on `Any` and unused configs; run `mypy src tests` for changes touching core logic.
- Use snake_case for modules, functions, and variables; CamelCase for classes; prefer descriptive names for DSP and UI elements.

## Testing Guidelines
- Framework: pytest with verbose, short tracebacks (`-v --tb=short`). Test paths live in `tests/` with `test_*.py` naming.
- Add unit tests for new DSP methods, UI logic, and file exports; reuse fixtures in `tests/conftest.py` and favor deterministic sample data over hardware dependencies.
- Coverage: aim to keep or improve `pytest --cov=roomeq`; document gaps if unavoidable.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject lines; group related changes (logic, tests, UI) together. Include brief body when behavior changes.
- PRs: describe scope, user impact, and testing performed; link issues. For UI changes, attach before/after screenshots or a short clip of the wizard flow.
- Keep changelog notes in descriptions when you adjust user-facing behavior (measurement flow, export format, device detection).

## Security & Configuration Notes
- Do not commit API keys, audio captures, or hardware-specific calibration files; prefer `.env` and local paths.
- macOS-only app: verify PyQt widgets on macOS 11+; avoid platform-specific file dialogs unless required.
