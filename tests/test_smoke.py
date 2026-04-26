"""One smoke test so CI has something real to run from day one."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_has_readme() -> None:
    assert (REPO_ROOT / "README.md").is_file()


def test_repo_has_openapi() -> None:
    assert (REPO_ROOT / "openapi.yaml").is_file()


def test_repo_has_pyproject() -> None:
    assert (REPO_ROOT / "pyproject.toml").is_file()
