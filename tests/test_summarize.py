"""Test the summarize CLI doen't crash on basic inputs"""

from typer.testing import CliRunner

from llm_quickstart.summarize import app

runner = CliRunner()


def test_summarize_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Summarise" in result.output


def test_summarize_missing_file():
    result = runner.invoke(app, ["nonexistent.txt"])
    assert result.exit_code != 0
    assert "Error: File not found" in result.output
