import os

from dotenv import load_dotenv


def test_env_loads():
    load_dotenv()
    assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY not found in environment variables"
    assert "ANTHROPIC_API_KEY" in os.environ, "ANTHROPIC_API_KEY not found in environment variables"


def test_sdk_imports():
    import anthropic
    import openai

    assert openai.__version__
    assert anthropic.__version__
