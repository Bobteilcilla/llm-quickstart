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


def test_new_files_exist():
    assert os.path.exists(
        "compare_with_systemprompt.py"
    ), "compare_with_systemprompt.py does not exist"
    assert os.path.exists("extract_openapi.py"), "extract_openapi.py does not exist"
    assert os.path.exists("extract_anthropic.py"), "extract_anthropic.py does not exist"
    assert os.path.exists("stream_anthropic.py"), "stream_anthropic.py does not exist"
    assert os.path.exists("stream_openai.py"), "stream_openai.py does not exist"
