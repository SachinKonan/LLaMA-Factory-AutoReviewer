"""Local pytest config for the scoring integration test.

Registers the ``gpu`` marker so ``-m gpu`` works without touching the
top-level pyproject (the LF test suite doesn't define it).
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: needs a CUDA GPU + the paperlenstraininfer venv (vLLM scorer)"
    )
