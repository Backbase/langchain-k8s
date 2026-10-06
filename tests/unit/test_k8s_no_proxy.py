"""The installed kubernetes client must honour proxy env vars.

kubernetes 34.1.0-35.0.0 overwrote ``no_proxy`` with ``None`` in
``Configuration.__init__`` (https://github.com/kubernetes-client/python/issues/2460),
sending port-forward traffic through ``HTTPS_PROXY``. The ``kubernetes>=36.0.3``
floor in ``pyproject.toml`` is what keeps that out.
"""

from __future__ import annotations

import pytest
from kubernetes.client.configuration import Configuration

_PROXY_VARS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy")


@pytest.fixture(autouse=True)
def _clean_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _PROXY_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("var", ["NO_PROXY", "no_proxy"])
def test_no_proxy_env_is_honoured(monkeypatch: pytest.MonkeyPatch, var: str) -> None:
    monkeypatch.setenv(var, "127.0.0.1,localhost")

    assert Configuration().no_proxy == "127.0.0.1,localhost"


def test_proxy_and_no_proxy_survive_together(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8888")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    cfg = Configuration()

    assert cfg.proxy == "http://proxy:8888"
    assert cfg.no_proxy == "127.0.0.1,localhost"
