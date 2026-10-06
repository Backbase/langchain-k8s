"""Integration tests for the sandboxd runtime against a Kind cluster.

``kind-setup.sh`` does not apply the sandboxd manifests. Apply them first::

    kubectl apply -f k8s/sandboxd-template.yaml
    kubectl apply -f k8s/sandboxd-warmpool.yaml

The tests skip when ``sandboxd-pool`` is absent. Shell ``execute`` is expected
to work. The stock ``sandboxd:latest-main`` image has no ``python3``, so
``write`` fails; that assertion is the canary for the image growing a Python
interpreter.
"""

from __future__ import annotations

import subprocess
from collections.abc import Generator

import pytest

from langchain_k8s import KubernetesSandbox

pytestmark = pytest.mark.integration

WARMPOOL = "sandboxd-pool"
NAMESPACE = "agent-sandbox-system"


def _sandboxd_pool_present() -> bool:
    """Return whether the sandboxd warm pool is installed in the current context."""
    try:
        result = subprocess.run(
            ["kubectl", "get", "sandboxwarmpool", WARMPOOL, "-n", NAMESPACE, "-o", "name"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and WARMPOOL in result.stdout


@pytest.fixture(scope="module")
def sandboxd() -> Generator[KubernetesSandbox]:
    """One sandboxd pod for the module. Skips when the pool or grpc extra is missing."""
    if not _sandboxd_pool_present():
        pytest.skip("sandboxd-pool is not installed. Apply k8s/sandboxd-template.yaml and k8s/sandboxd-warmpool.yaml.")
    pytest.importorskip("grpc", reason="sandboxd execute needs the k8s-agent-sandbox grpc extra")
    from k8s_agent_sandbox.models import SandboxdPodTunnelConnectionConfig

    sb = KubernetesSandbox(
        warmpool_name=WARMPOOL,
        namespace=NAMESPACE,
        connection_config=SandboxdPodTunnelConnectionConfig(),
    )
    yield sb
    if sb._started:
        sb.stop()


class TestSandboxdExecute:
    def test_echo(self, sandboxd: KubernetesSandbox) -> None:
        resp = sandboxd.execute("echo hello-sandboxd")
        assert resp.exit_code == 0
        assert "hello-sandboxd" in resp.output

    def test_exit_code(self, sandboxd: KubernetesSandbox) -> None:
        resp = sandboxd.execute("sh -c 'exit 7'")
        assert resp.exit_code == 7


class TestSandboxdFileTools:
    def test_python3_is_absent(self, sandboxd: KubernetesSandbox) -> None:
        """The stock image has no interpreter, so the file-tool scripts cannot run."""
        resp = sandboxd.execute("python3 --version || echo NO_PYTHON3")
        assert resp.exit_code == 0
        assert "NO_PYTHON3" in resp.output

    def test_write_fails_without_python3(self, sandboxd: KubernetesSandbox) -> None:
        result = sandboxd.write("/workspace/it-sandboxd.txt", "hello")
        assert result.error is not None
        assert "python3" in result.error
