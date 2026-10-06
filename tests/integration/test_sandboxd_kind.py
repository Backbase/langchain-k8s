"""Integration tests for SandboxdBackend against a Kind cluster.

``scripts/kind-setup.sh`` applies ``k8s/sandboxd-template.yaml`` and
``k8s/sandboxd-warmpool.yaml``. The tests skip when ``sandboxd-pool`` is
absent. Shell ``execute`` goes over gRPC. File tools go over the REST
filesystem API, so they work on the stock image, which has no ``python3``.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Generator

import pytest
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_k8s import SandboxdBackend
from tests.conftest import FakeToolModel

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
def sandboxd() -> Generator[SandboxdBackend]:
    """One sandboxd pod for the module. Skips when the pool or grpc extra is missing."""
    if not _sandboxd_pool_present():
        pytest.skip("sandboxd-pool is not installed. Apply k8s/sandboxd-template.yaml and k8s/sandboxd-warmpool.yaml.")
    pytest.importorskip("grpc", reason="sandboxd execute needs the k8s-agent-sandbox grpc extra")

    sb = SandboxdBackend(
        warmpool_name=WARMPOOL,
        namespace=NAMESPACE,
        enable_capture_offload=True,
    )
    yield sb
    if sb._started:
        sb.stop()


class TestSandboxdExecute:
    def test_echo(self, sandboxd: SandboxdBackend) -> None:
        resp = sandboxd.execute("echo hello-sandboxd")
        assert resp.exit_code == 0
        assert "hello-sandboxd" in resp.output

    def test_exit_code(self, sandboxd: SandboxdBackend) -> None:
        resp = sandboxd.execute("sh -c 'exit 7'")
        assert resp.exit_code == 7


class TestSandboxdFileTools:
    def test_python3_is_absent(self, sandboxd: SandboxdBackend) -> None:
        """The stock image has no interpreter. Native file tools must not need one."""
        resp = sandboxd.execute("python3 --version || echo NO_PYTHON3")
        assert resp.exit_code == 0
        assert "NO_PYTHON3" in resp.output

    def test_native_file_round_trip(self, sandboxd: SandboxdBackend) -> None:
        written = sandboxd.write("/workspace/src/a.py", "alpha\n")
        assert written.error is None
        assert written.path == "/workspace/src/a.py"

        read = sandboxd.read("/workspace/src/a.py")
        assert read.error is None
        assert read.file_data is not None
        assert read.file_data["content"] == "alpha\n"

        edited = sandboxd.edit("/workspace/src/a.py", "alpha", "beta")
        assert edited.error is None
        assert edited.occurrences == 1

        listed = sandboxd.ls("/workspace/src")
        assert listed.error is None
        assert listed.entries is not None
        assert any(entry["path"] == "/workspace/src/a.py" and entry["is_dir"] is False for entry in listed.entries)

        found = sandboxd.glob("**/*.py", path="/workspace/src")
        assert found.error is None
        assert found.matches is not None
        assert any(item["path"] == "/workspace/src/a.py" for item in found.matches)

        grepped = sandboxd.grep("beta", path="/workspace/src", glob="*.py")
        assert grepped.error is None
        assert grepped.matches is not None
        assert grepped.matches[0]["path"] == "/workspace/src/a.py"
        assert grepped.matches[0]["line"] == 1

        seen = sandboxd.execute("cat /workspace/src/a.py")
        assert seen.exit_code == 0
        assert "beta" in seen.output

        uploaded = sandboxd.upload_files([("/workspace/src/b.txt", b"uploaded")])
        assert uploaded[0].error is None
        downloaded = sandboxd.download_files(["/workspace/src/b.txt"])
        assert downloaded[0].error is None
        assert downloaded[0].content == b"uploaded"

        removed = sandboxd.delete("/workspace/src/a.py")
        assert removed.error is None
        gone = sandboxd.read("/workspace/src/a.py")
        assert gone.error == "file_not_found"

    def test_offload_pointer_is_readable(self, sandboxd: SandboxdBackend) -> None:
        """A large execute leaves a file that read_file can open through the REST API."""
        model = FakeToolModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "execute",
                            "args": {"command": "echo offload-marker-sandboxd"},
                            "id": "call_sandboxd_offload",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Saved."),
            ]
        )
        agent = create_agent(
            model,
            middleware=[FilesystemMiddleware(backend=sandboxd, tool_token_limit_before_evict=1)],
        )
        result = agent.invoke({"messages": [HumanMessage(content="Run a command")]})

        tool_msgs = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        assert tool_msgs
        content = str(tool_msgs[0].content)
        assert "read_file" in content
        match = re.search(r"(/large_tool_results/\S+)", content)
        assert match is not None
        pointer = match.group(1).rstrip(".")

        saved = sandboxd.read(pointer)
        assert saved.error is None
        assert saved.file_data is not None
        assert "offload-marker-sandboxd" in saved.file_data["content"]
