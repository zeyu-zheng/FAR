"""Where an opencode agent runs: on this machine, or in a Docker container.

Both runtimes take the same opencode command line and the same per-attempt
directories, and differ only in how the process is started, stopped, and read
back:

    <stage_dir>/workspace/          the agent's working directory (/workspace in Docker)
    <stage_dir>/.opencode/NNN/      opencode's own store for attempt NNN

The store holds the session database (with its WAL and attachments) and outlives
the process or container, so a session can be exported after a crash or a
cancel. Credentials never go into it: the selected provider's entry is passed
through OPENCODE_AUTH_CONTENT, and any auth.json opencode writes is removed.

The `[far]` lines in an attempt's output log are written by this side, not by
opencode.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from src.utils import PipelineCancelled

ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = ROOT / "agents"

OPENCODE_VERSION = "1.18.32"
# Docker's OpenCode sandbox template; 0.7.0 ships OpenCode 1.18.32.
DOCKER_IMAGE = (
    "docker/sandbox-templates:opencode-0.7.0"
    "@sha256:b3b69aa5148a20d1c0e3ec4d5db2975b8b9d6830d46d50a93fd766aec7d6668b"
)
CONTAINER_HOME = "/home/agent"
CONTAINER_WORKSPACE = "/workspace"

# Bound on each step of winding an attempt down: stop, export, remove.
FINISH_TIMEOUT = 120
STOP_GRACE = 10

# Pinned behaviour in both modes: no self-update, no external plugins, and no
# project or Claude Code configuration picked up from around the workspace.
OPENCODE_ENV = {
    "OPENCODE_DISABLE_AUTOUPDATE": "1",
    "OPENCODE_PURE": "1",
    "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
    "OPENCODE_DISABLE_CLAUDE_CODE": "1",
}
# Workspace snapshots would track the enclosing git repository, not the task.
BASE_CONFIG = {"snapshot": False}


# ── Logging and session export, shared by both runtimes ─────────────────────


def host_log(log: IO[bytes], message: str) -> None:
    log.write(f"\n[far] {time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n".encode())


def save_session(runtime: Any, state: Path, session_id: str | None, path: Path, log: IO[bytes]) -> None:
    """Export one attempt's native session next to its output log.

    A missing or unreadable export is recorded, never papered over: the store
    under `state` is kept either way and can be exported again by hand.
    """
    if not session_id:
        host_log(log, "session not exported: opencode reported no session id")
        return
    exported = export_path(state)
    try:
        exported.unlink(missing_ok=True)
        runtime.export(state, session_id)
        session = json.loads(exported.read_bytes())
        messages = session["messages"]
        tools = sum(1 for m in messages for p in m.get("parts", []) if p.get("type") == "tool")
        exported.replace(path)
        host_log(log, f"session {session_id} exported to {path.name}: {len(messages)} messages, {tools} tool calls")
    except Exception as exc:  # noqa: BLE001 - an export failure must not lose the answer
        reason = str(exc).strip().splitlines()[-1:] or [type(exc).__name__]
        host_log(log, f"session {session_id} not exported ({reason[0][:300]}); native store kept at {state}")


def export_path(state: Path) -> Path:
    # opencode exits before a pipe drains, cutting stdout at 64 KiB, so the
    # export goes to a file inside the store and is moved out once it parses.
    return state / "opencode" / "far-export.json"


def remove_auth(state: Path) -> None:
    (state / "opencode" / "auth.json").unlink(missing_ok=True)


# ── Credentials ─────────────────────────────────────────────────────────────


def _user_data_dir() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "opencode"


def _user_provider_config(provider: str) -> dict[str, Any] | None:
    config_dir = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "opencode"
    for name in ("opencode.json", "config.json"):
        path = config_dir / name
        if path.exists():
            try:
                found = (json.loads(path.read_text()).get("provider") or {}).get(provider)
            except (json.JSONDecodeError, AttributeError):
                continue
            if found:
                return found
    return None


def provider_env(model_id: str, container: bool) -> dict[str, str]:
    """Environment carrying only the selected provider's credentials and config.

    The stored login for that provider goes in OPENCODE_AUTH_CONTENT. A local
    run already inherits the user's environment and global config; a container
    additionally gets the provider's section of that config and its
    `<PROVIDER>_API_KEY` / `<PROVIDER>_BASE_URL` variables.
    """
    provider = model_id.split("/", 1)[0]
    env: dict[str, str] = {}
    auth_path = _user_data_dir() / "auth.json"
    if auth_path.exists():
        try:
            entry = json.loads(auth_path.read_text()).get(provider)
        except json.JSONDecodeError:
            entry = None
        if entry:
            env["OPENCODE_AUTH_CONTENT"] = json.dumps({provider: entry})
    config: dict[str, Any] = dict(BASE_CONFIG)
    if container:
        section = _user_provider_config(provider)
        names = {f"{provider.upper().replace('-', '_')}_{suffix}" for suffix in ("API_KEY", "BASE_URL")}
        if section:
            config["provider"] = {provider: section}
            text = json.dumps(section)
            names.update(part.split("}", 1)[0] for part in text.split("{env:")[1:])
        env.update({name: os.environ[name] for name in names if os.environ.get(name)})
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    return env


# ── Process helpers ─────────────────────────────────────────────────────────


@dataclass
class Call:
    """One running attempt."""

    proc: subprocess.Popen
    state: Path
    container: str = ""
    env: dict[str, str] = field(default_factory=dict)
    groups: set[int] = field(default_factory=set)
    last_scan: float = 0


def _run(command: list[str], timeout: float = FINISH_TIMEOUT, **kwargs) -> subprocess.CompletedProcess:
    # A new session keeps the terminal's Ctrl-C away from the bookkeeping
    # commands that are winding an attempt down.
    return subprocess.run(
        command, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout, start_new_session=True, **kwargs
    )


def _popen(command: list[str], **kwargs) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        **kwargs,
    )


def _process_groups(root: int) -> set[int]:
    """Process groups of `root` and all its descendants.

    Tool commands may start their own groups; collect them while the tree is
    still connected, before the parent dies and they are re-parented.
    """
    try:
        rows = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,pgid="], capture_output=True, text=True).stdout
    except OSError:
        return {root}
    children: dict[int, list[tuple[int, int]]] = {}
    for line in rows.splitlines():
        parts = line.split()
        if len(parts) == 3:
            pid, ppid, pgid = map(int, parts)
            children.setdefault(ppid, []).append((pid, pgid))
    groups, stack = {root}, [root]
    while stack:
        for pid, pgid in children.get(stack.pop(), []):
            groups.add(pgid)
            stack.append(pid)
    groups.discard(os.getpgid(0))
    return groups


def _signal_groups(groups: set[int], sig: int) -> None:
    for group in groups:
        try:
            os.killpg(group, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _wait(proc: subprocess.Popen, timeout: float) -> bool:
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ── Local ───────────────────────────────────────────────────────────────────


class LocalRuntime:
    """opencode on this machine, one process group per attempt."""

    mode = "local"

    def __init__(self, work_root: Path) -> None:
        # opencode writes package files into its config directory, so point it
        # at a directory of its own that links to the repository's agents.
        self.config_dir = work_root / ".opencode"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        link = self.config_dir / "agents"
        if not link.is_symlink():
            try:
                link.symlink_to(AGENTS_DIR, target_is_directory=True)
            except FileExistsError:
                pass

    def preflight(self) -> None:
        if shutil.which("opencode") is None:
            raise SystemExit(
                "opencode executable not found on PATH; install opencode or update PATH "
                "before running the solve, judge, or grade stages."
            )
        version = _run(["opencode", "--version"], timeout=60, text=True).stdout.strip()
        if version != OPENCODE_VERSION:
            raise SystemExit(f"opencode {OPENCODE_VERSION} is required, found {version or 'unknown'}.")

    def recover(self) -> None:
        """Local attempts leave nothing behind but their store, which is kept."""

    def path(self, workspace_file: Path) -> str:
        return str(workspace_file)

    @contextmanager
    def slot(self, stop_event: threading.Event | None):
        yield

    def _env(self, state: Path) -> dict[str, str]:
        return {
            **os.environ,
            **OPENCODE_ENV,
            "OPENCODE_CONFIG_DIR": str(self.config_dir),
            "XDG_DATA_HOME": str(state),
            "XDG_STATE_HOME": str(state / "state"),
        }

    def start(self, argv: list[str], workspace: Path, state: Path, inputs: list[Path], model_id: str, log) -> Call:
        env = {**self._env(state), **provider_env(model_id, container=False)}
        host_log(log, f"local opencode in {workspace}")
        return Call(_popen(argv, cwd=workspace, env=env), state, env=env)

    def stop(self, call: Call) -> None:
        groups = call.groups | _process_groups(call.proc.pid)
        _signal_groups(groups, signal.SIGTERM)
        _wait(call.proc, STOP_GRACE)
        # Tools in their own process groups may survive after the parent has
        # already exited; do not make their cleanup depend on the parent.
        _signal_groups(groups | _process_groups(call.proc.pid), signal.SIGKILL)
        if call.proc.poll() is None:
            _wait(call.proc, STOP_GRACE)

    def poll(self, call: Call) -> int | None:
        now = time.monotonic()
        if now - call.last_scan >= 1:
            call.groups.update(_process_groups(call.proc.pid))
            call.last_scan = now
        return call.proc.poll()

    def status(self, call: Call) -> tuple[int, str]:
        return call.proc.wait(), ""

    def export(self, state: Path, session_id: str) -> None:
        with open(export_path(state), "wb") as out:
            result = subprocess.run(
                ["opencode", "export", session_id], cwd=state, env=self._env(state), stdin=subprocess.DEVNULL,
                stdout=out, stderr=subprocess.PIPE, timeout=FINISH_TIMEOUT, start_new_session=True,
            )  # fmt: skip
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace") or f"exit {result.returncode}")

    def cleanup(self, call: Call) -> None:
        # Whatever the agent left running in the background goes with it.
        _signal_groups(call.groups | _process_groups(call.proc.pid), signal.SIGKILL)
        remove_auth(call.state)


# ── Docker ──────────────────────────────────────────────────────────────────


class DockerRuntime:
    """opencode in a fresh container per attempt.

    `jobs` bounds the containers alive at once across all stages. An attempt
    holds its slot from `docker create` to `docker rm`, including the helper
    container that exports its session, so winding down never waits for a slot.
    """

    mode = "docker"

    def __init__(self, jobs: int, cpus: str, memory: str, pids: int = 1024) -> None:
        self.budget = threading.BoundedSemaphore(max(jobs, 1))
        self.limits = ["--cpus", cpus, "--memory", memory, "--memory-swap", memory, "--pids-limit", str(pids)]
        self.owner = f"{socket.gethostname()}:{os.getpid()}"

    def preflight(self) -> None:
        try:
            server = _run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=30, text=True)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SystemExit(f"--mode docker needs a running Docker daemon: {exc}")
        if server.returncode != 0:
            raise SystemExit(f"--mode docker needs a running Docker daemon: {server.stderr.strip()}")
        if _run(["docker", "image", "inspect", DOCKER_IMAGE], timeout=30).returncode != 0:
            raise SystemExit(f"Docker image missing; pull it first:\n    docker pull {DOCKER_IMAGE}")
        version = _run(self._helper(["opencode", "--version"]), timeout=FINISH_TIMEOUT, text=True).stdout.strip()
        if version != OPENCODE_VERSION:
            raise SystemExit(f"{DOCKER_IMAGE} should carry opencode {OPENCODE_VERSION}, found {version or 'unknown'}.")

    def path(self, workspace_file: Path) -> str:
        return f"{CONTAINER_WORKSPACE}/{workspace_file.name}"

    @contextmanager
    def slot(self, stop_event: threading.Event | None):
        while not self.budget.acquire(timeout=0.5):
            if stop_event is not None and stop_event.is_set():
                raise PipelineCancelled("pipeline cancelled")
        try:
            yield
        finally:
            self.budget.release()

    def _helper(self, command: list[str], mounts: list[str] = ()) -> list[str]:
        """A short-lived container with no network, for work that calls no model."""
        return [
            "docker", "run", "--rm", "--pull", "never", "--network", "none",
            "--name", f"far-helper-{uuid.uuid4().hex[:12]}", "--label", f"far.helper={self.owner}",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            *mounts, *[arg for k, v in OPENCODE_ENV.items() for arg in ("-e", f"{k}={v}")],
            DOCKER_IMAGE, *command,
        ]  # fmt: skip

    def start(self, argv: list[str], workspace: Path, state: Path, inputs: list[Path], model_id: str, log) -> Call:
        data = state / "opencode"
        data.mkdir(parents=True, exist_ok=True)
        secrets = provider_env(model_id, container=True)
        mounts = ["-v", f"{workspace}:{CONTAINER_WORKSPACE}"]
        for path in inputs:
            mounts += ["-v", f"{workspace / path.name}:{self.path(path)}:ro"]
        mounts += [
            "-v", f"{AGENTS_DIR}:{CONTAINER_HOME}/.config/opencode/agents:ro",
            "-v", f"{data}:{CONTAINER_HOME}/.local/share/opencode",
        ]  # fmt: skip
        command = [
            "docker", "create", "--pull", "never",
            "--label", f"far.owner={self.owner}", "--label", f"far.log={log.name}", "--label", f"far.state={state}",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", *self.limits,
            "-w", CONTAINER_WORKSPACE, *mounts,
            *[arg for k, v in OPENCODE_ENV.items() for arg in ("-e", f"{k}={v}")],
            # Names only: docker reads the values from its own environment, so
            # they never appear on a command line.
            *[arg for name in secrets for arg in ("-e", name)],
            DOCKER_IMAGE, *argv,
        ]  # fmt: skip
        created = _run(command, timeout=FINISH_TIMEOUT, env={**os.environ, **secrets}, text=True)
        if created.returncode != 0:
            raise RuntimeError(f"docker create failed: {created.stderr.strip()}")
        container = created.stdout.strip()
        host_log(log, f"docker container {container[:12]} image {DOCKER_IMAGE}")
        try:
            return Call(_popen(["docker", "start", "-a", container]), state, container=container)
        except OSError:
            _run(["docker", "rm", "-f", container])
            raise

    def poll(self, call: Call) -> int | None:
        return call.proc.poll()

    def stop(self, call: Call) -> None:
        if call.container:
            try:
                stopped = _run(["docker", "stop", "-t", str(STOP_GRACE), call.container])
                if stopped.returncode != 0:
                    _run(["docker", "kill", call.container], timeout=30)
            except subprocess.TimeoutExpired:
                _run(["docker", "kill", call.container], timeout=30)
        if not _wait(call.proc, STOP_GRACE):
            call.proc.kill()
            call.proc.wait()

    def status(self, call: Call) -> tuple[int, str]:
        call.proc.wait()
        return self._exit_state(call.container, call.proc.returncode)

    def _exit_state(self, container: str, fallback: int | None) -> tuple[int, str]:
        # OOM is read from Docker's record, not guessed from exit code 137.
        result = _run(["docker", "inspect", "-f", "{{.State.ExitCode}} {{.State.OOMKilled}}", container], text=True)
        code, oom = (result.stdout.split() + ["", ""])[:2]
        if result.returncode != 0 or not code.lstrip("-").isdigit():
            return fallback or 1, "container state unavailable"
        return int(code), "container OOM-killed" if oom == "true" else ""

    def export(self, state: Path, session_id: str) -> None:
        data = f"{CONTAINER_HOME}/.local/share/opencode"
        mount = ["-v", f"{state / 'opencode'}:{data}"]
        target = f"{data}/{export_path(state).name}"
        command = self._helper(["sh", "-c", 'opencode export "$1" > "$2"', "sh", session_id, target], mount)
        try:
            result = _run(command)
        except subprocess.TimeoutExpired:
            _run(["docker", "rm", "-f", command[command.index("--name") + 1]], timeout=30)
            raise
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace") or f"exit {result.returncode}")

    def cleanup(self, call: Call) -> None:
        try:
            if call.container:
                result = _run(["docker", "rm", "-f", call.container], timeout=FINISH_TIMEOUT)
                if result.returncode:
                    raise RuntimeError(result.stderr.decode(errors="replace").strip())
        finally:
            remove_auth(call.state)

    def recover(self) -> None:
        """Wind down containers a dead FAR process on this host left behind.

        Only containers labelled by FAR whose owning process is gone are
        touched; a live owner, another host, or an unlabelled container is
        left alone. The output log gets the container's own log and the
        session is exported before the container is removed. The store on
        the host is never deleted.
        """
        fmt = '{{.ID}}\t{{.Label "far.owner"}}\t{{.Label "far.log"}}\t{{.Label "far.state"}}'
        listed = _run(["docker", "ps", "-a", "--filter", "label=far.owner", "--format", fmt], timeout=60, text=True)
        host = socket.gethostname()
        for row in listed.stdout.splitlines():
            container, owner, log_name, state = (row.split("\t") + ["", "", ""])[:4]
            owner_host, _, pid = owner.rpartition(":")
            if owner_host != host or not pid.isdigit() or pid_alive(int(pid)) or not log_name:
                continue
            log_path, state_path = Path(log_name), Path(state)
            print(f"[docker] recovering container {container} left by process {pid}", flush=True)
            _run(["docker", "stop", "-t", str(STOP_GRACE), container])
            with open(log_path, "ab", buffering=0) as log:
                host_log(log, f"recovering container {container} left by dead process {pid}; its log follows")
                logs = _run(["docker", "logs", container])
                log.write(logs.stdout + logs.stderr)
                code, oom = self._exit_state(container, None)
                host_log(log, f"recovered container exit code {code} {oom}".rstrip())
                session_path = log_path.with_name(log_path.name.replace("output_", "session_")).with_suffix(".json")
                save_session(self, state_path, session_id_in(log_path), session_path, log)
                _run(["docker", "rm", "-f", container])
                remove_auth(state_path)


def session_id_in(log_path: Path) -> str | None:
    marker = b'"sessionID":"'
    with open(log_path, "rb") as handle:
        for line in handle:
            start = line.find(marker)
            if start >= 0:
                start += len(marker)
                return line[start : line.index(b'"', start)].decode()
    return None


def make_runtime(args) -> "LocalRuntime | DockerRuntime":
    work_root = Path(args.work_root).expanduser().resolve()
    if args.mode == "docker":
        return DockerRuntime(jobs=8, cpus="2", memory="4g")
    return LocalRuntime(work_root)
