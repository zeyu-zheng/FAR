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

Every attempt is tagged with its owner, `<host>:<pid>` of this FAR process: a
container through its labels, a local attempt through the FAR_CALL variable
its processes inherit. Stopping, cleanup, and recovery act only on processes
proven to be an attempt's (see `_members`), so they reach tools that left the
process tree and never touch anything else.

The `[far]` lines in an attempt's output log are written by this side, not by
opencode.
"""

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import psutil

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"

# Docker's OpenCode sandbox template; 0.7.0 ships OpenCode 1.18.32.
DOCKER_IMAGE = "docker/sandbox-templates:opencode-0.7.0"
CONTAINER_WORKSPACE = "/workspace"
# Containers run as the host user, who has no home in the image: the agent
# gets one under its store, and the session data is mounted on its own. Both
# live under /tmp, where that user can create them when nothing is mounted.
CONTAINER_HOME = "/tmp/far/home"
CONTAINER_DATA = "/tmp/far/data"
CONTAINER_LIMITS = ["--cpus", "2", "--memory", "4g", "--memory-swap", "4g", "--pids-limit", "1024"]

# Pinned behaviour in both modes: no self-update, no external plugins, and no
# project or Claude Code configuration picked up from around the workspace.
OPENCODE_ENV = {
    "OPENCODE_DISABLE_AUTOUPDATE": "1",
    "OPENCODE_PURE": "1",
    "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
    "OPENCODE_DISABLE_CLAUDE_CODE": "1",
}
TAG_VAR = "FAR_CALL"


def process_owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def owner_gone(owner: str) -> bool:
    """True for an owner on this host whose process has exited."""
    host, _, pid = owner.rpartition(":")
    return host == socket.gethostname() and pid.isdigit() and not psutil.pid_exists(int(pid))


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


def opencode_env(model_id: str, container: bool) -> dict[str, str]:
    """The agents, the pinned settings, and only the selected provider's credentials.

    Everything opencode is configured with travels in OPENCODE_CONFIG_CONTENT,
    the same way in both modes: the agent definitions, the pinned settings, and for a
    container the provider's section of the user's global config. The stored
    login for that provider goes in OPENCODE_AUTH_CONTENT. A local run already
    inherits the user's environment; a container additionally gets the
    provider's `<PROVIDER>_API_KEY` / `<PROVIDER>_BASE_URL` variables.
    """
    provider = model_id.split("/", 1)[0]
    env: dict[str, str] = {}
    auth_path = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "opencode" / "auth.json"
    if auth_path.exists():
        try:
            entry = json.loads(auth_path.read_text()).get(provider)
        except json.JSONDecodeError:
            entry = None
        if entry:
            env["OPENCODE_AUTH_CONTENT"] = json.dumps({provider: entry})
    # agents/agents.json is opencode's own `agent` config minus the prompts,
    # which are agents/<name>.md.
    agents = json.loads((AGENTS_DIR / "agents.json").read_text(encoding="utf-8"))
    for name, agent in agents.items():
        agent["prompt"] = (AGENTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
    # Workspace snapshots would track the enclosing git repository, not the task.
    config: dict[str, Any] = {"snapshot": False, "agent": agents}
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
    tag: str = ""
    container: str = ""
    # Local only: every descendant seen while the attempt ran, by pid and
    # creation time, so a pid taken over by another process is not mistaken.
    seen: dict[int, float] = field(default_factory=dict)


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    # A new session keeps the terminal's Ctrl-C away from the bookkeeping
    # commands that are winding an attempt down.
    return subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, start_new_session=True, **kwargs)


def _popen(command: list[str], **kwargs) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        **kwargs,
    )


# ── Which local processes belong to an attempt ──────────────────────────────


@dataclass
class _Proc:
    ppid: int
    created: float
    pgid: int
    tag: str | None


_table_lock = threading.Lock()
_table_cache: tuple[float, dict[int, _Proc]] = (0.0, {})


def _table(max_age: float = 0.0) -> dict[int, _Proc]:
    """This user's live processes. Attempts side by side share one listing up to `max_age` old.

    The tag is None where it cannot be read: macOS hides the environment of
    its own binaries (sh, sleep, ...).
    """
    global _table_cache
    with _table_lock:
        taken, table = _table_cache
        if time.monotonic() - taken > max_age:
            table = {}
            mine = os.getpgid(0)
            for proc in psutil.process_iter(["ppid", "create_time", "status", "uids"]):
                info = proc.info
                if info["status"] == psutil.STATUS_ZOMBIE or not info["uids"] or info["uids"].real != os.getuid():
                    continue
                try:
                    pgid = os.getpgid(proc.pid)
                except OSError:
                    continue  # exited meanwhile
                try:
                    tag = proc.environ().get(TAG_VAR)
                except psutil.Error:
                    tag = None
                if pgid == mine:
                    continue  # never this FAR process or anything in its group
                table[proc.pid] = _Proc(info["ppid"], info["create_time"], pgid, tag)
            _table_cache = (time.monotonic(), table)
        return table


def _members(match: Callable[[str], bool], calls: list[Call] = ()) -> set[int]:
    """Live processes proven to belong to the matching attempts.

    Proof is one of: a matching FAR_CALL tag, a descendant seen with the same
    creation time, or a call's own opencode process not yet reaped. Then the
    descendants of a proven process, and the processes in its group, belong
    too: that reaches the tools whose tag macOS will not show.
    """
    table = _table()
    own = {pid for pid, proc in table.items() if proc.tag is not None and match(proc.tag)}
    for call in calls:
        own |= {pid for pid, created in call.seen.items() if pid in table and table[pid].created == created}
        if call.proc.returncode is None and call.proc.pid in table:
            own.add(call.proc.pid)
    while True:
        groups = {table[pid].pgid for pid in own}
        more = {pid for pid, proc in table.items() if pid not in own and (proc.ppid in own or proc.pgid in groups)}
        if not more:
            return own
        own |= more


def _signal(pids: set[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


# ── Local ───────────────────────────────────────────────────────────────────


class LocalRuntime:
    """opencode on this machine, its processes tagged per attempt."""

    mode = "local"

    def __init__(self) -> None:
        self.owner = process_owner()
        # Attempts between start and cleanup, for kill_all.
        self.active: dict[str, Call] = {}

    def recover(self) -> None:
        """Stop processes a dead FAR process on this host left running.

        Their stores are kept; the sessions in them can be exported by hand.
        """

        def orphaned(tag: str) -> bool:
            return owner_gone(tag.rpartition(":")[0])

        left = _members(orphaned)
        if left:
            print(f"[local] stopping {len(left)} agent processes left by a dead FAR process", flush=True)
            _signal(left, signal.SIGKILL)

    def path(self, workspace_file: Path) -> str:
        return str(workspace_file)

    def _env(self, state: Path) -> dict[str, str]:
        return {
            **os.environ,
            **OPENCODE_ENV,
            "XDG_DATA_HOME": str(state),
            "XDG_STATE_HOME": str(state / "state"),
        }

    def start(self, argv: list[str], workspace: Path, state: Path, inputs: list[Path], model_id: str, log) -> Call:
        tag = f"{self.owner}:{uuid.uuid4().hex[:12]}"
        env = {
            **self._env(state),
            **opencode_env(model_id, container=False),
            TAG_VAR: tag,
            # opencode takes its directory, and so its tools' cwd, from PWD.
            "PWD": str(workspace),
        }
        host_log(log, f"local opencode in {workspace}")
        call = Call(_popen(argv, cwd=workspace, env=env), state, tag=tag)
        self.active[tag] = call
        return call

    def stop(self, call: Call) -> None:
        # Its tools go in cleanup().
        call.proc.kill()

    def poll(self, call: Call) -> int | None:
        code = call.proc.poll()
        if code is None:
            # Record the attempt's descendants while the tree is connected.
            table = _table(max_age=1.0)
            stack = [call.proc.pid]
            while stack:
                parent = stack.pop()
                for pid, proc in table.items():
                    if proc.ppid == parent:
                        call.seen[pid] = proc.created
                        stack.append(pid)
        return code

    def status(self, call: Call) -> tuple[int, str]:
        return call.proc.wait(), ""

    def export(self, state: Path, session_id: str) -> None:
        with open(export_path(state), "wb") as out:
            result = subprocess.run(
                ["opencode", "export", session_id], cwd=state, env=self._env(state), stdin=subprocess.DEVNULL,
                stdout=out, stderr=subprocess.PIPE, start_new_session=True,
            )  # fmt: skip
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace") or f"exit {result.returncode}")

    def cleanup(self, call: Call) -> None:
        # Whatever the agent left running in the background goes with it.
        _signal(_members(lambda tag: tag == call.tag, [call]), signal.SIGKILL)
        self.active.pop(call.tag, None)
        remove_auth(call.state)

    def kill_all(self) -> None:
        """Kill every agent process this FAR process started, at once."""
        mine = f"{self.owner}:"
        _signal(_members(lambda tag: tag.startswith(mine), list(self.active.values())), signal.SIGKILL)


# ── Docker ──────────────────────────────────────────────────────────────────


class DockerRuntime:
    """opencode in a fresh container per attempt, run as the host user."""

    mode = "docker"

    def __init__(self) -> None:
        self.owner = process_owner()
        # Files the agent writes stay the host user's, so the next attempt can
        # rebuild its workspace on any host.
        self.user = ["--user", f"{os.getuid()}:{os.getgid()}"]

    def path(self, workspace_file: Path) -> str:
        return f"{CONTAINER_WORKSPACE}/{workspace_file.name}"

    def _env_args(self, home: str) -> list[str]:
        env = {**OPENCODE_ENV, "HOME": home, "XDG_DATA_HOME": CONTAINER_DATA}
        return [arg for k, v in env.items() for arg in ("-e", f"{k}={v}")]

    def start(self, argv: list[str], workspace: Path, state: Path, inputs: list[Path], model_id: str, log) -> Call:
        data = state / "opencode"
        home = state / "home"
        # Made here, as the host user: Docker would make them as root.
        for path in (data, home):
            path.mkdir(parents=True, exist_ok=True)
        secrets = opencode_env(model_id, container=True)
        mounts = ["-v", f"{workspace}:{CONTAINER_WORKSPACE}"]
        for path in inputs:
            mounts += ["-v", f"{workspace / path.name}:{self.path(path)}:ro"]
        mounts += [
            "-v", f"{home}:{CONTAINER_HOME}",
            "-v", f"{data}:{CONTAINER_DATA}/opencode",
        ]  # fmt: skip
        command = [
            "docker", "create",
            "--label", f"far.owner={self.owner}", "--label", f"far.log={log.name}", "--label", f"far.state={state}",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", *self.user, *CONTAINER_LIMITS,
            "-w", CONTAINER_WORKSPACE, *mounts, *self._env_args(CONTAINER_HOME),
            # Names only: docker reads the values from its own environment, so
            # they stay off the command line. `docker inspect` still shows them
            # while the container exists; it is removed when the attempt ends.
            *[arg for name in secrets for arg in ("-e", name)],
            DOCKER_IMAGE, *argv,
        ]  # fmt: skip
        created = _run(command, env={**os.environ, **secrets}, text=True)
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
        _run(["docker", "kill", call.container])

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
        # A short-lived container with no network: exporting calls no model.
        target = f"{CONTAINER_DATA}/opencode/{export_path(state).name}"
        result = _run([
            "docker", "run", "--rm", "--network", "none",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", *self.user,
            "-v", f"{state / 'opencode'}:{CONTAINER_DATA}/opencode", *self._env_args("/tmp"),
            DOCKER_IMAGE, "sh", "-c", 'opencode export "$1" > "$2"', "sh", session_id, target,
        ])  # fmt: skip
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace") or f"exit {result.returncode}")

    def cleanup(self, call: Call) -> None:
        try:
            result = _run(["docker", "rm", "-f", call.container])
            if result.returncode:
                raise RuntimeError(result.stderr.decode(errors="replace").strip())
        finally:
            remove_auth(call.state)
            shutil.rmtree(call.state / "home", ignore_errors=True)

    def kill_all(self) -> None:
        """Kill every container this FAR process started; recover() finishes them."""
        listed = _run(["docker", "ps", "-q", "--filter", f"label=far.owner={self.owner}"], text=True)
        if listed.stdout.split():
            _run(["docker", "kill", *listed.stdout.split()])

    def recover(self) -> None:
        """Wind down containers a dead FAR process on this host left behind.

        Only containers labelled by FAR whose owning process is gone are
        touched; a live owner, another host, or an unlabelled container is
        left alone. The output log gets the container's own log and the
        session is exported before the container is removed. The store on
        the host is never deleted. One container failing to recover does not
        hold up the others or the run.
        """
        fmt = '{{.ID}}\t{{.Label "far.owner"}}\t{{.Label "far.log"}}\t{{.Label "far.state"}}'
        listed = _run(["docker", "ps", "-a", "--filter", "label=far.owner", "--format", fmt], text=True)
        for row in listed.stdout.splitlines():
            container, owner, log_name, state = (row.split("\t") + ["", "", ""])[:4]
            if not owner_gone(owner) or not log_name:
                continue
            print(f"[docker] recovering container {container} left by {owner}", flush=True)
            log_path, state_path = Path(log_name), Path(state)
            try:
                _run(["docker", "kill", container])
                with open(log_path, "ab", buffering=0) as log:
                    host_log(log, f"recovering container {container} left by dead process {owner}; its log follows")
                    logs = _run(["docker", "logs", container])
                    log.write(logs.stdout + logs.stderr)
                    code, oom = self._exit_state(container, None)
                    host_log(log, f"recovered container exit code {code} {oom}".rstrip())
                    found = re.search(rb'"sessionID":"([^"]+)"', log_path.read_bytes())
                    session_path = log_path.with_name(log_path.name.replace("output_", "session_")).with_suffix(".json")
                    save_session(self, state_path, found and found[1].decode(), session_path, log)
                _run(["docker", "rm", "-f", container])
                remove_auth(state_path)
                shutil.rmtree(state_path / "home", ignore_errors=True)
            except Exception as exc:  # noqa: BLE001 - recover the rest, keep this one for a later run
                print(f"[docker] could not recover {container}: {exc}", flush=True)
