"""Where an opencode agent runs: on this machine, or in a Docker container.

Both runtimes take the same opencode command line and the same workspace,
<stage_dir>/workspace/, and differ only in how the process is started,
stopped, and read back. A local opencode keeps its sessions where it always
does; a container mounts nothing from the host, so its workspace is copied out
of it at the end (see DockerRuntime). Either way, each session is exported to
session_NNN.json. Model credentials reach opencode only as environment
variables.

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
import tarfile
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
# An idle container stops by itself after this, whatever happened to FAR.
CONTAINER_LIFETIME = "12h"

# How long a stopped agent gets to exit on SIGTERM before SIGKILL.
STOP_GRACE = 10

# Pinned behaviour in both modes: no self-update, no external plugins, and no
# project or Claude Code configuration picked up from around the workspace.
OPENCODE_ENV = {
    "OPENCODE_DISABLE_AUTOUPDATE": "1",
    "OPENCODE_PURE": "1",
    "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
    "OPENCODE_DISABLE_CLAUDE_CODE": "1",
}
TAG_VAR = "FAR_CALL"


# This FAR process, as `<host>:<pid>`.
OWNER = f"{socket.gethostname()}:{os.getpid()}"


def owner_gone(owner: str) -> bool:
    """True for an owner on this host whose process has exited."""
    host, _, pid = owner.rpartition(":")
    return host == socket.gethostname() and pid.isdigit() and not psutil.pid_exists(int(pid))


# ── Logging and session export, shared by both runtimes ─────────────────────


def host_log(log: IO[bytes], message: str) -> None:
    log.write(f"\n[far] {time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n".encode())


def save_session(runtime: Any, call: "Call", session_id: str | None, path: Path, log: IO[bytes]) -> None:
    """Export one attempt's native session to `path`, next to its output log.

    opencode exits before a pipe drains, cutting its stdout at 64 KiB, so the
    export is written to a file, never piped. One that does not parse is
    removed and recorded as an error; the answer stands.
    """
    if not session_id:
        host_log(log, "session not exported: opencode reported no session id")
        return
    try:
        runtime.export(call, session_id, path)
        messages = json.loads(path.read_bytes())["messages"]
        tools = sum(1 for m in messages for p in m.get("parts", []) if p.get("type") == "tool")
        host_log(log, f"session {session_id} exported to {path.name}: {len(messages)} messages, {tools} tool calls")
    except Exception as exc:  # noqa: BLE001 - an export failure must not lose the answer
        path.unlink(missing_ok=True)
        reason = str(exc).strip().splitlines()[-1:] or [type(exc).__name__]
        host_log(log, f"error: session {session_id} not exported ({reason[0][:300]})")


# ── opencode configuration ──────────────────────────────────────────────────


def opencode_env(model_id: str) -> dict[str, str]:
    """Everything opencode is configured with: OPENCODE_ENV, and in
    OPENCODE_CONFIG_CONTENT the agents, the pinned settings, and the model
    provider's section of the user's global config, which a container cannot
    read from the host.
    """
    # agents/agents.json is opencode's own `agent` config minus the prompts,
    # which are agents/<name>.md.
    agents = json.loads((AGENTS_DIR / "agents.json").read_text(encoding="utf-8"))
    for name, agent in agents.items():
        agent["prompt"] = (AGENTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
    # Workspace snapshots would track the enclosing git repository, not the task.
    config: dict[str, Any] = {"snapshot": False, "agent": agents}
    provider = model_id.split("/", 1)[0]
    config_dir = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "opencode"
    for name in ("opencode.json", "config.json"):
        try:
            section = (json.loads((config_dir / name).read_text()).get("provider") or {}).get(provider)
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        if section:
            config["provider"] = {provider: section}
            break
    return {**OPENCODE_ENV, "OPENCODE_CONFIG_CONTENT": json.dumps(config)}


# ── Process helpers ─────────────────────────────────────────────────────────


@dataclass
class Call:
    """One running attempt."""

    proc: subprocess.Popen | None
    tag: str = ""
    container: str = ""
    workspace: Path | None = None
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


def _stop(find: Callable[[], set[int]]) -> None:
    """SIGTERM what `find` returns, then SIGKILL whatever is left after STOP_GRACE."""
    _signal(find(), signal.SIGTERM)
    deadline = time.monotonic() + STOP_GRACE
    while time.monotonic() < deadline and find():
        time.sleep(0.5)
    _signal(find(), signal.SIGKILL)


# ── Local ───────────────────────────────────────────────────────────────────


class LocalRuntime:
    """opencode on this machine, its processes tagged per attempt."""

    mode = "local"

    def __init__(self) -> None:
        # Attempts between start and cleanup, for kill_all.
        self.active: dict[str, Call] = {}

    def recover(self) -> None:
        """Stop processes a dead FAR process on this host left running.

        Their sessions stay in opencode's own database.
        """

        def orphaned(tag: str) -> bool:
            return owner_gone(tag.rpartition(":")[0])

        left = _members(orphaned)
        if left:
            print(f"[local] stopping {len(left)} agent processes left by a dead FAR process", flush=True)
            _stop(lambda: _members(orphaned))

    def start(self, argv: list[str], workspace: Path, model_id: str, log) -> Call:
        tag = f"{OWNER}:{uuid.uuid4().hex[:12]}"
        env = {
            **os.environ,
            **opencode_env(model_id),
            TAG_VAR: tag,
            # opencode takes its directory, and so its tools' cwd, from PWD.
            "PWD": str(workspace),
        }
        host_log(log, f"local opencode in {workspace}")
        call = Call(_popen(argv, cwd=workspace, env=env), tag=tag)
        self.active[tag] = call
        return call

    def stop(self, call: Call) -> None:
        # Its tools go in cleanup().
        call.proc.terminate()
        try:
            call.proc.wait(timeout=STOP_GRACE)
        except subprocess.TimeoutExpired:
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

    def export(self, call: Call, session_id: str, path: Path) -> None:
        with open(path, "wb") as out:
            result = subprocess.run(
                ["opencode", "export", session_id], env={**os.environ, **OPENCODE_ENV}, stdin=subprocess.DEVNULL,
                stdout=out, stderr=subprocess.PIPE, start_new_session=True,
            )  # fmt: skip
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace") or f"exit {result.returncode}")

    def cleanup(self, call: Call) -> None:
        # Whatever the agent left running in the background goes with it.
        _stop(lambda: _members(lambda tag: tag == call.tag, [call]))
        self.active.pop(call.tag, None)

    def kill_all(self) -> None:
        """Kill every agent process this FAR process started, at once."""
        mine = f"{OWNER}:"
        _signal(_members(lambda tag: tag.startswith(mine), list(self.active.values())), signal.SIGKILL)


# ── Docker ──────────────────────────────────────────────────────────────────


def _docker(*args: str, **kwargs) -> str:
    result = _run(["docker", *args], text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed: {result.stderr.strip()}")
    return result.stdout


def _exec_stream(container: str, script: str, *args: str, stdin: int = subprocess.DEVNULL) -> subprocess.Popen:
    """`sh -c script` in the container's working directory, its stdout piped back."""
    return subprocess.Popen(
        ["docker", "exec", "-i", container, "sh", "-c", script, "sh", *args],
        stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )  # fmt: skip


def _finish(proc: subprocess.Popen, what: str) -> None:
    if proc.wait() != 0:
        raise RuntimeError(f"{what} failed: {proc.stderr.read().decode(errors='replace').strip()}")


def _copy_in(container: str, source: Path) -> None:
    """The contents of `source` into the container's working directory."""
    proc = _exec_stream(container, "tar -x", stdin=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdin, mode="w|") as archive:
        for path in source.iterdir():
            archive.add(path, arcname=path.name)
    proc.stdin.close()
    _finish(proc, "copy into container")


def _copy_out(container: str, script: str, target: Path) -> None:
    """Unpack the tar stream `script` writes into `target`, as the host user.

    Whatever the agent left that would reach outside `target`, through a link
    or an absolute name, is skipped; everything else still comes out.
    """

    def safe(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
        try:
            return tarfile.data_filter(member, path)
        except tarfile.FilterError:
            return None

    target.mkdir(parents=True, exist_ok=True)
    proc = _exec_stream(container, script)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
        archive.extractall(target, filter=safe)
    proc.stdout.read()  # the archive's trailing padding
    _finish(proc, "copy out of container")


class DockerRuntime:
    """opencode in a fresh container per attempt, nothing mounted from the host.

    The container idles as `sleep` under the image's own user, home, and
    working directory. Everything else goes through `docker exec`, which runs
    in that working directory: the inputs go in as a tar stream, opencode
    runs, is stopped, and exports its session, and the workspace comes back
    out as a tar stream, as the host user's files, before the container is
    removed.
    """

    mode = "docker"

    def __init__(self, env_names: list[str]) -> None:
        self.env_names = env_names

    def start(self, argv: list[str], workspace: Path, model_id: str, log) -> Call:
        env = {name: os.environ[name] for name in self.env_names if os.environ.get(name)}
        env.update(opencode_env(model_id))
        container = _docker(
            "run", "--detach", "--entrypoint", "sleep",
            "--label", f"far.owner={OWNER}", "--label", f"far.log={log.name}",
            "--label", f"far.workspace={workspace}",
            # Names only: docker reads the values from its own environment, so
            # they stay off the command line. `docker inspect` still shows them
            # while the container exists; it is removed when the attempt ends.
            *[arg for name in env for arg in ("-e", name)],
            DOCKER_IMAGE, CONTAINER_LIFETIME,
            env={**os.environ, **env},
        ).strip()  # fmt: skip
        host_log(log, f"docker container {container[:12]} image {DOCKER_IMAGE}")
        try:
            _copy_in(container, workspace)
            proc = _popen(["docker", "exec", container, *argv])
        except Exception:
            _run(["docker", "rm", "-f", container])
            raise
        return Call(proc, container=container, workspace=workspace)

    def stop(self, call: Call) -> None:
        """SIGTERM every process but PID 1, then SIGKILL what is left after STOP_GRACE.

        PID 1 is the idle `sleep`, which no signal from inside the container
        can stop, so the container stays up for the export.
        """
        container = call.container
        _run(["docker", "exec", container, "sh", "-c", "kill -s TERM -- -1"])
        deadline = time.monotonic() + STOP_GRACE
        while time.monotonic() < deadline:
            rows = _run(["docker", "top", container, "-o", "stat"], text=True).stdout.split()[1:]
            if sum(1 for stat in rows if not stat.startswith("Z")) <= 1:
                return
            time.sleep(0.5)
        _run(["docker", "exec", container, "sh", "-c", "kill -s KILL -- -1"])

    def poll(self, call: Call) -> int | None:
        return call.proc.poll()

    def export(self, call: Call, session_id: str, path: Path) -> None:
        # A killed container idles again on start; its files are all still there.
        _docker("start", call.container)
        # Into a file inside, then cat, which does drain the pipe.
        script = 'f=$(mktemp) && opencode export "$1" > "$f" && cat "$f"'
        proc = _exec_stream(call.container, script, session_id)
        path.write_bytes(proc.stdout.read())
        _finish(proc, "session export")

    def cleanup(self, call: Call) -> None:
        try:
            # Whatever the agent left running in the background goes first.
            self.stop(call)
            shutil.rmtree(call.workspace)
            _copy_out(call.container, "tar -c .", call.workspace)
        finally:
            _docker("rm", "-f", call.container)

    def kill_all(self) -> None:
        """Kill every container this FAR process started; recover() finishes them."""
        containers = _run(["docker", "ps", "-q", "--filter", f"label=far.owner={OWNER}"], text=True).stdout.split()
        if containers:
            _run(["docker", "kill", *containers])

    def recover(self) -> None:
        """Finish containers a dead FAR process on this host left behind.

        Only containers labelled by FAR whose owning process is gone are
        touched. Each one is started again if it was killed, its agent is
        stopped, its session exported, and its files copied out, as at the end
        of a normal attempt. One failing does not hold up the others or the run.
        """
        fmt = "\t".join(f'{{{{.Label "far.{name}"}}}}' for name in ("owner", "log", "workspace"))
        rows = _docker("ps", "-a", "--filter", "label=far.owner", "--format", "{{.ID}}\t" + fmt)
        for row in rows.splitlines():
            container, owner, log_name, workspace = (row.split("\t") + [""] * 3)[:4]
            if not owner_gone(owner) or not log_name:
                continue
            print(f"[docker] recovering container {container} left by {owner}", flush=True)
            log_path = Path(log_name)
            call = Call(None, container=container, workspace=Path(workspace))
            try:
                _docker("start", container)
                self.stop(call)  # its agent, if it was still running
                with open(log_path, "ab", buffering=0) as log:
                    host_log(log, f"recovering container {container} left by dead process {owner}")
                    found = re.search(rb'"sessionID":"([^"]+)"', log_path.read_bytes())
                    session_path = log_path.with_name(log_path.name.replace("output_", "session_")).with_suffix(".json")
                    save_session(self, call, found and found[1].decode(), session_path, log)
                self.cleanup(call)
            except Exception as exc:  # noqa: BLE001 - recover the rest, keep this one for a later run
                print(f"[docker] could not recover {container}: {exc}", flush=True)
