"""FAR: Find, Attempt, and Recommend.

    Find       label -> extract -> check      corpus to an attemptable pool P
    Attempt    solve                          one attempt per conjecture
    Recommend  judge -> grade                 correctness, then significance

Stages are chained by bounded queues, so one conjecture can be in Solve while
another paper is still being labelled. Every stage also appends its output to
disk, so any stage can be run on its own against an earlier stage's file:

    python src/pipeline.py --stage all   --input data/raw/corpus
    python src/pipeline.py --stage find  --input data/raw/corpus
    python src/pipeline.py --stage check --input data/extracted.jsonl
    python src/pipeline.py --stage grade --input data/judged.jsonl

Re-running Judge or Grade on their own rebuilds the agent workspace from the
persisted record, so changing a judging rule does not re-run the prover.

Model and concurrency settings are flags; see scripts/.
"""

import argparse
import concurrent.futures
import itertools
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import check as check_stage
from src import extract as extract_stage
from src import grade as grade_stage
from src import judge as judge_stage
from src import label as label_stage
from src import solve as solve_stage
from src.agent import format_elapsed, install_opencode_agents, require_opencode, result_key
from src.reader import Corpus, iter_rows
from src.utils import (
    HTTP_RETRIES,
    MODEL_OUTPUT_LOG,
    CHAT_API,
    LLMClient,
    PipelineCancelled,
    RESPONSES_API,
    STAGE_ENV,
    append_jsonl,
    iter_jsonl,
    load_env_file,
    parse_bool,
    require_env,
    tail,
    wait_retry,
)

_SENTINEL = object()

STAGES = ("label", "extract", "check", "solve", "judge", "grade")
# Find stages call the model API directly; Attempt and Recommend go through the
# opencode agent, which carries its own credentials.
API_STAGES = ("label", "extract", "check")
AGENT_STAGES = ("solve", "judge", "grade")
# Each stage reads what the one before it wrote.
PREVIOUS = dict(zip(STAGES[1:], STAGES))
PHASES = {
    "all": STAGES,
    "find": ("label", "extract", "check"),
    "attempt": ("solve",),
    "recommend": ("judge", "grade"),
}
OUTPUT_NAMES = {
    "label": "labeled.jsonl",
    "extract": "extracted.jsonl",
    "check": "checked.jsonl",
    "solve": "solved.jsonl",
    "judge": "judged.jsonl",
    "grade": "graded.jsonl",
}
QUEUE_MAXSIZE = 100
# Seconds between attempts at one item. Transport failures are already absorbed
# inside the client, so what reaches here is a response that would not parse --
# waiting longer does not help, it just gives the model a fresh sample.
ITEM_RETRY_DELAY = 5.0


# ── Concurrency primitives ──────────────────────────────────────────────────


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.counts: dict[str, int] = {}

    def bump(self, stage: str, name: str, amount: int = 1) -> None:
        with self.lock:
            key = f"{stage}.{name}"
            self.counts[key] = self.counts.get(key, 0) + amount

    def line(self) -> str:
        with self.lock:
            groups = []
            for stage in STAGES:
                prefix = stage + "."
                items = [(k[len(prefix) :], v) for k, v in sorted(self.counts.items()) if k.startswith(prefix)]
                if items:
                    groups.append(f"{stage}: " + " ".join(f"{n}={v}" for n, v in items))
            return " | ".join(groups)

    def report(self) -> None:
        line = self.line()
        if line:
            print(line, flush=True)


def iter_stage_input(
    backfill: Iterable[Any], in_queue: "queue.Queue | None", stop_event: threading.Event
) -> Iterator[Any]:
    """Yield backfill first, then live queue items until upstream finishes."""
    yield from backfill
    if in_queue is None:
        return
    while not stop_event.is_set():
        try:
            item = in_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is _SENTINEL:
            return
        yield item


def put_item(out_queue: "queue.Queue | None", item: Any, stop_event: threading.Event) -> bool:
    """Put with cancellation so a failed downstream cannot deadlock producers."""
    if out_queue is None:
        return False
    while not stop_event.is_set():
        try:
            out_queue.put(item, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False


def run_pool(
    tasks: Iterable[Any],
    submit: Callable[[ThreadPoolExecutor, Any], concurrent.futures.Future],
    handle_done: Callable[[concurrent.futures.Future], None],
    jobs: int,
    ramp_delay: float,
    stop_event: threading.Event,
) -> None:
    """Run a bounded pool without blocking result collection on upstream input.

    A dedicated producer may block while waiting for an upstream stage. The main
    thread remains free to reap completed futures immediately, so streaming
    stages do not wait for a full batch of ``jobs`` tasks before forwarding
    results downstream.
    """
    task_queue: queue.Queue = queue.Queue(maxsize=max(jobs * 2, 1))
    producer_done = threading.Event()
    producer_errors: list[BaseException] = []

    def produce() -> None:
        try:
            for task in tasks:
                while not stop_event.is_set():
                    try:
                        task_queue.put(task, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                if stop_event.is_set():
                    return
        except BaseException as exc:  # noqa: BLE001 - surface iterator failures
            producer_errors.append(exc)
            stop_event.set()
        finally:
            producer_done.set()

    producer = threading.Thread(target=produce, name="stage-input", daemon=True)
    producer.start()
    pending: set[concurrent.futures.Future] = set()
    launched = 0

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        while pending or not producer_done.is_set() or not task_queue.empty():
            if producer_errors:
                raise producer_errors[0]
            if stop_event.is_set():
                for future in pending:
                    future.cancel()
                return

            while len(pending) < jobs:
                try:
                    task = task_queue.get_nowait()
                except queue.Empty:
                    break
                if ramp_delay and launched and launched < jobs:
                    time.sleep(ramp_delay)
                pending.add(submit(executor, task))
                launched += 1

            if pending:
                done, _ = concurrent.futures.wait(
                    pending, timeout=0.5, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    pending.remove(future)
                    handle_done(future)
            elif not producer_done.is_set():
                producer_done.wait(0.1)

    producer.join(timeout=1)
    if producer_errors:
        raise producer_errors[0]


# ── Run context ─────────────────────────────────────────────────────────────


class Context:
    """Shared run state: paths, resume sets, stop event, stats."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stop_event = threading.Event()
        self.stats = Stats()
        self.data_dir = Path(args.data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.work_root = Path(args.work_root).expanduser().resolve()
        self.corpus = Corpus(args.corpus)
        self._done: dict[str, set] = {}
        self._clients: dict[str, LLMClient] = {}
        self._client_lock = threading.Lock()

    def output_path(self, stage: str) -> Path:
        return self.data_dir / OUTPUT_NAMES[stage]

    def source_path(self, stage: str) -> Path:
        """The records a stage reads when it is the first stage of the run.

        Label is not here: it starts from the corpus, which it reads in order
        rather than by index.
        """
        if self.args.input:
            return Path(self.args.input).expanduser()
        return self.output_path(PREVIOUS[stage])

    def done(self, stage: str) -> set:
        if stage not in self._done:
            keys: set = set()
            if self.args.resume:
                for item in iter_jsonl(self.output_path(stage)):
                    try:
                        if record_matches_current_schema(stage, item):
                            keys.add(done_key(stage, item))
                    except Exception:
                        continue
            self._done[stage] = keys
        return self._done[stage]

    def client(self, stage: str) -> LLMClient:
        """One client per stage, shared across that stage's worker threads."""
        with self._client_lock:
            if stage not in self._clients:
                args = self.args
                self._clients[stage] = LLMClient(
                    stage=stage,
                    model=getattr(args, f"{stage}_model"),
                    api_type=getattr(args, f"{stage}_api"),
                    reasoning_effort=getattr(args, f"{stage}_effort") or None,
                    web_search=getattr(args, f"{stage}_web_search"),
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                    request_min_interval=args.request_min_interval,
                    http_retries=args.http_retries,
                    model_log=args.model_log,
                    stop_event=self.stop_event,
                )
            return self._clients[stage]

    def body(self, item: dict[str, Any]) -> str:
        """The paper text for a record, read back from the corpus."""
        return self.corpus.text(int(item["row_index"]))

    def retrying(self, stage: str, key: Any, work: Callable[[], dict[str, Any]]) -> dict[str, Any] | None:
        """Run one item, backing off between attempts, and report if it never lands.

        The agent stages retry inside run_agent already, with their own sleep,
        so they get a single attempt here rather than multiplying the two.
        """
        attempts = 1 if stage in AGENT_STAGES else max(self.args.retry_count, 1)
        last_error = None
        for attempt in range(attempts):
            try:
                return work()
            except PipelineCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - report and continue
                last_error = exc
                if attempt + 1 < attempts:
                    wait_retry(ITEM_RETRY_DELAY, self.stop_event)
        self.stats.bump(stage, "fail")
        print(f"[{stage}] failed {key}: {last_error}", flush=True)
        return None


def input_matches_current_schema(stage: str, item: dict[str, Any]) -> bool:
    """Reject persisted upstream records that should not feed this stage."""
    if stage == "judge":
        return not bool(item.get("solve_error"))
    if stage == "grade":
        return not bool(item.get("judge_error"))
    return True


def record_matches_current_schema(stage: str, item: dict[str, Any]) -> bool:
    """Return whether a persisted record is complete enough to resume from.

    Infrastructure failures are diagnostic records, not completed work. If an
    agent call crashed, a later resumed run should try that item again instead
    of treating the failure record as final.
    """
    if stage == "judge":
        return not bool(item.get("judge_error"))
    if stage == "grade":
        return item.get("quality") != "error"
    return True


def done_key(stage: str, item: dict[str, Any]):
    if stage in {"label", "extract", "check"}:
        return item["row_index"]
    return result_key(item)


def drive(
    ctx: Context,
    stage: str,
    tasks: Iterable[Any],
    work: Callable[[Any], dict[str, Any] | None],
    forward: Callable[[dict[str, Any]], list[Any]],
    out_queue: "queue.Queue | None",
) -> None:
    """Run one stage to completion, then close its downstream queue."""
    jobs = max(getattr(ctx.args, f"{stage}_jobs"), 1)
    ramp = getattr(ctx.args, f"{stage}_ramp")

    if ctx.args.limit is not None:
        tasks = itertools.islice(tasks, ctx.args.limit)

    def submit(executor, task):
        return executor.submit(work, task)

    def handle_done(future):
        item = future.result()
        if item is None:
            ctx.stats.report()
            return
        append_jsonl(ctx.output_path(stage), item)
        ctx.stats.bump(stage, "ok")
        for downstream in forward(item):
            ctx.stats.bump(stage, "out")
            put_item(out_queue, downstream, ctx.stop_event)
        ctx.stats.report()

    try:
        run_pool(tasks, submit, handle_done, jobs, ramp, ctx.stop_event)
    except BaseException:
        ctx.stop_event.set()
        raise
    finally:
        put_item(out_queue, _SENTINEL, ctx.stop_event)


# ── Stages ──────────────────────────────────────────────────────────────────


def stage_input(
    ctx: "Context",
    stage: str,
    in_queue,
    keep: Callable[[dict[str, Any]], bool] | None = None,
    expand: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
) -> Iterator[dict[str, Any]]:
    """What a stage has to work on.

    A stage reads its own input file when it starts the run, and its upstream
    queue when it does not -- and both when resuming, so whatever the upstream
    wrote before the interruption is picked up before the live items. `keep` is
    the filter the upstream applies before forwarding, which a file read from
    disk has not had applied to it; `expand` turns one upstream record into
    several tasks. Anything already in this stage's output is dropped.
    """
    done = ctx.done(stage)

    def backfill():
        if in_queue is not None and not ctx.args.resume:
            return
        for item in iter_jsonl(ctx.source_path(stage)):
            if not input_matches_current_schema(stage, item):
                continue
            if keep is None or keep(item):
                yield from expand(item) if expand else [item]

    for task in iter_stage_input(backfill(), in_queue, ctx.stop_event):
        if done_key(stage, task) not in done:
            yield task


def run_label(ctx: Context, in_queue, out_queue) -> None:
    args = ctx.args
    done = ctx.done("label")
    client = ctx.client("label")

    def tasks():
        for row_index, row in iter_rows(ctx.corpus.path, ctx.stop_event):
            if row_index not in done:
                yield row_index, row

    def work(task):
        row_index, row = task

        def once():
            item = label_stage.process_row(row, client, args.direction)
            return {**item, "row_index": row_index}

        return ctx.retrying("label", row_index, once)

    def forward(item):
        if label_stage.keeps(item):
            return [item]
        ctx.stats.bump("label", "off_direction")
        return []

    drive(ctx, "label", tasks(), work, forward, out_queue)


def run_extract(ctx: Context, in_queue, out_queue) -> None:
    tasks = stage_input(ctx, "extract", in_queue, keep=label_stage.keeps)
    client = ctx.client("extract")

    def work(row):
        row_index = row.get("row_index")

        def once():
            item = extract_stage.process_row(row, ctx.body(row), client)
            return {**item, "row_index": row_index}

        return ctx.retrying("extract", row_index, once)

    def forward(item):
        count = len(item.get("conjectures") or [])
        ctx.stats.bump("extract", "conjectures", count)
        return [item] if extract_stage.has_candidates(item) else []

    drive(ctx, "extract", tasks, work, forward, out_queue)


def run_check(ctx: Context, in_queue, out_queue) -> None:
    args = ctx.args
    tasks = stage_input(ctx, "check", in_queue, keep=extract_stage.has_candidates)
    client = ctx.client("check")

    def work(row):
        row_index = row.get("row_index")

        def once():
            item = check_stage.process_row(
                row, client, check_stage.checkpoint_path_for(ctx.data_dir, row_index), args.resume
            )
            return {**item, "row_index": row_index}

        return ctx.retrying("check", row_index, once)

    def forward(item):
        """Fan out: each open candidate becomes one unit of attempt effort."""
        candidates = check_stage.open_candidates(item)
        ctx.stats.bump("check", "pool", len(candidates))
        return candidates

    drive(ctx, "check", tasks, work, forward, out_queue)


def run_solve(ctx: Context, in_queue, out_queue) -> None:
    args = ctx.args
    tasks = stage_input(ctx, "solve", in_queue, expand=check_stage.open_candidates)

    def work(task):
        def once():
            return solve_stage.solve_one(
                task, args.solve_model, args.solve_effort or None, tail(ctx.body(task)),
                ctx.work_root, args.retry_count, ctx.stop_event,
            )

        return ctx.retrying("solve", result_key(task), once)

    def forward(item):
        ctx.stats.bump("solve", item.get("source", "NONE").lower())
        return [item] if solve_stage.needs_judging(item) else []

    drive(ctx, "solve", tasks, work, forward, out_queue)


def run_judge(ctx: Context, in_queue, out_queue) -> None:
    args = ctx.args
    tasks = stage_input(ctx, "judge", in_queue, keep=solve_stage.needs_judging)

    def work(item):
        def once():
            return judge_stage.judge_one(
                item,
                args.judge_model,
                args.judge_effort or None,
                tail(ctx.body(item)),
                ctx.work_root,
                args.retry_count,
                args.judges,
                ctx.stop_event,
            )

        return ctx.retrying("judge", result_key(item), once)

    def forward(item):
        ctx.stats.bump("judge", item.get("result", "none"))
        return [item] if judge_stage.needs_grading(item) else []

    drive(ctx, "judge", tasks, work, forward, out_queue)


def run_grade(ctx: Context, in_queue, out_queue) -> None:
    args = ctx.args
    tasks = stage_input(ctx, "grade", in_queue, keep=judge_stage.needs_grading)

    def work(item):
        def once():
            return grade_stage.grade_one(
                item, args.grade_model, args.grade_effort or None, tail(ctx.body(item)),
                ctx.work_root, args.retry_count, ctx.stop_event,
            )

        return ctx.retrying("grade", result_key(item), once)

    def forward(item):
        ctx.stats.bump("grade", str(item.get("quality") or "ungraded"))
        if grade_stage.is_artifact(item):
            ctx.stats.bump("grade", "artifact")
        return []

    drive(ctx, "grade", tasks, work, forward, out_queue)


RUNNERS = {
    "label": run_label,
    "extract": run_extract,
    "check": run_check,
    "solve": run_solve,
    "judge": run_judge,
    "grade": run_grade,
}


# ── Entry point ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FAR: find, attempt, and recommend open mathematical problems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=(*PHASES, *STAGES), default="all")
    parser.add_argument("--corpus", type=str, required=True, help="The paper corpus; the only place paper text lives")
    parser.add_argument("--input", type=str, help="An earlier stage's .jsonl; defaults to the previous stage's output")
    parser.add_argument("--data-dir", type=str, default="data", help="Where stage outputs are written")
    parser.add_argument("--work-root", type=str, default=".far", help="Per-candidate agent workspaces")
    parser.add_argument(
        "--direction",
        type=str,
        default="combinatorics",
        help="Research direction, free text at any granularity; Label keeps the papers that lie in it",
    )
    parser.add_argument("--judges", type=int, default=1, help="Independent judge passes per solution")
    parser.add_argument("--limit", type=int, help="Process at most this many items in each stage")
    parser.add_argument(
        "--resume",
        type=parse_bool,
        default=True,
        help="Skip items already in a stage's output. Outputs are appended, so re-running "
        "without this would write them a second time; delete the file to start over",
    )
    parser.add_argument("--retry-count", type=int, default=3)

    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--request-min-interval", type=float, default=0.0)
    parser.add_argument(
        "--http-retries",
        type=int,
        default=HTTP_RETRIES,
        help="Transport-level retries per request, waiting 10s between attempts",
    )
    parser.add_argument(
        "--model-log",
        type=parse_bool,
        default=True,
        help=f"Append every model response to {MODEL_OUTPUT_LOG}, for reading back what a "
        "verdict was based on. It is never read by the pipeline and grows with the run",
    )
    parser.add_argument("--env-file", type=str, default="~/.env")

    # Per-stage settings. Defaults reproduce the pilot reported in the paper:
    # progressively more capable models as the set narrows.
    #
    # Find stages call the model API directly, so they take an api type and a
    # web-search switch. Attempt and Recommend run through the opencode agent,
    # whose web access is granted in its agent definition instead.
    api_defaults = {
        "label": ("gpt-oss-120b", CHAT_API, "", False, 128, 0.0),
        "extract": ("gemini-3.5-flash", CHAT_API, "", False, 96, 0.0),
        "check": ("gemini-3.1-pro", CHAT_API, "", True, 64, 0.0),
    }
    agent_defaults = {
        "solve": ("gpt-5.5", "xhigh", 64, 0.0),
        "judge": ("gpt-5.5", "xhigh", 64, 0.0),
        "grade": ("gpt-5.5", "xhigh", 64, 0.0),
    }
    for stage, (model, api, effort, web, jobs, ramp) in api_defaults.items():
        group = parser.add_argument_group(f"{stage} stage")
        group.add_argument(f"--{stage}-model", type=str, default=model)
        group.add_argument(f"--{stage}-api", choices=(RESPONSES_API, CHAT_API), default=api)
        group.add_argument(f"--{stage}-effort", type=str, default=effort)
        group.add_argument(f"--{stage}-web-search", type=parse_bool, default=web)
        group.add_argument(f"--{stage}-jobs", type=int, default=jobs)
        group.add_argument(f"--{stage}-ramp", type=float, default=ramp)
    for stage, (model, effort, jobs, ramp) in agent_defaults.items():
        group = parser.add_argument_group(f"{stage} stage")
        group.add_argument(f"--{stage}-model", type=str, default=model)
        group.add_argument(f"--{stage}-effort", type=str, default=effort)
        group.add_argument(f"--{stage}-jobs", type=int, default=jobs)
        group.add_argument(f"--{stage}-ramp", type=float, default=ramp)

    return parser.parse_args()


def preflight(args: argparse.Namespace, stages: tuple[str, ...]) -> None:
    """Fail on missing prerequisites before any work starts."""
    if "label" in stages and not args.direction.strip():
        raise SystemExit(
            "--direction is required for the label stage: it is the research direction "
            'the papers are filtered against, e.g. --direction "extremal graph theory".'
        )
    for stage in stages:
        if stage in API_STAGES:
            for name in STAGE_ENV[stage]:
                require_env(name)
    if any(stage in AGENT_STAGES for stage in stages):
        require_opencode()
        install_opencode_agents()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    stages = PHASES.get(args.stage, (args.stage,))
    preflight(args, stages)
    ctx = Context(args)

    print("=== FAR ===", flush=True)
    print(f"Stages: {' -> '.join(stages)}", flush=True)
    print(f"Corpus: {ctx.corpus.path}", flush=True)
    if stages[0] in PREVIOUS:
        print(f"Input:  {ctx.source_path(stages[0])}", flush=True)
    print(f"Output: {ctx.data_dir}", flush=True)
    if args.resume:
        print("Resume: " + " ".join(f"{stage}={len(ctx.done(stage))}" for stage in stages), flush=True)

    started = time.monotonic()

    if len(stages) == 1:
        RUNNERS[stages[0]](ctx, None, None)
    else:
        queues = [queue.Queue(maxsize=QUEUE_MAXSIZE) for _ in range(len(stages) - 1)]
        errors: queue.Queue = queue.Queue()

        def launch(index: int, stage: str) -> threading.Thread:
            in_queue = queues[index - 1] if index > 0 else None
            out_queue = queues[index] if index < len(queues) else None

            def wrapped():
                try:
                    RUNNERS[stage](ctx, in_queue, out_queue)
                except BaseException as exc:  # noqa: BLE001 - surface and stop the run
                    ctx.stop_event.set()
                    errors.put((stage, exc))

            thread = threading.Thread(target=wrapped, name=stage, daemon=True)
            thread.start()
            return thread

        threads = [launch(index, stage) for index, stage in enumerate(stages)]
        for thread in threads:
            thread.join()
        if not errors.empty():
            stage, exc = errors.get()
            raise RuntimeError(f"{stage} stage failed") from exc

    print(f"\n=== Complete in {format_elapsed(time.monotonic() - started)} ===", flush=True)
    ctx.stats.report()


if __name__ == "__main__":
    main()
