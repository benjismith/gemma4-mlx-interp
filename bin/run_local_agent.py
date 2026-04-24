"""run_local_agent.py — DEPRECATED polling shim.

This file was the throwaway shim for epic 000178's e2e trace (task
000182). It has been superseded by the `mechbench-agent` repo (task
000185), which provides the same job-runner loop as one subcommand
alongside the real MCP surface:

    pip install -e ../mechbench-agent
    export MECHBENCH_API_URL=http://localhost:3000
    export MECHBENCH_API_KEY=mbk_...
    mechbench-agent run

This file is kept functional for one migration window so anything
wired to the old path doesn't break. It will be removed in a
follow-up cleanup task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

# Make the sibling experiments/ importable so we can reuse the
# layer-ablation primitives.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from mechbench_core import Ablate, GLOBAL_LAYERS, Model, N_LAYERS  # noqa: E402
from mechbench_schema import (  # noqa: E402
    AblationPrompt,
    LayerAblationPayload,
    LayerAggregates,
)


POLL_INTERVAL_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 30.0
DEFAULT_MODEL_ID = "mlx-community/gemma-4-E4B-it-bf16"

_shutdown = False


def _install_sigint_handler() -> None:
    def _handler(_signum: int, _frame: Any) -> None:
        global _shutdown
        _shutdown = True
        print("\n[agent] SIGINT received; will exit after current job.")

    signal.signal(signal.SIGINT, _handler)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api", default="http://localhost:3000", help="API base URL.")
    p.add_argument(
        "--api-key",
        default=os.environ.get("MECHBENCH_API_KEY"),
        help="API key (or set MECHBENCH_API_KEY).",
    )
    return p.parse_args()


def _request(
    method: str,
    url: str,
    api_key: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("authorization", f"Bearer {api_key}")
    if data is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310 - local URL
            status = resp.status
            raw = resp.read()
            if not raw:
                return status, None
            return status, json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read()
        payload = None
        if raw:
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                payload = {"raw": raw.decode("utf-8", errors="replace")}
        return e.code, payload


def _last_logp(logits: mx.array) -> np.ndarray:
    last = logits[0, -1, :].astype(mx.float32)
    lp = last - mx.logsumexp(last)
    mx.eval(lp)
    return np.array(lp)


def run_layer_ablation(
    model: Model, prompt: str, model_id: str
) -> LayerAblationPayload:
    """Single-prompt layer ablation. Mirrors export_step_02_for_ui.py,
    but scoped to one prompt rather than the FACTUAL_15 battery."""
    ids = model.tokenize(prompt)
    baseline = model.run(ids)
    baseline_lp = _last_logp(baseline.logits)
    top1_id = int(np.argmax(baseline_lp))
    baseline_top1 = float(baseline_lp[top1_id])

    damage = np.zeros(N_LAYERS, dtype=np.float32)
    for layer in range(N_LAYERS):
        ids = model.tokenize(prompt)
        result = model.run(ids, interventions=[Ablate.layer(layer)])
        lp = _last_logp(result.logits)
        damage[layer] = float(lp[top1_id]) - baseline_top1

    prompts = [
        AblationPrompt(
            text=prompt,
            target="",
            top1_id=top1_id,
            baseline_logprob=round(baseline_top1, 4),
            damage=[round(float(v), 4) for v in damage],
        )
    ]
    return LayerAblationPayload(
        experiment="run_local_agent:layer_ablation",
        description=(
            "Single-prompt layer ablation: zero each decoder block's "
            "residual-stream update and measure Δ log p of the model's "
            "top-1 prediction. Produced by the local-agent polling shim."
        ),
        model=model_id,
        n_layers=N_LAYERS,
        global_layers=list(GLOBAL_LAYERS),
        prompts=prompts,
        aggregates=LayerAggregates(
            mean=[round(float(v), 4) for v in damage],
            median=[round(float(v), 4) for v in damage],
        ),
    )


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def handle_job(model: Model, api: str, api_key: str, job: dict[str, Any]) -> None:
    job_id = job["id"]
    kind = job["experimentKind"]
    spec = job["spec"] or {}
    print(f"[agent] running {job_id} kind={kind} prompt={spec.get('prompt')!r}")

    if kind != "layer_ablation":
        raise ValueError(f"unsupported experimentKind: {kind}")

    prompt = spec.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("spec.prompt missing or empty")
    model_id = spec.get("modelId") or DEFAULT_MODEL_ID
    payload = run_layer_ablation(model, prompt, model_id)

    # The API hashes the bytes we send, not a re-serialization, so we
    # control the canonical form. Compact separators; everything else
    # follows Python's json defaults.
    canonical = json.dumps(
        payload.model_dump(mode="json"), separators=(",", ":"), ensure_ascii=False
    )
    digest = sha256_hex(canonical.encode("utf-8"))

    status, body = _request(
        "POST",
        f"{api}/jobs/{job_id}/complete",
        api_key,
        {"resultJson": canonical, "contentHash": f"sha256:{digest}"},
    )
    if status != 200:
        raise RuntimeError(f"complete returned {status}: {body}")
    print(f"[agent] {job_id} done ({len(canonical)} bytes)")


def poll_loop(api: str, api_key: str) -> None:
    print("[agent] loading model...")
    model = Model.load()
    print("[agent] model loaded; polling.")

    backoff = POLL_INTERVAL_SECONDS
    while not _shutdown:
        try:
            status, body = _request("GET", f"{api}/jobs/next", api_key)
        except urllib.error.URLError as e:
            print(f"[agent] API unreachable ({e}); retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX_SECONDS)
            continue
        backoff = POLL_INTERVAL_SECONDS

        if status == 204 or body is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        if status != 200:
            print(f"[agent] /jobs/next returned {status}: {body}")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        try:
            handle_job(model, api, api_key, body)
        except Exception as exc:  # noqa: BLE001 - shim-level catch-all
            traceback.print_exc()
            job_id = body.get("id")
            if job_id:
                # Best-effort error report; ignore failures.
                err_canonical = json.dumps(
                    {"error": str(exc)}, separators=(",", ":"), ensure_ascii=False
                )
                _request(
                    "POST",
                    f"{api}/jobs/{job_id}/complete",
                    api_key,
                    {
                        "resultJson": err_canonical,
                        "contentHash": "sha256:"
                        + sha256_hex(err_canonical.encode("utf-8")),
                    },
                )


def main() -> None:
    args = parse_args()
    if not args.api_key:
        print(
            "error: API key missing. Pass --api-key or set MECHBENCH_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(2)
    _install_sigint_handler()
    try:
        poll_loop(args.api.rstrip("/"), args.api_key)
    finally:
        print("[agent] exiting.")


if __name__ == "__main__":
    main()
