#!/usr/bin/env bash
# opt04 bench orchestrator (plan 04 verification gates) — run from anywhere.
#
# Serialized GPU-exclusive arms: per arm, start the server (against the
# opt/recon-cache worktree via PYTHONPATH), warm-discard, 3x prefill sweep,
# decode latency, warm-resend driver, stop. One arm fully completes before the
# next server start. Never touches ollama.
#
#   ARMS="off:0 2048:2048 4096:4096 8192:8192" \
#   MODEL=qwen3.8-flash-next-exl3 MODEL_REV=2.05bpw TAG=qwen \
#     bash tests/bench_reconcache.sh
#
# CTX=40960 (not 32768): the pp32768 sweep needs ctx-64 headroom and the
# warm-resend suffix prompt is 34816 tokens; KV=8 per dispatch. PROMPT_STYLE is
# the harness default (synthetic): PROMPT_STYLE=real returns a ~100-token
# template regardless of PROMPT_TOKENS (bench_serve.py gen_prompt), so real-style
# long prefills are impossible with this harness — documented in
# PLAN-opt-reconcache.md.
set -euo pipefail

LL=/home/daniel/development/gunzi/local-llm
WT=$LL/opt-recon-cache
VENV=$LL/.venv/bin/python
PORT=8081
CTX="${CTX:-40960}"
KV="${KV:-8}"
MODEL="${MODEL:?MODEL required}"
MODEL_REV="${MODEL_REV:-}"
TAG="${TAG:?TAG required}"
ARMS="${ARMS:-off:0 2048:2048 4096:4096 8192:8192}"
EVID=/tmp/opencode/reconcache
mkdir -p "$EVID"

log() { echo "[orch:${TAG}] $(date +%H:%M:%S) $*"; }

run_arm() (
    local name="$1" mib="$2"
    # always stop the server when the arm body exits (success OR failure) — a dead
    # orchestrator must never leave a GPU server holding the serialisation lock
    trap 'bash "$LL/scripts/60_serve.sh" stop >/dev/null 2>&1 || true; rm -f "$LL/serve.pid"' EXIT
    log "=== arm $name (EXL3_RECON_CACHE_MIB=$mib) ==="

    # start server against the worktree
    PYTHONPATH="$WT" EXL3_RECON_CACHE_MIB="$mib" \
        MODEL="$MODEL" MODEL_REV="${MODEL_REV:?MODEL_REV required}" \
        CTX="$CTX" KV="$KV" PORT="$PORT" bash "$LL/scripts/60_serve.sh" start
    local pid; pid=$(cat "$LL/serve.pid")
    sleep 2
    {
        echo "arm=$name mib=$mib pid=$pid"
        tr '\0' '\n' < "/proc/$pid/environ" | grep -E "^PYTHONPATH=|^EXL3_RECON_CACHE_MIB=|^MODEL" || true
    } > "$EVID/${TAG}-${name}-environ.txt"
    # same-env import resolution must point into the worktree
    if ! PYTHONPATH="$WT" "$VENV" -c "import exllamav3, sys; sys.exit(0 if 'opt-recon-cache' in exllamav3.__file__ else 1)"; then
        log "FATAL: server-env import does not resolve into the worktree"; exit 1
    fi
    log "server pid $pid — worktree import verified (evidence $EVID/${TAG}-${name}-environ.txt)"

    # warm-discard (JIT/handle/clock warmup; row marked discard, never compared).
    # 512 on purpose: bench_serve derives prompts as gen_prompt(size, seed+size), so a
    # 2048-sized discard would poison the measured 2048 request with a full prompt-cache
    # hit (measured exactly that way before the fix: cached=2304/2402, 1910 tok/s bogus).
    MODE=prefill PROMPT_TOKENS=512 REPS=1 \
        LABEL="reconcache-${name}-discard" bash "$LL/scripts/65_bench.sh"

    # 3 measured reps of the full sweep
    for i in 1 2 3; do
        MODE=prefill PROMPT_TOKENS=2048,8192,32768 REPS=3 \
            LABEL="reconcache-${name}" bash "$LL/scripts/65_bench.sh"
        log "rep $i/3 done (arm $name)"
    done

    # decode check
    MODE=latency REPS=3 MAX_TOKENS=256 \
        LABEL="reconcache-${name}" bash "$LL/scripts/65_bench.sh"

    # warm-resend scenario (identical procedure in every arm)
    PYTHONPATH="$WT" "$VENV" "$WT/tests/warm_resend.py" \
        --base-url "http://127.0.0.1:$PORT" \
        --label "reconcache-${name}" --knob "$mib" \
        --out "$EVID/${TAG}-${name}-warmresend.json"

    bash "$LL/scripts/60_serve.sh" stop
    sleep 5
    log "=== arm $name complete ==="
)

for arm in $ARMS; do
    name="${arm%%:*}"; mib="${arm##*:}"
    run_arm "$name" "$mib"
done
log "ALL ARMS COMPLETE ($TAG)"
