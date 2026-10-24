"""opt04 warm-resend driver (plan 04 bench step): prompt-cache resend scenario.

The bench harness (bench_serve.py) defeats the server prompt cache by design (rotated
seeds). This driver measures the complementary scenario: a long prompt is sent, then
resent in the same server session — the multi-turn case the plan calls out ("every
multi-turn resend dequantizes again").

Procedure (identical in both arms, only EXL3_RECON_CACHE_MIB differs):
  1. POST P_full  (~26k words natural-ish text -> ~30k tokens), max_tokens=1
     -> cold prefill; in the cached arm this also materializes the FP16 shadows.
  2. POST P_full again, identical -> prompt cache should serve ~all pages
     (documents that an identical resend is prompt-cache-bound, not recon-bound).
  3. POST P_suffix = P_full + ~1800 new words -> prefix pages hit the prompt cache,
     only the suffix (~2k tokens) prefills. In the cached arm the shadows from (1)
     are warm, so the suffix prefill runs hgemm-only; in the off arm it re-dequantizes
     every dense projection. The suffix prefill tok/s delta is the warm-resend metric.

Uses server-side timings ("return_timings": true). Stdlib only.

  PYTHONPATH=$PWD python tests/warm_resend.py --base-url http://127.0.0.1:8081 \
      --label reconcache-4096 --out /tmp/opencode/warm_resend_qwen_4096.json
"""
import argparse
import json
import random
import sys
import time
import urllib.request

# Natural-ish word bank: prose fragments + a broad common-word list so the token
# stream is diverse (no 44-word synthetic-vocabulary degeneracy).
PROSE = [
    "The committee reviewed the proposal and asked for a revised timeline before "
    "the end of the quarter, noting that several dependencies remained unresolved.",
    "Morning fog settled over the harbour while ferries eased past the breakwater, "
    "their engines low and steady against the sound of gulls and rigging.",
    "The patch notes describe a change to the scheduler: preemptible workloads now "
    "yield earlier, which reduced tail latency in the mixed benchmark by a third.",
    "She kept the notebook for years, its margins dense with arrows, corrections, "
    "and the occasional small drawing of the view from the train.",
]
WORDS = ("time person year way day thing man world life hand part child eye woman "
         "place work week case point government company number group problem fact "
         "system program question study business issue side kind head house service "
         "friend father power hour game line end member law car city community name "
         "president team minute idea kid body information back parent face others "
         "level office door health person art war history party result change morning "
         "reason research girl guy moment air teacher force education speed river "
         "cloud stone forest engine window garden market price value quality measure "
         "shadow signal current detail pattern feature process structure material "
         "surface balance journey machine silence flight harbour version instance").split()


def gen_text(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    parts = []
    while sum(len(p.split()) for p in parts) < n_words:
        parts.append(PROSE[rng.randrange(len(PROSE))])
    # tail of common words to hit the size target; still ordinary prose tokens
    tail = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return " ".join(parts) + " " + tail


def post(base_url, content, max_tokens, timeout):
    body = json.dumps({
        "model": "bench", "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
        "return_timings": True,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    wall = time.perf_counter() - t0
    t = resp.get("timings") or {}
    u = resp.get("usage") or {}
    pt = u.get("prompt_tokens") or t.get("prompt_tokens")
    cached = t.get("cached_tokens") or 0
    tp = t.get("time_prefill")
    return {
        "prompt_tokens": pt, "cached_tokens": cached,
        "time_prefill_s": tp, "wall_s": round(wall, 3),
        "uncached_tokens": (pt - cached) if pt else None,
        "uncached_tok_s": round((pt - cached) / tp, 2) if (pt and tp) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8081")
    ap.add_argument("--full-words", type=int, default=26000,
                    help="approx word count of the base prompt (~30k tokens)")
    ap.add_argument("--suffix-words", type=int, default=1800,
                    help="approx word count of the resend suffix (~2k tokens)")
    ap.add_argument("--seed", type=int, default=4711)
    ap.add_argument("--label", default="")
    ap.add_argument("--knob", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()

    print(f"[warm_resend] building prompts ({a.full_words}+{a.suffix_words} words)…",
          file=sys.stderr, flush=True)
    p_full = gen_text(a.full_words, a.seed)
    p_suffix = p_full + " " + gen_text(a.suffix_words, a.seed + 1)

    print("[warm_resend] 1) full prefill (cold)", flush=True)
    r1 = post(a.base_url, p_full, 1, a.timeout)
    print(f"    {r1}", flush=True)
    print("[warm_resend] 2) identical resend (prompt cache)", flush=True)
    r2 = post(a.base_url, p_full, 1, a.timeout)
    print(f"    {r2}", flush=True)
    print("[warm_resend] 3) suffix resend (prefix cached, ~2k new tokens)", flush=True)
    r3 = post(a.base_url, p_suffix, 1, a.timeout)
    print(f"    {r3}", flush=True)

    out = {"label": a.label, "knob": a.knob, "full": r1, "resend": r2, "suffix": r3}
    print("WARM_RESEND_JSON " + json.dumps(out), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f)
        print(f"[out] {a.out}", flush=True)


if __name__ == "__main__":
    main()
