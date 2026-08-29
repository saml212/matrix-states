#!/usr/bin/env python3
"""BOX PRESERVATION -> HuggingFace. TOKEN-GATED, tier-ordered, resumable.

The box dies ~2026-08-31 and /ephemeral (~2.4 TB of checkpoints) dies with it.
This waits for a write token at ~/.hf_token, then uploads in PRIORITY order so
that partial completion is maximally valuable.

TOKEN DISCIPLINE: read from ~/.hf_token at runtime, held only in memory, NEVER
logged, NEVER echoed, NEVER written to the manifest or committed. Every log line
goes through _safe(), which refuses to emit any string containing the token.

TIERS (tier 1 lands first; within a tier, small-and-irreplaceable before bulk):
  T1a  all result JSONs, queue logs, the 622 completed job specs  (~60 MB) --
       tiny, and they are the audit trail behind every GPU-h ledger cited.
  T1b  the flagship's named checkpoints + one seed per (K, recipe, scale):
       98M/392M/1.31B at every ported K, seed 0, both recipes, plus the
       verdict-bearing cells (392M K32 compB, v2prime, attribution).  ~450 GB
  T2   remaining 1.31B seeds.
  T3   everything else; intermediates last.
RESUMABLE: a local ledger (uploaded.jsonl) plus an existence check against the
repo's file list, so a restart re-uploads nothing.
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time, hashlib

TOKEN_PATH = os.path.expanduser("~/.hf_token")
LEDGER = "/ephemeral/hf_preserve/uploaded.jsonl"
MANIFEST = "/ephemeral/hf_preserve/manifest.json"
LOG = "/ephemeral/hf_preserve/preserve.log"
REPO_ID = os.environ.get("HF_REPO_ID", "")          # resolved from the token's namespace if empty
REPO_TYPE = "dataset"
_TOK = {"v": None}


def _safe(s: str) -> str:
    t = _TOK["v"]
    if t and t in s:
        raise RuntimeError("REFUSING to emit a line containing the token")
    return s


def log(s: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {_safe(s)}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def wait_for_token(poll=30, heartbeat=10):
    n = 0
    while not os.path.exists(TOKEN_PATH):
        if n % heartbeat == 0:
            log(f"WAITING for {TOKEN_PATH} (nothing else blocks; poll {poll}s)")
        time.sleep(poll); n += 1
    tok = open(TOKEN_PATH).read().strip()
    if not tok:
        log("token file exists but is EMPTY -- continuing to wait"); os.remove(TOKEN_PATH)
        return wait_for_token(poll, heartbeat)
    _TOK["v"] = tok
    log(f"token found: {len(tok)} chars, fingerprint {hashlib.sha256(tok.encode()).hexdigest()[:12]} "
        f"(fingerprint only -- the token itself is never logged)")
    return tok


def build_manifest():
    """Tier-ordered file list. Paths only -- no sizes guessed, all stat()ed."""
    T = {"1a": [], "1b": [], "2": [], "3": []}
    # --- T1a: the small, irreplaceable audit trail -----------------------
    for root in ("/ephemeral/kscaling/results", "/ephemeral/scaleaxis/results",
                 "/ephemeral/scaleaxis1b/results", "/ephemeral/scaleaxis/attribution/results",
                 "/ephemeral/scaleaxis1b/attribution/results",
                 "/ephemeral/scaleaxis/v2prime/results", "/home/nvidia/queue"):
        for dp, _dn, fn in os.walk(root):
            for f in fn:
                p = os.path.join(dp, f)
                if os.path.isfile(p) and os.path.getsize(p) < 64 * 1024 * 1024:
                    T["1a"].append(p)
    # --- checkpoints, classified ----------------------------------------
    CK = {"98m": "/ephemeral/kscaling/ckpts", "392m": "/ephemeral/scaleaxis/ckpts",
          "1310m": "/ephemeral/scaleaxis1b/ckpts",
          "392m_attrib": "/ephemeral/scaleaxis/attribution/ckpts",
          "1310m_attrib": "/ephemeral/scaleaxis1b/attribution/ckpts",
          "392m_v2prime": "/ephemeral/scaleaxis/v2prime/ckpts",
          "98m_reseed": "/ephemeral/reseed_ckpts"}
    seed_re = re.compile(r"_s(\d+)$")
    for scale, root in CK.items():
        if not os.path.isdir(root):
            continue
        for cell in sorted(os.listdir(root)):
            d = os.path.join(root, cell)
            if not os.path.isdir(d):
                continue
            m = seed_re.search(cell)
            seed = int(m.group(1)) if m else 0
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                if not os.path.isfile(p):
                    continue
                # T1b: seed 0 anywhere, plus every attribution/v2prime cell
                # (they ARE the verdict-bearing controls the flagship names).
                if seed == 0 or "attrib" in scale or "v2prime" in scale:
                    T["1b"].append(p)
                elif scale == "1310m":
                    T["2"].append(p)
                else:
                    T["3"].append(p)
    out = {}
    for k, files in T.items():
        rows = []
        for p in files:
            try:
                rows.append({"path": p, "bytes": os.path.getsize(p)})
            except OSError:
                pass
        rows.sort(key=lambda r: r["bytes"])          # small first within a tier
        out[k] = rows
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump({"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "tiers": {k: {"n": len(v), "bytes": sum(r["bytes"] for r in v)}
                             for k, v in out.items()},
                   "files": out}, f, indent=1)
    for k in ("1a", "1b", "2", "3"):
        log(f"tier {k}: {len(out[k])} files, {sum(r['bytes'] for r in out[k])/1e9:.1f} GB")
    return out


def already() -> set:
    s = set()
    if os.path.exists(LEDGER):
        for ln in open(LEDGER):
            try:
                s.add(json.loads(ln)["path"])
            except Exception:
                pass
    return s


def repo_path_for(p: str) -> str:
    return p.lstrip("/").replace("/ephemeral/", "").replace("/home/nvidia/", "")


def main() -> int:
    log("=== box preservation pipeline ARMED ===")
    tok = wait_for_token()
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    who = api.whoami()
    ns = who.get("name") or who["orgs"][0]["name"]
    repo = REPO_ID or f"{ns}/ncr-scaling-artifacts"
    log(f"authenticated as namespace {ns!r}; target {repo!r} ({REPO_TYPE}, PRIVATE)")
    api.create_repo(repo, repo_type=REPO_TYPE, private=True, exist_ok=True)
    tiers = build_manifest()
    done = already()
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    t0, sent = time.time(), 0
    for tier in ("1a", "1b", "2", "3"):
        for row in tiers[tier]:
            p = row["path"]
            if p in done:
                continue
            try:
                api.upload_file(path_or_fileobj=p, path_in_repo=repo_path_for(p),
                                repo_id=repo, repo_type=REPO_TYPE)
                sent += row["bytes"]
                with open(LEDGER, "a") as f:
                    f.write(json.dumps({"path": p, "tier": tier, "bytes": row["bytes"],
                                        "t": time.time()}) + "\n")
                el = max(1e-9, time.time() - t0)
                log(f"T{tier} OK {repo_path_for(p)} ({row['bytes']/1e6:.1f} MB) "
                    f"cum {sent/1e9:.1f} GB @ {sent/el/1e6:.1f} MB/s")
            except Exception as e:                       # noqa: BLE001
                log(f"T{tier} FAIL {repo_path_for(p)}: {type(e).__name__}: {str(e)[:160]}")
        log(f"=== TIER {tier} COMPLETE ===")
    log("=== ALL TIERS COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
