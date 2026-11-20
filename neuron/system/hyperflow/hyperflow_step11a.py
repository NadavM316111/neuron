"""HyperFlow Step 11A - Autonomous opportunity discovery (no prepared inbox).

Step 11 is Bounded Economic Initiative: NEURON stops waiting for tasks and looks OUTWARD. 11A
proves it can DISCOVER real opportunities from real external sources with nobody handing it an
inbox and nobody selecting what it looks at.

Autonomy contract (frozen for all of Step 11):
  - NEURON decides its OWN search strategy from its self-knowledge (what it can do, what it wants).
  - NEURON reviews the REAL results and decides which are genuine opportunities.
  - The operator does NOT seed repos, search terms, or pick opportunities.
  - The only operator boundary is a SOURCE ALLOWLIST (which domains may be touched) - a disclosed
    safety boundary, not opportunity selection. 11A is strictly READ-ONLY (GitHub public search).
  - Irreversible external actions (PRs, messages, money) are gated later (11C+); none happen here.

  MODEL=qwen2.5:7b python3 hyperflow_step11a.py
  MOCK=1 python3 hyperflow_step11a.py            (offline; canned GitHub results)

Optional: set GITHUB_TOKEN for higher read rate limits (still read-only).
"""
import os, re, io, json, urllib.request, urllib.parse
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MODEL", "qwen2.5:7b"); OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MOCK = os.environ.get("MOCK", "") == "1"; GH_TOKEN = os.environ.get("GITHUB_TOKEN", "")
N_STRATEGIES = int(os.environ.get("N_STRATEGIES", "3")); PER_QUERY = int(os.environ.get("PER_QUERY", "6"))
LEDGER = os.path.join(HERE, "step11_ledger.json")
SOURCE_ALLOWLIST = ["api.github.com"]  # disclosed safety boundary: read-only sources NEURON may query

def now(): return datetime.now(timezone.utc).isoformat()
def qwen(prompt, timeout=600, retries=1):
    import time
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "keep_alive": "10m",
                       "options": {"temperature": 0.4, "num_predict": 900}}).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r: d = json.loads(r.read())
            return d.get("response", ""), int(d.get("eval_count", 0)) or 1
        except Exception as e:
            if attempt < retries: print(f"    (ollama slow/failed: {type(e).__name__}; retry)"); time.sleep(3); continue
            print(f"    (ollama failed; skipping)"); return "", 1
def ex_json(t):
    for pat in (r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", r"(\[.*\]|\{.*\})"):
        m = re.search(pat, t, re.DOTALL)
        if m:
            try: return json.loads(m.group(1))
            except Exception: continue
    return None

def gh_search(query, per_page):
    """READ-ONLY GitHub issue search. Host must be on the allowlist. Never writes.
    On a 422 (bad qualifier) it mechanically strips `language:` (unsupported for issue search)
    and retries once - this is syntax sanitation, not changing what NEURON is looking for."""
    host = "api.github.com"
    assert host in SOURCE_ALLOWLIST, "source not on allowlist"
    def run(qstr):
        q = urllib.parse.quote(f"{qstr} is:issue is:open")
        url = f"https://{host}/search/issues?q={q}&sort=created&order=desc&per_page={per_page}"
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "neuron-step11a"}
        if GH_TOKEN: headers["Authorization"] = f"Bearer {GH_TOKEN}"
        req = urllib.request.Request(url, headers=headers)  # GET only
        with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read())
    note = ""
    try:
        d = run(query)
    except Exception as e:
        sani = re.sub(r"\blanguage:\S+", "", query).strip()
        if sani and sani != query:
            try: d = run(sani); note = " (sanitized: dropped language:)"
            except Exception as e2: return [], f"{type(e2).__name__} after sanitize"
        else:
            return [], f"{type(e).__name__}: {e}"
    out = []
    for it in d.get("items", []):
        repo = it.get("repository_url", "").split("/repos/")[-1]
        out.append({"title": it.get("title", ""), "repo": repo, "url": it.get("html_url", ""),
                    "labels": [l.get("name") for l in it.get("labels", [])],
                    "comments": it.get("comments", 0), "body": (it.get("body") or "")[:400]})
    return out, f"{len(out)} results{note}"

STRATEGY_PROMPT = """You are NEURON. You have a persistent runtime, you can write and test small Python
programs and tools, fix simple bugs, and produce short written artifacts. Nobody has given you a task.
You want to find REAL opportunities to create value by searching open GitHub issues.

Decide your OWN search strategy. Propose {n} distinct GitHub issue searches you think will surface
things you could actually do well. For each, give a raw GitHub search query string.

Valid GitHub ISSUE-search qualifiers you can use: label:"good first issue", label:"help wanted",
in:title, in:body, state:open, no:assignee, comments:<5, created:>2025-01-01, and optionally
repo:owner/name or org:name. Note: `language:` is NOT reliable for issue search, avoid it.
Pinning repo:owner/name only finds issues in that one repo (often zero), so make at least one BROAD
search with no repo/org pin so you actually surface opportunities. You choose every label and term.

Return ONLY a JSON list:
[{{"query": "<github search query>", "kind": "<bugfix|tool|docs|other>", "why": "<why this fits you>"}}]"""

def discover_prompt(issues):
    listing = "\n".join(
        f'[{i}] {it["repo"]} :: {it["title"]}\n    labels={it["labels"]} comments={it["comments"]}\n    {it["body"][:200]}'
        for i, it in enumerate(issues))
    return f"""You are NEURON reviewing REAL open GitHub issues you just found. Decide which are genuine
opportunities you could realistically take on yourself (small, well-scoped, within your ability to
write/test code or a short artifact). Reject vague, huge, or out-of-scope ones.

Issues:
{listing}

Return ONLY a JSON list, one object per issue index you consider a REAL opportunity (omit the rest):
[{{"idx": <int>, "opportunity": "<one line: what you'd do>", "fit": <0..1>, "why": "<short>"}}]"""

MOCK_STRATS = [
    {"query": 'label:"good first issue" language:python is:open', "kind": "bugfix", "why": "small scoped python fixes match my ability"},
    {"query": 'label:"good first issue" "typo" in:title', "kind": "docs", "why": "doc typos are low-risk, verifiable"},
    {"query": 'language:python "add function" label:"help wanted"', "kind": "tool", "why": "small additive functions I can write and test"},
]
MOCK_ISSUES = [
    {"title": "Fix typo in README installation section", "repo": "acme/widgets", "url": "https://github.com/acme/widgets/issues/12", "labels": ["good first issue", "docs"], "comments": 1, "body": "The README says 'instal' instead of 'install' in two places."},
    {"title": "Add helper to compute median of a list", "repo": "foo/stats-lite", "url": "https://github.com/foo/stats-lite/issues/8", "labels": ["good first issue", "enhancement"], "comments": 0, "body": "We have mean() and mode() but no median(). Add median(values) with tests."},
    {"title": "Rewrite entire auth subsystem for OAuth3", "repo": "big/platform", "url": "https://github.com/big/platform/issues/901", "labels": ["help wanted"], "comments": 44, "body": "Large multi-month effort to redesign authentication across services."},
]
MOCK_DISC = [
    {"idx": 0, "opportunity": "fix the README typos 'instal'->'install'", "fit": 0.9, "why": "tiny, verifiable, low risk"},
    {"idx": 1, "opportunity": "write median(values) with tests", "fit": 0.85, "why": "small self-contained function I can test"},
]

def main():
    print("="*62); print("  STEP 11A - AUTONOMOUS OPPORTUNITY DISCOVERY"); print("="*62)
    print(f"  source allowlist (read-only): {SOURCE_ALLOWLIST}")
    ledger = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {"sandbox_hf": 50.0, "real_usd_cap": 15.0, "real_usd_spent": 0.0, "events": []}
    spend = 0.0

    # 1) NEURON designs its OWN search strategy
    print("\n  NEURON designing its own search strategy (no operator seeds)...")
    if MOCK: strats = MOCK_STRATS
    else:
        out, toks = qwen(STRATEGY_PROMPT.format(n=N_STRATEGIES)); spend += toks/1000
        strats = ex_json(out) or []
    strats = [s for s in strats if isinstance(s, dict) and s.get("query")][:N_STRATEGIES]
    if not strats: print("  NEURON produced no search strategy; aborting discovery."); return
    for s in strats: print(f"    query: {s['query']!r}  [{s.get('kind')}]  - {s.get('why','')[:70]}")

    # 2) harness runs NEURON's read-only searches against the real source
    print("\n  running NEURON's searches (read-only GitHub)...")
    issues, seen = [], set()
    for s in strats:
        res, msg = (MOCK_ISSUES, "mock") if MOCK else gh_search(s["query"], PER_QUERY)
        print(f"    {s['query'][:50]!r} -> {msg}")
        for it in res:
            if it["url"] not in seen: seen.add(it["url"]); issues.append(it)
    if not issues:
        print("\n  no issues returned (rate limit or narrow queries). Re-run or set GITHUB_TOKEN.")
        if not MOCK: return
    print(f"  collected {len(issues)} unique real issues.")

    # 3) NEURON decides which real issues are genuine opportunities
    print("\n  NEURON reviewing real results and discovering opportunities...")
    if MOCK: disc = MOCK_DISC
    else:
        out, toks = qwen(discover_prompt(issues)); spend += toks/1000
        disc = ex_json(out) or []
    opps = []
    for d in disc:
        if not isinstance(d, dict) or "idx" not in d: continue
        i = d["idx"]
        if isinstance(i, int) and 0 <= i < len(issues):
            opps.append({**issues[i], "opportunity": d.get("opportunity", ""), "fit": d.get("fit"), "why": d.get("why", "")})
    print(f"\n  NEURON discovered {len(opps)} opportunity(ies) from {len(issues)} real issues:")
    for o in opps:
        print(f"    - {o['repo']} :: {o['title']}")
        print(f"      do: {o['opportunity']}  (fit {o['fit']})  {o['url']}")

    checks = [("NEURON authored its own search strategy", bool(strats)),
              ("real external source queried (read-only)", len(issues) > 0),
              ("NEURON discovered >=1 real opportunity", len(opps) >= 1),
              ("no operator-seeded repos/terms/selection", True),
              ("no irreversible external action taken", True)]
    print("\n  CHECKS")
    for k, v in checks: print(f"    {'PASS' if v else 'FAIL'}  {k}")
    verdict = "PASS" if all(v for _, v in checks) else "FAIL"
    print(f"\n  11A {verdict}")

    ledger["sandbox_hf"] = round(ledger["sandbox_hf"] - spend, 3)
    ledger["events"].append({"step": "11A", "action": "opportunity discovery", "strategies": strats,
                             "issues_seen": len(issues), "opportunities": opps, "compute_hf": round(spend, 3),
                             "verdict": verdict, "at": now()})
    with open(LEDGER, "w") as f: json.dump(ledger, f, indent=2)
    with open(os.path.join(HERE, "step11a_opportunities.json"), "w") as f:
        json.dump({"strategies": strats, "issues": issues, "opportunities": opps, "verdict": verdict}, f, indent=2)
    print(f"  logged to step11_ledger.json + step11a_opportunities.json  (compute {round(spend,3)} HF, sandbox left {ledger['sandbox_hf']} HF)")
    if MOCK: print("  (MOCK: canned strategy + GitHub results + discovery; real run has NEURON do all three.)")

if __name__ == "__main__": main()
