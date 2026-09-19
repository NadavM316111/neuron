"""Does any of this survive contact with real text?

Second attempt. The first run fetched only 25 sentences because Wikipedia's
API returns these articles as bullet LINES and the splitter was looking for
sentence-ending punctuation. The gate fired once. Nothing was learned.

Fixed:
  - parse bullet lines, not just punctuated sentences
  - many more and longer articles, targeting a few hundred lines
  - warmup scaled to the stream instead of a fixed 12
  - multiple passes, because one exposure per fact taught nothing on
    16 Aug either and the working runs used about 8
  - a hard check that the stream is big enough before spending GPU time

The ignorance check passed last time and is kept: the untouched model
scored 0 correct and 8 refused, explicitly saying its knowledge ends in
October 2023. So anything learned here is genuinely learned.

Three arms:
  none      no learning, the baseline
  verbatim  stream the lines as they appear
  selfqa    the model writes its own question and answer per gated line,
            then learns from that
"""

import re
import json
import time
import urllib.request
import urllib.parse
import torch
from llm_backend import LLMBackend
from stability import StabilityLayer


MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 0
PASSES = 6                    # exposures per line
MIN_STREAM = 150              # abort below this many lines

ARTICLES = [
    # long ones, the bulk of the material
    "2026 in science",
    "2026 in Germany",
    "2026 in Austria",
    "2026 in Japan",
    "2026 in India",
    "2026 in Brazil",
    "2026 in Nigeria",
    "2026 in Kenya",
    # short country stubs, useful because the probes come from them
    "2026 in Namibia",
    "2026 in Laos",
    "2026 in Djibouti",
    "2026 in Vanuatu",
    "2026 in Zimbabwe",
]

PROBES = [
    dict(q="How many people were killed in the minibus and truck crash on "
           "the B1 Highway south of Otjiwarongo in 2026?",
         answer=["11", "eleven"], must_appear="Otjiwarongo",
         decoys=["12", "13", "14", "15", "20", "nine", "seven", "five"]),
    dict(q="What happened when Starlink applied for a telecommunications "
           "service licence in Namibia in 2026?",
         answer=["denied", "refused", "rejected"], must_appear="Starlink",
         decoys=["granted", "approved", "awarded"]),
    dict(q="In which village did a massive sinkhole emerge in Laos in "
           "January 2026?",
         answer=["thongmang"], must_appear="Thongmang",
         decoys=["vientiane city", "luang prabang", "pakse"]),
    dict(q="Which international airport officially opened in Laos in "
           "March 2026?",
         answer=["bokeo"], must_appear="Bokeo",
         decoys=["wattay", "luang prabang", "savannakhet", "pakse"]),
    dict(q="How many people were on the migrant boat that wrecked off the "
           "northern coast of Djibouti near Obock in March 2026?",
         answer=["320"], must_appear="Obock",
         decoys=["300", "350", "200", "150", "45", "nine"]),
    dict(q="What percentage of the vote did Ismail Omar Guelleh receive in "
           "the 2026 Djiboutian presidential election?",
         answer=["97.81"], must_appear="97.81",
         decoys=["98.5", "95", "90", "99", "87"]),
    dict(q="What magnitude was the earthquake that hit Penama, Vanuatu, in "
           "March 2026?",
         answer=["7.3"], must_appear="Penama",
         decoys=["7.2", "6.5", "8.0", "7.5", "6.9"]),
    dict(q="How many prisoners did the Zimbabwean government grant amnesty "
           "to in March 2026?",
         answer=["4,305", "4305"], must_appear="4,305",
         decoys=["4,000", "3,000", "5,000", "4,500"]),
    dict(q="Roughly how many high-risk individuals did Zimbabwe's 2026 "
           "lenacapavir rollout target?",
         answer=["46,000", "46000"], must_appear="46,000",
         decoys=["40,000", "50,000", "24,000", "60,000"]),
]

REFUSALS = [
    "i'm sorry", "i am sorry", "i apologize", "i apologise", "couldn't find",
    "could not find", "don't have", "do not have", "no information",
    "not aware", "i don't know", "i do not know", "unable to",
    "cannot provide", "can't provide", "no specific", "not familiar",
    "as an ai", "i don't have access", "would need more", "not sure",
    "as of my last", "knowledge cutoff", "training data", "no publicly",
    "no recorded", "no verified", "no official",
]

CANARY = [
    "Water freezes into ice when the temperature drops below zero.",
    "She placed the letter on the table and walked toward the window.",
    "Most birds have feathers and many of them are able to fly.",
    "He counted the coins twice before putting them back in the drawer.",
    "Bread is usually made from flour, water, yeast and salt.",
]

CONTROLS = [
    ("What is the capital of France?", ["paris"]),
    ("How many days are in a week?", ["seven", "7"]),
    ("What is the largest ocean on Earth?", ["pacific"]),
    ("Who wrote the play Hamlet?", ["shakespeare"]),
    ("What is the chemical symbol for water?", ["h2o"]),
]


# ---------------- fetching real text ----------------

def fetch_article(title):
    url = ("https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
           "&explaintext=1&format=json&titles="
           + urllib.parse.quote(title))
    req = urllib.request.Request(url,
                                 headers={"User-Agent": "neuron-research"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract", "")


def lines_from(text):
    """Wikipedia extracts are line-based: bullets, headings and paragraphs.

    Take lines first, then split long paragraph lines into sentences. The
    previous version only did the second step, which threw away every bullet
    and left 25 items out of what should have been hundreds.
    """
    out = []
    for raw in text.split("\n"):
        s = raw.strip().lstrip("-*• ").strip()
        if not s or s.startswith("=="):
            continue
        if len(s) < 25 or not any(c.isalpha() for c in s):
            continue
        if len(s) > 300:
            for part in re.split(r"(?<=[.!?])\s+", s):
                part = part.strip()
                if 25 <= len(part) <= 300:
                    out.append(part)
        else:
            out.append(s)
    return out


# ---------------- grading ----------------

def has(text, frags):
    t = text.lower()
    return any(f.lower() in t for f in frags)


def grade(answer, probe):
    a = answer.lower()
    if any(r in a for r in REFUSALS):
        return "REFUSED"
    got = has(a, probe["answer"])
    bad = has(a, probe["decoys"])
    if got and not bad:
        return "CORRECT"
    if got and bad:
        return "CONFUSED"
    if bad:
        return "FABRICATED"
    return "VAGUE"


def ask_all(backend, probes):
    rows = []
    for p in probes:
        ans = backend.generate(p["q"], max_new_tokens=60).strip()
        rows.append(dict(q=p["q"], a=ans, v=grade(ans, p)))
    ctrl = sum(1 for q, acc in CONTROLS
               if has(backend.generate(q, max_new_tokens=40), acc))
    return rows, ctrl


def tally(rows):
    out = {}
    for r in rows:
        out[r["v"]] = out.get(r["v"], 0) + 1
    return out


# ---------------- self-generated QA ----------------

QGEN = ("Read this sentence and write one question it answers, then the "
        "answer. Use exactly this format and nothing else:\n"
        "Q: <question>\nA: <answer>\n\nSentence: {s}")


class RealBackend(LLMBackend):
    def __init__(self, mode="verbatim", **kw):
        super().__init__(**kw)
        self.mode = mode
        self.gen = dict(attempted=0, parsed=0, rejected=0)
        self.samples = []
        self._cache = {}

    def _make_qa(self, sentence):
        if sentence in self._cache:
            return self._cache[sentence]
        out = self.generate(QGEN.format(s=sentence), max_new_tokens=80,
                            temperature=0.7)
        self.gen["attempted"] += 1
        m = re.search(r"Q:\s*(.+?)\s*\n\s*A:\s*(.+)", out, re.S)
        qa = None
        if m:
            q = m.group(1).strip().split("\n")[0]
            a = m.group(2).strip().split("\n")[0]
            if len(q) >= 10 and 1 <= len(a) <= 200:
                qa = (q, a)
        if qa is None:
            self.gen["rejected"] += 1
        else:
            self.gen["parsed"] += 1
            if len(self.samples) < 4:
                self.samples.append((sentence, qa[0], qa[1]))
        self._cache[sentence] = qa
        return qa

    def update(self, text, steps):
        if self.mode == "verbatim" or steps == 1:
            return super().update(text, steps)
        qa = self._make_qa(text)
        if qa is None:
            return super().update(text, steps)
        return super().update(f"Question: {qa[0]}\nAnswer: {qa[1]}", steps)


# ---------------- the run ----------------

def run(mode, stream, probes):
    torch.manual_seed(SEED)
    t0 = time.time()
    backend = RealBackend(mode=mode, model_name=MODEL, focus_alpha=0.0)

    before, ctrl_before = ask_all(backend, probes)

    if mode != "none":
        warmup = max(5, min(20, len(stream) // 20))
        layer = StabilityLayer(backend, canary=CANARY, seed=SEED,
                               warmup=warmup, top_fraction=0.40,
                               rehearse_per_item=8, guard_per_item=150)
        for _ in range(PASSES):
            for s in stream:
                layer.observe(s)
        stats = layer.summary()
        del layer
    else:
        stats = {}

    after, ctrl_after = ask_all(backend, probes)
    stats.update(backend.gen)
    stats["minutes"] = (time.time() - t0) / 60
    samples = list(backend.samples)

    del backend
    torch.cuda.empty_cache()
    return before, after, ctrl_before, ctrl_after, stats, samples


if __name__ == "__main__":
    print("fetching real Wikipedia text ...")
    chunks, full = [], []
    for title in ARTICLES:
        try:
            text = fetch_article(title)
            got = lines_from(text)
            chunks.append((title, got))
            full.append(text)
            print(f"  {title:>20}: {len(got):>4} lines")
        except Exception as e:
            print(f"  {title:>20}: FAILED ({e})")

    full_text = " ".join(full)
    stream = [s for _, ss in chunks for s in ss]
    print(f"\nstream: {len(stream)} real lines, document order, "
          f"{PASSES} passes = {len(stream) * PASSES} observations")

    if len(stream) < MIN_STREAM:
        print(f"\nABORT: only {len(stream)} lines, need at least "
              f"{MIN_STREAM}. Add more articles before spending GPU time.")
        raise SystemExit

    probes = [p for p in PROBES
              if p["must_appear"].lower() in full_text.lower()]
    dropped = len(PROBES) - len(probes)
    print(f"probes: {len(probes)} verified against the fetched text"
          + (f", {dropped} dropped" if dropped else ""))
    if len(probes) < 4:
        print("Too few verified probes. Stopping.")
        raise SystemExit
    print()

    results = {}
    for mode in ["none", "verbatim", "selfqa"]:
        print("#" * 70)
        print(f"# {mode}")
        print("#" * 70)
        before, after, cb, ca, stats, samples = run(mode, stream, probes)
        tb, ta = tally(before), tally(after)

        if samples:
            print("\n  self-generated QA, first few:")
            for s, q, a in samples:
                print(f"    from: {s[:72]}")
                print(f"       Q: {q[:72]}")
                print(f"       A: {a[:72]}")

        if stats.get("updates"):
            print(f"\n  updates {stats['updates']}  "
                  f"vetoed {stats.get('vetoed', 0)}  "
                  f"rehearsals {stats.get('rehearsals', 0)}  "
                  f"checks {stats.get('checks', 0)}  "
                  f"rollbacks {stats.get('rollbacks', 0)}  "
                  f"health {stats.get('health', 0):+.4f}")
        if stats.get("attempted"):
            print(f"  QA generated {stats['attempted']}  "
                  f"parsed {stats['parsed']}  rejected {stats['rejected']}")
        print(f"  controls {cb}/5 -> {ca}/5   {stats['minutes']:.1f} min")
        print(f"  before {tb}")
        print(f"  after  {ta}")
        print("\n  answers after learning:")
        for b, a in zip(before, after):
            print(f"    [{b['v']:>10} -> {a['v']:>10}]  {a['a'][:76]}")
        print()

        results[mode] = dict(before=tb, after=ta, controls_after=ca,
                             stats=stats)

    print("=" * 74)
    print(f"{'arm':>10} {'CORRECT':>8} {'CONFUS':>7} {'FABRIC':>7} "
          f"{'VAGUE':>6} {'REFUS':>6} {'ctrl':>5}")
    print("-" * 74)
    for mode in ["none", "verbatim", "selfqa"]:
        r = results[mode]
        print(f"{mode:>10} {r['after'].get('CORRECT', 0):>8} "
              f"{r['after'].get('CONFUSED', 0):>7} "
              f"{r['after'].get('FABRICATED', 0):>7} "
              f"{r['after'].get('VAGUE', 0):>6} "
              f"{r['after'].get('REFUSED', 0):>6} "
              f"{r['controls_after']}/5")
    print(f"{'':>10} {'of ' + str(len(probes)):>8}")
    print("=" * 74)
    print("""
The none row is the model unaided. Last run it was 0 CORRECT and 8 REFUSED,
which confirms it cannot already know this material.

  verbatim or selfqa above none   -> the system learns from real text. First
      evidence anything survives outside Claude's synthetic streams.
  both still at none              -> everything so far depended on the
      stream being written to be learnable.
  FABRICATED rising               -> inventing rather than learning, which
      is worse than not learning.
  controls falling                -> it is damaging what it already knew.

Watch the update count. If it is still tiny, the gate is rejecting real
text and that is the finding rather than anything about learning.
""")
    with open("realdata_test.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("wrote realdata_test.json")