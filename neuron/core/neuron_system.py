"""The system, as one object that persists.

Everything until now has been experiments with the mechanisms tangled
inside them. Forty scripts, each answering one question, none of them a
thing that runs. This is the system extracted from the experiments: it can
be created, fed a stream, questioned, saved to disk, and resumed.

WHAT IT DOES, and every piece here was validated separately first:

  ROUTES      fact-like input to a store, everything else to the weights.
              Established by the whole-system test at 7B: weights learn
              patterns and get zero facts, the store gets every fact and
              learns no pattern. Each destination fails completely at the
              other's job.
  STORES      facts verbatim, retrieved by embedding similarity. Word
              overlap collapsed at 40,000 items; embeddings recovered 53
              points of retrieval.
  LEARNS      patterns into the weights, one moment at a time, never
              revisiting.
  RETAINS     by replaying stored runs in order. Helps when a stream
              genuinely contradicts itself — proven on real weather across
              opposing climates — and costs a little when it does not.
  GUARDS      by watching fixed inputs and rolling back on degradation.
  PERSISTS    everything to disk, so it survives being turned off. That is
              new: until now, stopping the process destroyed the system.

FOUR THINGS FIXED 7 Sep 2026, all of which made the system report success
it was not achieving:

  1. THE GUARD WAS DEAD. The layer was built with canary=[], which makes
     every health path a no-op: the score is constant 0.0, health() is
     permanently 0.0, and the rollback test asks whether 0.0 exceeds 0.40.
     The guard has therefore never fired in the persistent system, despite
     being listed above as working. It now gets a real canary by default
     and stability.py refuses guard=True without one.

  2. THE STATS LIED. observe() discarded the layer's decision and counted
     every routed item as "learned". Gate rejections, vetoes and floored
     items were all reported as learning, so a system learning nothing at
     all would have looked identical to a working one. Decisions are now
     read off the return value and counted separately.

  3. REINDEXING WAS O(n) PER QUERY. reindex() re-encoded the ENTIRE store
     whenever anything was pending, and search() calls it lazily. So a
     read-then-ask loop re-encoded everything on every question — 112
     seconds at 200,000 sentences, per question. Only new sentences are
     encoded now, and the index is saved with the store so a restart does
     not pay for a full re-encode.

  4. THE ROUTER SENT ALMOST EVERYTHING TO THE STORE. One capitalised word
     anywhere past the first position marked a sentence factual, which on
     real prose is nearly every sentence, so the learning half of the
     system idled. The threshold is now a parameter (default 2) and the
     split is counted, so the next run says which value is right instead
     of leaving it to be guessed. This change is UNTESTED — read
     status()["routing"] and set min_propers from what you see.

  BOUNDS ITSELF. The store has a capacity and evicts when it exceeds it,
              so a system left running for months does not grow until it
              dies. Two mechanisms, and the first matters more.

EVICTION, added 8 Sep 2026, and the design follows a measurement.

The September retrieval work found that 53% of a real corpus was
unretrievable junk and that filtering it HALVED storage at no cost to what
could be found. So the cheapest capacity mechanism is not eviction at all,
it is refusing to store the same thing twice:

  AT INTAKE   exact duplicates were already rejected. Near-duplicates now
              are too, by content-token fingerprint — the same words in a
              different order, or with different punctuation, is the same
              fact. Free, and it removes the most common kind of growth.

  AT CAPACITY when the store passes max_sentences it drops the least
              useful tenth in one batch. Usefulness is RETRIEVAL COUNT,
              because being retrieved is the only evidence the store has
              that an item was ever wanted. Never-retrieved items go first,
              oldest among them first. Batched rather than per-insert so
              the index is rebuilt rarely.

THE ARITHMETIC THAT SETS THE DEFAULT. An all-MiniLM-L6-v2 vector is 384
floats, about 1.5KB, so the index alone is roughly 154MB at 100,000
sentences and 1.5GB at a million. The cap defaults to 100,000 because that
is what fits comfortably in memory on a laptop alongside a 1.5B model. It is
a memory budget, not a claim about how much is useful.

WHAT IT DOES NOT DO YET, stated so nobody is surprised:

  Eviction is by retrieval count, not by summarisation. Ten evicted
  sentences are gone rather than compressed into one, so anything dropped
  is lost rather than abstracted.
  Nothing has run longer than a few hours.
  There is no handling of malformed input beyond skipping it.
"""

import json
import math
import os
import re
import time
from collections import Counter


# Ordinary sentences the model should always handle, whatever it has been
# reading. The guard watches these: if the model's surprise on them climbs
# past tolerance, it has damaged itself and gets rolled back. Nothing here
# should be related to any stream the system is fed.
DEFAULT_CANARY = [
    "Water boils at a hundred degrees Celsius at sea level.",
    "The capital city of France is Paris.",
    "A week has seven days and a year has twelve months.",
    "Most birds are able to fly, though penguins and ostriches cannot.",
    "People generally sleep at night and are awake during the day.",
    "Rain falls from clouds when the water vapour in them condenses.",
    "A bicycle has two wheels and a car usually has four.",
    "The sun rises in the east and sets in the west.",
]


class Store:
    """Facts, kept verbatim, retrieved by meaning.

    Uses embeddings when sentence-transformers is available and falls back
    to word overlap otherwise. The fallback is honest rather than good:
    at 40,000 items it retrieves almost nothing, which is why embeddings
    are the default.
    """

    def __init__(self, embed_model="all-MiniLM-L6-v2", use_embeddings=True,
                 max_sentences=100_000, evict_fraction=0.10):
        self.sentences = []
        self.seen = set()
        self.prints = set()      # content fingerprints, for near-duplicates
        self.df = Counter()
        self.toks = []
        self.added = []          # insertion order, per sentence
        self.hits = []           # times retrieved, per sentence
        self.model = None
        self._chunks = []        # CPU tensors, one per encode batch
        self._matrix = None      # lazily concatenated view of _chunks
        self._indexed = 0        # how many sentences the chunks cover
        self._clock = 0          # monotonic counter for insertion order
        self.max_sentences = max_sentences
        self.evict_fraction = evict_fraction
        self.evicted = 0
        self.near_dups = 0

        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(embed_model)
            except Exception as e:
                print(f"  embeddings unavailable ({e}), using word overlap")

    @property
    def pending(self):
        return len(self.sentences) - self._indexed

    def fingerprint(self, tokens):
        """Identity of a sentence's CONTENT, ignoring order and punctuation.

        "The gate is opened when the beacon glows" and "when the beacon
        glows, the gate is opened" are the same fact stored twice. Exact
        string matching misses that, and on a real stream it is the most
        common kind of growth. A frozenset of the content tokens catches it
        for free.

        The cost is real and worth naming: two sentences with the same words
        in a different order are NOT always the same claim ("A caused B"
        versus "B caused A"). This treats them as one. Cheap and slightly
        lossy, and the loss is visible in near_dups.
        """
        return frozenset(tokens)

    def add(self, sentence):
        if sentence in self.seen:
            return False
        t = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        fp = self.fingerprint(t)
        if fp in self.prints:
            self.near_dups += 1
            return False

        self.seen.add(sentence)
        self.prints.add(fp)
        self.sentences.append(sentence)
        self.toks.append(t)
        self.added.append(self._clock)
        self.hits.append(0)
        self._clock += 1
        for tok in t:
            self.df[tok] += 1

        if len(self.sentences) > self.max_sentences:
            self.evict()
        return True

    def evict(self):
        """Drop the least useful tenth, in one batch.

        Usefulness is retrieval count. Being retrieved is the only evidence
        the store has that an item was ever wanted; age alone would discard
        the oldest facts, which are often the most established ones.
        Never-retrieved items go first, oldest among those first.

        Batched because it rebuilds the vector matrix, which is O(n). Doing
        that per insert would make every insert past the cap expensive; a
        tenth at a time amortises it to roughly one rebuild per 10,000
        inserts at the default cap.
        """
        n = len(self.sentences)
        drop = max(1, int(n * self.evict_fraction))
        order = sorted(range(n), key=lambda i: (self.hits[i], self.added[i]))
        doomed = set(order[:drop])
        keep = [i for i in range(n) if i not in doomed]

        self.sentences = [self.sentences[i] for i in keep]
        self.toks = [self.toks[i] for i in keep]
        self.added = [self.added[i] for i in keep]
        self.hits = [self.hits[i] for i in keep]

        # Rebuilt rather than decremented: eviction is rare and a drifting
        # df would silently corrupt the word-overlap fallback.
        self.seen = set(self.sentences)
        self.prints = {self.fingerprint(t) for t in self.toks}
        self.df = Counter()
        for t in self.toks:
            for tok in t:
                self.df[tok] += 1

        # Keep the vectors for surviving sentences, if they were indexed.
        if self._chunks:
            m = self.matrix()
            if m is not None:
                import torch
                rows = [i for i in keep if i < self._indexed]
                if rows:
                    idx = torch.tensor(rows)
                    self._chunks = [m.index_select(0, idx)]
                else:
                    self._chunks = []
                self._matrix = None
                self._indexed = len(rows)

        self.evicted += drop
        return drop

    def reindex(self):
        """Encode anything added since the last index.

        ONLY the new sentences. This used to re-encode the whole store on
        every call, which is 112 seconds at 200,000 items — paid again on
        every question, because search() reindexes lazily.

        Encoded vectors are moved to CPU immediately and kept as a list of
        chunks. Two reasons, both learned the hard way on 7 Sep:

          Concatenating two MPS tensors while a Hugging Face streaming
          thread is alive segfaults. Encoding on the GPU is worth keeping;
          storing there is not.

          Concatenating on every reindex copies the whole matrix each time,
          so a long read pays O(n^2) in memory traffic. Chunks are joined
          once, lazily, only when a search needs the matrix.
        """
        if self.model is None or self.pending == 0:
            return
        new = self.sentences[self._indexed:]
        vecs = self.model.encode(
            new, batch_size=256, convert_to_tensor=True,
            normalize_embeddings=True, show_progress_bar=False)
        self._chunks.append(vecs.detach().to("cpu"))
        self._matrix = None
        self._indexed = len(self.sentences)

    def matrix(self):
        """The full vector matrix, built at most once per batch of adds."""
        if self._matrix is not None:
            return self._matrix
        if not self._chunks:
            return None
        import torch
        if len(self._chunks) > 1:
            self._chunks = [torch.cat(self._chunks, dim=0)]
        self._matrix = self._chunks[0]
        return self._matrix

    def search(self, query, k=4):
        if not self.sentences:
            return []
        if self.model is not None:
            import torch
            if self.pending:
                self.reindex()
            q = self.model.encode([query], convert_to_tensor=True,
                                  normalize_embeddings=True)
            m = self.matrix()
            if m is None:
                return []
            sims = torch.mm(q.detach().to("cpu"), m.T).squeeze(0)
            idx = torch.topk(sims, min(k, m.shape[0])).indices
            out = []
            for i in idx:
                i = int(i)
                # Retrieval IS the usefulness signal eviction reads. If this
                # is not recorded, every item looks equally unwanted and
                # eviction degrades to dropping the oldest.
                if i < len(self.hits):
                    self.hits[i] += 1
                out.append(self.sentences[i])
            return out

        # fallback: word overlap weighted by rarity
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        n = max(1, len(self.sentences))
        scored = []
        for s, t in zip(self.sentences, self.toks):
            shared = q & t
            if not shared:
                continue
            score = sum(math.log(1 + n / (1 + self.df[tok]))
                        for tok in shared)
            scored.append((score / math.sqrt(len(t)), s))
        scored.sort(reverse=True)
        top = scored[:k]
        index = {sent: i for i, sent in enumerate(self.sentences)}
        for _, sent in top:
            i = index.get(sent)
            if i is not None and i < len(self.hits):
                self.hits[i] += 1
        return [s for _, s in top]

    def state(self):
        """Text plus the metadata eviction reads.

        Without hits and added, a restart resets every usefulness score to
        zero and the first eviction after it discards by age alone.
        """
        return dict(sentences=self.sentences, hits=self.hits,
                    added=self.added, clock=self._clock,
                    evicted=self.evicted, near_dups=self.near_dups)

    def load(self, state):
        sents = state.get("sentences", [])
        hits = state.get("hits") or []
        added = state.get("added") or []
        for i, sent in enumerate(sents):
            if self.add(sent):
                if i < len(hits):
                    self.hits[-1] = hits[i]
                if i < len(added):
                    self.added[-1] = added[i]
        self._clock = max(state.get("clock", 0), self._clock)
        self.evicted = state.get("evicted", 0)
        self.near_dups = state.get("near_dups", 0)

    def save_index(self, path):
        """Persist the encoded vectors alongside the text.

        Without this a restart re-encodes the whole store before it can
        answer anything, which is the startup cost that makes a long-lived
        system impractical.
        """
        if self.model is None or not self._chunks:
            return False
        import torch
        m = self.matrix()
        if m is None:
            return False
        torch.save(dict(count=self._indexed, vectors=m), path)
        return True

    def load_index(self, path):
        """Restore vectors if they match the text that was loaded.

        A mismatch is not an error: it means the store was written by a
        different model or is out of step, so we drop the cache and let
        reindex() rebuild. Silently trusting a stale index would corrupt
        every retrieval.
        """
        if self.model is None or not os.path.exists(path):
            return False
        import torch
        try:
            blob = torch.load(path, weights_only=False)
            vecs = blob["vectors"]
            count = int(blob["count"])
        except Exception as e:
            print(f"  index cache unreadable ({e}), rebuilding")
            return False
        if count > len(self.sentences) or vecs.shape[0] != count:
            print("  index cache does not match store, rebuilding")
            return False
        self._chunks = [vecs.detach().to("cpu")]
        self._matrix = None
        self._indexed = count
        return True


def looks_factual(sentence, min_propers=2):
    """Route by content: numbers or proper nouns mean a fact.

    Deliberately crude. A learned router is an obvious improvement and an
    untested one, so this stays simple and legible until there is evidence
    a better one helps.

    min_propers was effectively 1 until 7 Sep, which on ordinary prose
    matches nearly every sentence and starves the weights. 2 is a guess,
    not a measurement. Read status()["routing"] after a real stream and
    set it from the split you actually want.
    """
    if re.search(r"\d", sentence):
        return True
    words = [w.strip(",.;:()\"'") for w in sentence.split()[1:]]
    propers = sum(1 for w in words if w and w[:1].isupper())
    return propers >= min_propers


# Phrases that mean the model declined. Kept refusal-SHAPED rather than
# merely negative: "unable to" or "not included" on their own appear inside
# perfectly good answers, and a detector that fires on those would inflate
# the over-refusal figure it is used to compute.
#
# Widened 7 Sep after a live run scored "I cannot provide an answer to this
# question as it was not included in the given notes" as a FABRICATION. It
# was a textbook refusal. The system was behaving correctly and the metric
# was wrong, which is the same failure this file has now had three times.
REFUSALS = ["don't know", "do not know", "not know", "no information",
            "cannot find", "not mentioned", "unclear", "not specified",
            "not provided", "notes do not", "no mention", "not contain",
            "cannot provide an answer", "can't provide an answer",
            "cannot answer", "can't answer", "cannot be answered",
            "not included in the", "no answer to this",
            "do not have enough", "don't have enough",
            "notes given do not", "based on the notes provided, i"]


class NeuronSystem:
    """The whole thing, as one object.

    backend is anything with score/update/generate, so the same system runs
    on a 7B transformer or on the tiny fake backend used for testing.
    """

    def __init__(self, backend=None, path="neuron_state",
                 use_embeddings=True, retention=True, guard=True,
                 canary=None, min_propers=2, verbose=False,
                 max_sentences=100_000):
        self.backend = backend
        self.path = path
        self.store = Store(use_embeddings=use_embeddings,
                           max_sentences=max_sentences)
        self.layer = None
        self.retention = retention
        self.min_propers = min_propers
        self.canary = list(canary) if canary is not None else list(
            DEFAULT_CANARY)
        self.stats = Counter()
        self.started = time.time()

        if backend is not None and retention:
            from stability import StabilityLayer
            self.layer = StabilityLayer(
                backend, canary=(self.canary if guard else None), seed=0,
                verbose=verbose,
                window=200, warmup=30, top_fraction=0.50,
                coherence_veto=1.3, loss_floor=0.35, steps_per_update=1,
                # 24, not 6, since 7 Sep. Measured on the language drift
                # benchmark over two seeds: a quarter of the replay work
                # matched the old setting exactly on final loss and beat it
                # on rise from best. See stability.py's REHEARSAL BUDGET
                # note. Not re-checked on the weather or grid-world configs.
                rehearse_per_item=24, rehearse_count=2, rehearse_steps=1,
                anchor_size=60, buffer_size=300, sequence_len=8,
                guard=guard, guard_per_item=500, canary_tolerance=0.40)

    # ---------- the stream ----------

    def observe(self, text):
        """One moment of input. Routed, then handled.

        Returns what actually happened, and the counter agrees with it.
        Until 7 Sep this incremented "learned" for every item that reached
        the weights regardless of whether the layer learned from it, so a
        gate that rejected everything looked exactly like one that worked.
        """
        text = text.strip()
        if not text or len(text) < 20:
            self.stats["skipped"] += 1
            return "skipped"

        if looks_factual(text, self.min_propers):
            self.stats["routed_store"] += 1
            if self.store.add(text):
                self.stats["stored"] += 1
                return "stored"
            self.stats["duplicate"] += 1
            return "duplicate"

        self.stats["routed_weights"] += 1

        if self.backend is None:
            self.stats["no_backend"] += 1
            return "no_backend"

        if self.layer is None:
            self.backend.update(text, 1)
            self.stats["learned"] += 1
            return "learned"

        d = self.layer.observe(text)
        if d.get("learned"):
            self.stats["learned"] += 1
            return "learned"
        reason = d.get("reason", "rejected")
        self.stats[f"rejected_{reason}"] += 1
        return reason

    def read(self, lines, progress_every=0):
        """A whole stream. Returns what happened to it."""
        for i, line in enumerate(lines):
            self.observe(line)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  {i + 1} lines, {dict(self.stats)}", flush=True)
        return dict(self.stats)

    # ---------- questions ----------

    def ask(self, question, k=4):
        """Answer from the store, or say so if it cannot.

        The strict instruction is not decoration. Without it the model
        answers unanswerable questions from irrelevant context; with it,
        refusal held at every scale tested including 40,000 sentences.
        """
        if self.backend is None:
            return "(no backend attached)"
        notes = self.store.search(question, k)
        if notes:
            joined = "\n".join(f"- {n}" for n in notes)
            prompt = (f"Notes:\n{joined}\n\nAnswer using ONLY these notes. "
                      f"If they do not contain the answer, say you do not "
                      f"know.\n\nQ: {question}\nA:")
        else:
            prompt = f"Q: {question}\nA:"
        return self.backend.generate(prompt, max_new_tokens=80).strip()

    def refused(self, answer):
        low = answer.lower()
        return any(r in low for r in REFUSALS)

    def health(self):
        """Current canary damage, or None if nothing is watching."""
        if self.layer is None or not self.layer.canary:
            return None
        return self.layer.health()

    # ---------- persistence ----------

    def save(self):
        """Everything needed to resume. Until now, stopping the process
        destroyed the system entirely."""
        os.makedirs(self.path, exist_ok=True)

        with open(os.path.join(self.path, "store.json"), "w") as f:
            json.dump(self.store.state(), f)

        self.store.save_index(os.path.join(self.path, "index.pt"))

        meta = dict(stats=dict(self.stats),
                    uptime=time.time() - self.started,
                    retention=self.retention,
                    min_propers=self.min_propers,
                    saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        if self.layer is not None:
            try:
                meta["layer"] = self.layer.summary()
            except Exception:
                pass
        with open(os.path.join(self.path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if self.backend is not None and hasattr(self.backend, "snapshot"):
            try:
                import torch
                torch.save(self.backend.snapshot(),
                           os.path.join(self.path, "weights.pt"))
            except Exception as e:
                print(f"  could not save weights ({e})")
        return self.path

    def load(self):
        """Resume from disk. Missing pieces are skipped rather than fatal,
        so a partial save still restores what it can."""
        if not os.path.isdir(self.path):
            return False

        sp = os.path.join(self.path, "store.json")
        if os.path.exists(sp):
            with open(sp) as f:
                self.store.load(json.load(f))
            # Use the cached index if it matches; only encode what it misses.
            self.store.load_index(os.path.join(self.path, "index.pt"))
            self.store.reindex()

        mp = os.path.join(self.path, "meta.json")
        if os.path.exists(mp):
            with open(mp) as f:
                self.stats = Counter(json.load(f).get("stats", {}))

        wp = os.path.join(self.path, "weights.pt")
        if (os.path.exists(wp) and self.backend is not None
                and hasattr(self.backend, "restore")):
            import torch
            self.backend.restore(torch.load(wp, weights_only=False))
            # The layer took its baseline and checkpoint from the untouched
            # model in __init__. After restoring learned weights both are
            # stale: the baseline would flag ordinary drift as damage, and a
            # rollback would throw away everything just loaded.
            if self.layer is not None:
                self.layer.canary_baseline = self.layer._canary_score()
                self.layer._canary_now = self.layer.canary_baseline
                self.layer._checkpoint = self.backend.snapshot()
        return True

    def status(self):
        routed = (self.stats["routed_store"] + self.stats["routed_weights"])
        st = dict(stored=len(self.store.sentences),
                  capacity=self.store.max_sentences,
                  evicted=self.store.evicted,
                  near_dups_rejected=self.store.near_dups,
                  retrieved_at_least_once=sum(
                      1 for h in self.store.hits if h),
                  embeddings=self.store.model is not None,
                  indexed=self.store._indexed,
                  retention=self.layer is not None,
                  guarded=bool(self.layer and self.layer.canary),
                  health=self.health(),
                  stats=dict(self.stats),
                  uptime_minutes=(time.time() - self.started) / 60)
        st["routing"] = dict(
            to_store=self.stats["routed_store"],
            to_weights=self.stats["routed_weights"],
            store_share=(self.stats["routed_store"] / routed
                         if routed else None),
            min_propers=self.min_propers)
        if self.layer is not None:
            st["layer"] = self.layer.summary()
        return st