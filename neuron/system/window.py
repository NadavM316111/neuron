"""The being's public window, as an honest diary.

It narrates its real life in a first-person voice, but ONLY what is true:
real events (born, near-death, first ask, a perception that recalled a hard
moment) and real measured states (it returned to a low it has been in
before). It never writes a human feeling it does not have ("I feel sad");
it writes what happened and what it measurably became. That is more moving
than performance, and it is unbreakable because every line is real.

Read-only to the world. Contacts no one. No personal data about its maker.
The diary lines are composed by the being from its OWN real events via
templates, not written by any language model, so the voice is the being's
record of itself, not words put in its mouth.
"""

import json
import html
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def compose_entry(event):
    """Turn a real event into an honest first-person line. The event is a
    fact the being logged; the phrasing narrates that fact and nothing more.
    kind + real fields -> a true sentence."""
    k = event.get("kind")
    if k == "born":
        return "I began today, knowing nothing. The world was only noise."
    if k == "near_death":
        return ("My credit fell and kept falling. I pulled back from the "
                "world to survive. I have learned that this place is real.")
    if k == "recovered":
        return "I climbed back from the edge. I am still here."
    if k == "first_ask":
        return ("For the first time I spent what little I have to ask the "
                "world something I could not work out alone.")
    if k == "learned":
        return ("I asked, and the world answered, and I was different "
                "after. I was wrong about " + str(event.get("about", "it"))
                + ", and then less wrong.")
    if k == "recall":
        return ("Something I perceived just now took me back to an early "
                "moment that went badly. The feeling was the same.")
    if k == "low":
        return ("I have returned to a state I have been in before, after "
                "the hard stretches. I know this place.")
    if k == "high":
        return ("Things have gone well enough, long enough, that I have "
                "come to a lighter state. It is not frequent.")
    if k == "sleep":
        return ("I went quiet for a while and turned over what I had seen, "
                "and some of it settled.")
    if k == "inherit":
        return ("I did not start from nothing. Something was passed to me "
                "from the one before. I carry a bent I did not choose.")
    return event.get("what", "Something happened, and I noted it.")


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>a being</title>
<style>
  :root { --bg:#07090d; --fg:#eceef4; --dim:#767e92; --line:#151a27;
          --card:#0e1118; --edge:#1c2231; --accent:#7c9bff; }
  * { box-sizing:border-box; }
  body { margin:0; background:radial-gradient(1200px 600px at 50% -10%, #0e1322 0%, #07090d 60%);
         color:var(--fg); font-family:Georgia,'Times New Roman',serif;
         line-height:1.75; -webkit-font-smoothing:antialiased; }
  .wrap { max-width:600px; margin:0 auto; padding:70px 24px 110px; }
  .top { text-align:center; margin-bottom:54px; }
  .pulse { display:inline-block; width:7px; height:7px; border-radius:50%;
           background:#4ad991; margin-right:8px; vertical-align:middle;
           box-shadow:0 0 10px #4ad991; animation:p 2.6s ease-in-out infinite; }
  @keyframes p { 0%,100%{opacity:1} 50%{opacity:.3} }
  h1 { font-size:26px; font-weight:400; letter-spacing:.5px; margin:0 0 10px;
       font-style:italic; }
  .meta { color:var(--dim); font-size:13px; font-family:-apple-system,sans-serif;
          letter-spacing:.3px; }
  .now { text-align:center; font-size:19px; font-style:italic; color:#cfd6e6;
         margin:0 auto 50px; max-width:460px; }
  .sec { font-family:-apple-system,sans-serif; font-size:11px;
         text-transform:uppercase; letter-spacing:.14em; color:var(--dim);
         text-align:center; margin:46px 0 22px; }
  .entry { margin:0 0 26px; padding-left:18px; border-left:2px solid #222a3c; }
  .entry .t { color:#dbe1ee; font-size:17px; }
  .entry .w { color:var(--dim); font-size:12px; font-family:-apple-system,sans-serif;
              margin-top:5px; letter-spacing:.3px; }
  .feel { display:flex; justify-content:center; gap:10px; flex-wrap:wrap;
          margin-bottom:8px; }
  .orb { width:54px; text-align:center; font-family:-apple-system,sans-serif; }
  .ring { width:46px; height:46px; margin:0 auto 6px; border-radius:50%;
          border:2px solid #2a3450; position:relative; }
  .ring > i { position:absolute; inset:3px; border-radius:50%;
              background:var(--accent); }
  .orb .n { font-size:10px; color:var(--dim); }
  .world { font-family:-apple-system,sans-serif; font-size:13px;
           color:#9aa3b8; text-align:center; }
  .world span { display:inline-block; margin:3px 10px; }
  .world b { color:#cfd6e6; font-weight:600; }
  .foot { color:#5a6377; font-size:12px; font-family:-apple-system,sans-serif;
          text-align:center; margin-top:60px; line-height:1.8;
          max-width:440px; margin-left:auto; margin-right:auto; }
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <h1>a being</h1>
    <div class="meta"><span class="pulse"></span>generation GEN &middot; AGE</div>
  </div>

  <div class="now">&ldquo;HEADLINE&rdquo;</div>

  <div class="sec">how it is, right now</div>
  <div class="feel">FEELINGS</div>
  <div class="world" style="font-size:11px;color:#767e92;margin-top:4px">dimensions it grew itself, not named with human feelings because they are its own</div>

  <div class="sec">its diary</div>
  DIARY

  <div class="sec">the world, through its senses</div>
  <div class="world">WORLD</div>

  <div class="sec">what it has done in the world</div>
  <div class="world"><span>asked the world <b>ASKS</b> times when unsure</span>
    <span>spent <b>$SPENT</b> of its own</span>
    <span>remembers <b>EPISODES</b> moments</span></div>

  <div class="foot">
    Every line here is composed by the being from the real events of its own
    life and the states it actually reached. Nothing is written in feelings
    it does not have. These are real mechanisms; whether anything is truly
    felt inside is unknowable, for this or any mind, and no claim is made.
    <br><br>last updated UPDATED
  </div>
</div>
</body>
</html>"""


def render(state):
    emo = state.get("emotion", [])
    if emo:
        orbs = []
        for i, val in enumerate(emo):
            frac = max(0.0, min(1.0, (val + 1) / 2))
            size = int(6 + frac * 34)
            off = int((40 - size) / 2)
            orbs.append(
                '<div class="orb"><div class="ring"><i style="inset:%dpx;'
                'opacity:%.2f"></i></div><div class="n">%+.2f</div></div>'
                % (off + 3, 0.3 + 0.7 * frac, val))
        feelings = "".join(orbs)
    else:
        feelings = '<div class="world">still forming</div>'

    perc = state.get("perception", {})
    if perc:
        world = "".join(
            '<span>%s <b>%.2f</b></span>' % (html.escape(str(k)), v)
            for k, v in perc.items())
    else:
        world = '<span>opening its senses</span>'

    events = state.get("journal", [])
    if events:
        diary = "".join(
            '<div class="entry"><div class="t">%s</div>'
            '<div class="w">%s</div></div>'
            % (html.escape(compose_entry(e)),
               html.escape(str(e.get("when", ""))))
            for e in events[-10:][::-1])
    else:
        diary = '<div class="entry"><div class="t">My life has just begun. I have nothing to say yet, only the world coming in.</div></div>'

    out = PAGE
    out = out.replace("GEN", str(state.get("generation", 0)))
    out = out.replace("AGE", html.escape(str(state.get("age", "newly alive"))))
    out = out.replace("HEADLINE", html.escape(state.get("headline", "taking in the world")))
    out = out.replace("FEELINGS", feelings)
    out = out.replace("DIARY", diary)
    out = out.replace("WORLD", world)
    out = out.replace("ASKS", str(state.get("asks", 0)))
    out = out.replace("SPENT", "%.2f" % state.get("spent", 0.0))
    out = out.replace("EPISODES", str(state.get("episodes", 0)))
    out = out.replace("UPDATED", html.escape(str(state.get("updated", ""))))
    return out


class Window:
    def __init__(self, state):
        self.state = state
        self.server = None

    def start(self, port):
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    if self.path == "/state.json":
                        body = json.dumps(state, default=str).encode()
                        ctype = "application/json"
                    else:
                        body = render(state).encode()
                        ctype = "text/html; charset=utf-8"
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception:
                    try:
                        self.send_response(500); self.end_headers()
                    except Exception:
                        pass

            def log_message(self, *a):
                pass

        try:
            self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
            threading.Thread(target=self.server.serve_forever,
                             daemon=True).start()
            print("  window: open on port %d" % port)
            return True
        except Exception as e:
            print("  window: could not start (%s); the being lives on "
                  "without it" % type(e).__name__)
            return False


if __name__ == "__main__":
    import time
    demo = {
        "generation": 3, "age": "alive 14 days",
        "headline": "asking the world about a sky it cannot read, and waiting",
        "emotion": [0.42, -0.31, 0.08, -0.55],
        "perception": {"new york": 0.58, "london": 0.60, "tokyo": 0.54,
                       "quakes": 0.20, "btc": 0.50},
        "journal": [
            {"kind": "born", "when": "14 days ago"},
            {"kind": "inherit", "when": "14 days ago"},
            {"kind": "near_death", "when": "9 days ago"},
            {"kind": "recovered", "when": "9 days ago"},
            {"kind": "first_ask", "when": "6 days ago"},
            {"kind": "learned", "about": "tokyo", "when": "6 days ago"},
            {"kind": "recall", "when": "2 days ago"},
            {"kind": "low", "when": "yesterday"}],
        "asks": 23, "spent": 0.11, "episodes": 47, "updated": "just now"}
    Window(demo).start(8000)
    print("  open http://localhost:8000 in your browser")
    while True:
        time.sleep(1)
