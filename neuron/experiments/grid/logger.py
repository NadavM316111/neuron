"""Rung 5, the proper version: log what a person is actually doing.

Three attempts at using data the machine already had all failed for the same
reason. Shell history was one day's work. File modification times turned out
to be a record of package managers, not of a person. Browsing history was
the closest but came up six points short of a bigram lookup, on 6,591
moments across 61 classes.

So collect the right data deliberately.

WHAT IS RECORDED: the name of the frontmost application, and the time.
Nothing else. No window titles, no URLs, no keystrokes, no screen contents,
no document names. A window title would leak everything — the document being
edited, the message being read, the site being visited — and is not worth it.

SAMPLING: once a minute, whether or not the app changed. That matters. If
only switches were logged, a few days would give maybe 1,500 events, which
is where the shell history failed. Minute samples give about 1,440 a day, so
three and a half days is roughly 5,000 moments, and it keeps accruing after
that.

THE TASK this feeds: given the recent minutes and the time of day, predict
which app will be in focus five minutes from now. Same shape as the weather
task, which is deliberate — everything from rung 4 transfers.

The log is a plain text file you can read, edit or delete at any time.
"""

import os
import subprocess
import sys
import time
from datetime import datetime


LOG = os.path.expanduser("~/neuron/activity_log.tsv")
INTERVAL = 10         # seconds between samples


def frontmost():
    """The name of the frontmost application, via AppleScript.

    Deliberately asks for the application NAME only. The window title is
    available through the same interface and is not requested, because it
    would carry document names, message previews and page titles.
    """
    script = ('tell application "System Events" to get name of first '
              'application process whose frontmost is true')
    try:
        r = subprocess.run(["osascript", "-e", script],
                           capture_output=True, text=True, timeout=5)
        name = r.stdout.strip()
        return name or "unknown"
    except Exception:
        return "unknown"


def main():
    first = not os.path.exists(LOG)
    with open(LOG, "a") as f:
        if first:
            f.write("# timestamp\tapp\n")
            f.flush()

        print(f"logging to {LOG}")
        print(f"one sample every {INTERVAL}s, application name only")
        print("stop with Ctrl+C, or close this terminal window\n")

        n = 0
        while True:
            app = frontmost()
            now = datetime.now()
            f.write(f"{now.isoformat(timespec='seconds')}\t{app}\n")
            f.flush()
            n += 1
            if n % 30 == 0:
                print(f"  {n} samples, {now:%H:%M}, currently {app}",
                      flush=True)
            time.sleep(INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")