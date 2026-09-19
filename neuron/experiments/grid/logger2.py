"""Log the causes, not just the effects.

Rung 5 failed five times. The last attempt used a logger that recorded the
frontmost application every ten seconds — 7,886 samples over three days. The
model could not beat "you will still be in the app you are in", and the
diagnosis was not the method:

    the logger records WHAT app but never WHY. A message arriving, a build
    finishing, a phone call, remembering something — all invisible. The
    model was asked to predict effects from a stream that does not contain
    the causes.

This logs the triggers.

WHAT IS RECORDED, and nothing else:

  app            the frontmost application's NAME. No window titles, which
                 would leak document names, message previews and page
                 titles.
  dwell          how many seconds the current app has been in focus. People
                 leave an app after a while; this is the strongest
                 non-content predictor there is.
  since_switch   seconds since the last change of any kind.
  idle           seconds since the last keyboard or mouse event, from the
                 system's own idle timer. Distinguishes reading from having
                 walked away.
  notif          whether the notification database grew since the last
                 sample. A COUNT of unread items, never their contents —
                 this is the single most likely cause of an interruption
                 and the old logger was blind to it.
  battery        charge percent and whether plugged in. A proxy for
                 location and for session length.
  audio          whether anything is playing. Music, a call, a video.
  net            whether the machine is on wifi or not.
  hour, weekday  from the timestamp.

WHAT IS NOT RECORDED: window titles, URLs, keystrokes, clipboard, file
names, screen contents, notification text, contacts. The log is a plain text
file you can read, edit or delete at any time.

WHY THIS MIGHT STILL FAIL: a month of one person is still one person, and
app switching may be close to unpredictable from anything observable. Five
attempts have failed. This is the only version not already ruled out, and it
is worth an hour and a month of patience rather than another afternoon of
cleverness.

    python logger2.py

Leave it running. Check back in a month with rung5f.py.
"""

import os
import subprocess
import time
from datetime import datetime


LOG = os.path.expanduser("~/neuron/neuron/results/activity2.tsv")
INTERVAL = 15          # seconds between samples


FIELDS = ["timestamp", "app", "dwell", "since_switch", "idle",
          "notif_count", "battery", "plugged", "audio", "wifi"]


def sh(cmd, timeout=4):
    """Run a shell command, returning empty string on any failure. A logger
    that crashes after two weeks is worse than one that records a blank."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def frontmost():
    """The application NAME only.

    The window title is available through the same interface and is
    deliberately not requested, because it would carry document names,
    message previews and page titles.
    """
    out = sh('osascript -e \'tell application "System Events" to get name '
             'of first application process whose frontmost is true\'')
    return out or "unknown"


def idle_seconds():
    """Seconds since the last keyboard or mouse event, from the system's
    own HID idle timer. Distinguishes reading from having walked away."""
    out = sh("ioreg -c IOHIDSystem | awk '/HIDIdleTime/ "
             "{print int($NF/1000000000); exit}'")
    try:
        return int(out)
    except ValueError:
        return -1


def notification_count():
    """How many notifications are pending. A COUNT, never contents.

    This is the most likely cause of an interruption and the previous
    logger was blind to it. Reads the badge counts the Dock already shows,
    which requires no special permission and exposes nothing.
    """
    out = sh("""osascript -e 'tell application "System Events" to tell """
             """process "Dock" to get value of attribute "AXStatusLabel" """
             """of every UI element of list 1' 2>/dev/null""")
    if not out:
        return 0
    total = 0
    for part in out.split(","):
        part = part.strip()
        if part.isdigit():
            total += int(part)
    return total


def battery():
    """Charge percent and whether plugged in. A proxy for location and for
    how long a session is likely to last."""
    out = sh("pmset -g batt")
    pct, plugged = -1, 0
    if "%" in out:
        try:
            pct = int(out.split("%")[0].split()[-1])
        except (ValueError, IndexError):
            pass
    if "AC Power" in out:
        plugged = 1
    return pct, plugged


def audio_playing():
    """Whether anything is producing sound. Music, a call, a video — all
    things that change how long someone stays put."""
    out = sh("pmset -g | grep -c 'coreaudiod'")
    return 1 if out.strip() not in ("", "0") else 0


def on_wifi():
    out = sh("ifconfig en0 2>/dev/null | grep -c 'status: active'")
    return 1 if out.strip() not in ("", "0") else 0


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    first = not os.path.exists(LOG)

    with open(LOG, "a") as f:
        if first:
            f.write("# " + "\t".join(FIELDS) + "\n")
            f.flush()

        print(f"logging to {LOG}")
        print(f"one sample every {INTERVAL}s")
        print("recorded: app name, dwell, idle, notification COUNT, "
              "battery, audio, wifi")
        print("NOT recorded: window titles, URLs, keystrokes, clipboard, "
              "file names,\n              screen contents, notification "
              "text\n")
        print("stop with Ctrl+C. Leave it running for a month.\n")

        last_app = None
        app_since = time.time()
        last_switch = time.time()
        n = 0

        while True:
            now = time.time()
            app = frontmost()

            if app != last_app:
                if last_app is not None:
                    last_switch = now
                app_since = now
                last_app = app

            pct, plugged = battery()
            row = [
                datetime.now().isoformat(timespec="seconds"),
                app,
                str(int(now - app_since)),
                str(int(now - last_switch)),
                str(idle_seconds()),
                str(notification_count()),
                str(pct),
                str(plugged),
                str(audio_playing()),
                str(on_wifi()),
            ]
            f.write("\t".join(row) + "\n")
            f.flush()

            n += 1
            if n % 40 == 0:
                print(f"  {n} samples, {datetime.now():%H:%M}, "
                      f"{app}, idle {row[4]}s, notif {row[5]}",
                      flush=True)

            time.sleep(max(0, INTERVAL - (time.time() - now)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")