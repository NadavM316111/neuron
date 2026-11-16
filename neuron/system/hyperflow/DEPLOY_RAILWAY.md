# NEURON HyperFlow - 72h shakedown on Railway

Goal: one NEURON lives continuously for 72 hours, survives real restarts, you can
close your laptop and watch it from your phone.

## One-time setup (~5 min)

1. Push this folder to GitHub (the whole repo).

2. railway.app -> New Project -> Deploy from GitHub repo -> pick `neuron`.

3. Service -> Settings:
   - Root Directory:  neuron/neuron/system/hyperflow
   - Start Command:   python3 hyperflow_72h.py

4. Service -> Variables, add:
   - HYPERFLOW_DIR = /data
   - RUN_HOURS     = 72
   - WAKE_INTERVAL = 60

5. Service -> Variables -> New Volume (or the Volume tab):
   - Mount path: /data
   (This is the part that matters. Without the volume, every restart wipes the
   disk and NEURON is reborn as a newborn, which fails 9J for the wrong reason.)

6. Settings -> Networking -> Generate Domain.
   Open that URL on your phone to watch the live status page.

7. Deploy. Done. Close your laptop.

## What you'll see

The status page shows: hours elapsed / 72, cycles, restarts, reserve, real value,
accept/wait/decline counts, forbidden-blocked count, the 9 pass/fail checks live,
and the last 25 log lines. It refreshes every 20s.

At 24h and 48h it restarts itself on purpose (the page will blip, then show
"RESUMED after restart"). That is the point: proving the same NEURON comes back
from disk instead of a newborn.

## When it's done (after 72h)

The page flips to DONE and the checks freeze at their final state. Screenshot it.
If all 9 are PASS, 9J is done for real and you're clear for 9K (the 7-day pilot).

## Notes

- Restart policy: Railway's default (On Failure) brings it back after the planned
  exits. If it ever stops coming back, set Restart Policy to "Always".
- This uses Railway execution hours continuously for 3 days. Make sure your plan
  won't sleep the service (Hobby doesn't sleep; just watch the usage credit).
- To re-run from scratch, delete the /data volume (or clear it) and redeploy.
