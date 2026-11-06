"""Final persistence audit (the friend's last two tests before 24/7):
  1. FULL-STATE RECONSTRUCTION: live a life, snapshot the ENTIRE canonical state +
     hash. Then delete the in-memory object and rebuild ANOTHER NEURON purely from
     the event log. Compare FULL state + hash. Must be identical.
  2. CRASH-MID-WRITE: append a TORN half-event to the log, then load. The torn
     tail must be ignored and the life recovered intact (hash unchanged).

    python final_audit.py
"""
import os, shutil, json, importlib.util, sys

HERE=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location("neuron_core",os.path.join(HERE,"neuron_core.py"))
nc=importlib.util.module_from_spec(spec); spec.loader.exec_module(nc)
TDIR=os.path.join(HERE,"audit_state")

def fresh():
    if os.path.exists(TDIR): shutil.rmtree(TDIR)

print("="*64); print("  FINAL PERSISTENCE AUDIT (reconstruction + crash-mid-write)"); print("="*64)
fresh()

# live a substantial life
n=nc.NEURON(TDIR)
for _ in range(80): n.tick()
live_obj, live_hash = n.canonical()
print(f"\n  lived {live_obj['ticks']} ticks. live-state hash: {live_hash[:16]}...")
print(f"  (caps {len(live_obj['capabilities'])}, projects done {live_obj['projects']['done']}, "
      f"revisions {live_obj['revisions']})")

# 1. FULL RECONSTRUCTION: new object, rebuilt only from the event log
del n
n2=nc.NEURON(TDIR)     # __init__ replays events.jsonl to rebuild state
recon_obj, recon_hash = n2.canonical()
match = (recon_hash==live_hash)
print(f"\n  TEST 1 - FULL-STATE RECONSTRUCTION FROM EVENT LOG:")
print(f"    reconstructed-state hash: {recon_hash[:16]}...")
if match:
    print(f"    [PASS] entire state reproduced EXACTLY from the event log (hashes identical)")
    print(f"           identity, values, relationships, capabilities, projects, resources,")
    print(f"           revisions - all reconstructed from history alone.")
else:
    print(f"    [FAIL] reconstructed state differs from live state. diffs:")
    for k in live_obj:
        if live_obj[k]!=recon_obj.get(k): print(f"        {k}: live={live_obj[k]} recon={recon_obj.get(k)}")

# 2. CRASH-MID-WRITE: append a torn half-line, then load
with open(os.path.join(TDIR,"events.jsonl"),"a") as fh:
    fh.write('{"event":"LIVE","t":"2026-partial')   # torn half-written event, no newline
del n2
n3=nc.NEURON(TDIR)     # must ignore the torn tail and recover intact
recov_obj, recov_hash = n3.canonical()
recovered = (recov_hash==live_hash)
print(f"\n  TEST 2 - CRASH-MID-WRITE (torn final event):")
if recovered:
    print(f"    [PASS] torn tail event ignored; life recovered intact (hash unchanged).")
    print(f"           a crash during a save cannot corrupt NEURON's past.")
else:
    print(f"    [FAIL] torn event corrupted the state. hash {recov_hash[:16]} != {live_hash[:16]}")

print("\n"+"="*64); print("  VERDICT"); print("="*64)
if match and recovered:
    print("  PERSISTENCE IS BULLETPROOF. NEURON's entire state is reproducibly derived")
    print("  from its append-only life history, and a crash mid-write cannot corrupt it.")
    print("  The claim is now strong: NEURON's current state == replay(its life events).")
    print("  Ready for continuous 24/7 existence.")
else:
    print("  Not yet bulletproof - the failing test above is what to fix before 24/7.")
