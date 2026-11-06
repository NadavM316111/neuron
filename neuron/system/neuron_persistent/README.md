# NEURON - 24/7 bounded persistent life

One NEURON living continuously. neuron_core.py is the event-sourced being (state =
replay of an append-only event log). neuron_daemon.py runs it on a real-time
schedule and serves a read-only dashboard.

## Local
    python3 neuron_daemon.py
Open http://localhost:8080 . It keeps living whether or not the page is open.
Stop it, run it again tomorrow: it continues the SAME life (state in ./life/).

## Railway (24/7)
Point the service start command at this file:
    python neuron/system/neuron_persistent/neuron_daemon.py
IMPORTANT: attach a PERSISTENT VOLUME and set NEURON_DIR to a path on it (e.g.
/data/life) so its life survives restarts/redeploys. Set TICK_SECONDS to pace its
life (e.g. 60 = one life-tick per minute). Generate a domain, open it tomorrow,
it has continued running and accumulating history while you were away.

Bounded sandbox: no internet action, accounts, money, or reproduction. One NEURON,
one identity, one history.
