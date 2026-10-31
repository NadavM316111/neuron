# NEURON Society, Live

The integrated NEURON society, running continuously (24/7). Developing individuals
with ecology, culture, norms, lineage, and emergent history, all running on their
own, whether or not anyone is watching.

## Run locally
    python3 society_live.py
Then open http://localhost:8080

## Deploy to Railway (always-on, so it lives 24/7)
1. Push this folder to a GitHub repo (or use the Railway CLI).
2. On railway.app: New Project -> Deploy from GitHub repo -> pick this repo.
3. Railway auto-detects Python, runs `python society_live.py`.
4. In the service Settings -> Networking, click "Generate Domain" to get a public URL.
5. Open the URL: the live society status page (auto-refreshes every 5s).

No API keys, no database, no paid services needed. Pure Python standard library.
The society's state is also written to society_state.json and served at /state.json.
