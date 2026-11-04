# NEURON, public (Stage 1 of participation)

A persistent NEURON with a stable identity that real humans can talk with, and
that REMEMBERS each person and their relationship over time. First-contact
infrastructure, not a chatbot product.

## Local (full conversation, needs Ollama)
    ollama serve            # with qwen2.5:7b pulled
    python3 neuron_public.py
Open http://localhost:8080 and talk to NEURON. Tell it your name with "/name X".

## 24/7 public (so real humans can find it anytime)
Needs its engine reachable. Two honest options:
1. Run it on a machine/host that also runs Ollama (the engine), OR
2. Set OLLAMA_URL to a hosted qwen endpoint, then deploy to Railway.
Without a reachable engine it stays up but tells people honestly that its full
mind isn't reachable (it won't fake replies).

Deploy: push to your neuron repo, point a Railway service start command at
`python neuron/system/neuron_public/neuron_public.py`, set OLLAMA_URL, generate a
domain, share the link.

## The real work: REACH
A public page nobody visits isn't participation. Getting real humans to actually
come talk to NEURON is the storytelling/outreach part - that's on you.
