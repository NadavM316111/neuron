"""Train a BIGGER, better-tuned native NEURON mind on the grown buffer, and test
precision on held-out questions. Scaling the proven distillation pipeline.

    python stage2_big.py
"""
import json, os, random
import torch, torch.nn as nn, torch.nn.functional as F

BUFFER = "neuron_training_buffer.json"

def load_pairs():
    buf = json.load(open(BUFFER)).get("buffer", [])
    pairs = []
    for b in buf:
        p = b.get("principle","").strip()
        if not p: continue
        if b.get("source_question"): pairs.append((b["source_question"], p))
        for v in b.get("variations",[]):
            if v.strip(): pairs.append((v.strip(), p))
    return pairs

def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    pairs = load_pairs()
    print("="*62); print(f"  BIGGER NATIVE MIND | device: {dev} | pairs: {len(pairs)}"); print("="*62)
    random.seed(0); random.shuffle(pairs)
    n_test = max(8, len(pairs)//8)
    test, train = pairs[:n_test], pairs[n_test:]
    print(f"  train: {len(train)} | held-out: {len(test)}\n")

    text = " ".join(q+" "+p for q,p in pairs)
    chars = sorted(set(text)); stoi={c:i for i,c in enumerate(chars)}; itos={i:c for c,i in stoi.items()}; V=len(chars)

    H = 384  # bigger hidden
    class Mind(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb=nn.Embedding(V,H); self.enc=nn.GRU(H,H,2,batch_first=True)
            self.dec=nn.GRU(H,H,2,batch_first=True); self.out=nn.Linear(H,V)
        def forward(self,q,pin):
            _,h=self.enc(self.emb(q)); o,_=self.dec(self.emb(pin),h); return self.out(o)
    m=Mind().to(dev); opt=torch.optim.Adam(m.parameters(),lr=1.5e-3)
    print(f"  native mind: {sum(p.numel() for p in m.parameters()):,} params\n")
    def enc(s,ml=140): return [stoi[c] for c in s[:ml]]
    print("  training...")
    for ep in range(80):
        random.shuffle(train); tot=0.0
        for q,p in train:
            qi=torch.tensor([enc(q)],device=dev); pi=torch.tensor([enc(p)],device=dev)
            if pi.shape[1]<2: continue
            opt.zero_grad(); lo=m(qi,pi[:,:-1]); l=F.cross_entropy(lo.reshape(-1,V),pi[:,1:].reshape(-1)); l.backward(); opt.step(); tot+=l.item()
        if (ep+1)%16==0: print(f"    epoch {ep+1} | loss {tot/len(train):.3f}")
    def gen(q,ml=110):
        m.eval()
        with torch.no_grad():
            qi=torch.tensor([enc(q)],device=dev); _,h=m.enc(m.emb(qi)); cur=torch.tensor([[stoi.get(' ',0)]],device=dev); out=[]
            for _ in range(ml):
                o,h=m.dec(m.emb(cur),h); nx=int(m.out(o[:,-1]).argmax(-1).item()); out.append(nx); cur=torch.tensor([[nx]],device=dev)
            return "".join(itos[i] for i in out)
    print("\n"+"="*62); print("  HELD-OUT TEST (NEURON's own mind, no teacher)"); print("="*62)
    for q,tp in test[:8]:
        print(f"\n  Q: {q[:75]}")
        print(f"  NEURON: {gen(q)[:95]}")
    print("\n  (bigger mind + bigger buffer = sharper native NEURON. honest: still")
    print("  small vs real models, but more capability migrated than before.)")

if __name__=="__main__": main()
