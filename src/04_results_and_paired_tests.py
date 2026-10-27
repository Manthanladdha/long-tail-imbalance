# %%
# ===================== CELL 6: results tables =====================
import os, glob, json
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
DATASET, RES = "isic2019", 224
OUT = f"/kaggle/working/{DATASET}"; CACHE = f"{OUT}/cache"; RUNS = f"{OUT}/runs_{RES}"
meta = json.load(open(f"{OUT}/meta.json")); classes, tail = meta["classes"], meta["tail"]; K = len(classes)
tail_i = [classes.index(c) for c in tail]; head_i = [i for i in range(K) if i not in tail_i]
L = lambda s: np.load(f"{CACHE}/{RES}_squash_all_{s}_y.npy"); Ytr, Yva, Yte = L("train"), L("val"), L("test"); prior = np.bincount(Ytr, minlength=K)/len(Ytr)
sm = lambda z: (lambda e: e/e.sum(1,keepdims=True))(np.exp(z-z.max(1,keepdims=True)))
def met(p, y):
    pr = p.argmax(1); r = recall_score(y, pr, labels=list(range(K)), average=None, zero_division=0)
    return {"bma": balanced_accuracy_score(y,pr), "gmean": float(np.prod(r)**(1/K)), "f1": f1_score(y,pr,average="macro"), "tail": r[tail_i].mean(), "head": r[head_i].mean(), "rec": r}
found = {}
for f in glob.glob(f"{RUNS}/*_s*_test.npy"):
    n = os.path.basename(f)[:-9]; m, s = n.rsplit("_s",1)
    if os.path.exists(f"{RUNS}/{n}_val.npy"): found.setdefault(m,[]).append(int(s))
print("runs found:", {m: sorted(v) for m,v in sorted(found.items())})
rng = np.random.RandomState(0); rows, pcl = [], {}
for m, seeds in sorted(found.items()):
    V = [np.load(f"{RUNS}/{m}_s{s}_val.npy") for s in seeds]; T = [np.load(f"{RUNS}/{m}_s{s}_test.npy") for s in seeds]; variants = [(m, 0.0)]
    if not m.startswith("bs"):
        taus = np.arange(0, 2.01, 0.25); tau = taus[int(np.argmax([balanced_accuracy_score(Yva, np.mean([sm(z-t*np.log(prior)) for z in V],0).argmax(1)) for t in taus]))]
        variants.append((f"{m} + LA(tau={tau})", tau))
    for label, tau in variants:
        P = [sm(z-tau*np.log(prior)) for z in T]; single = [met(p,Yte) for p in P]; ep = np.mean(P,0); en = met(ep,Yte); r = {"method": label, "seeds": len(seeds)}
        for k in ("bma","gmean","f1","tail","head"):
            v = [s[k] for s in single]; r[f"{k}_single"] = f"{np.mean(v):.3f}±{(np.std(v,ddof=1) if len(v)>1 else 0):.3f}"; r[f"{k}_ens"] = round(float(en[k]),4)
        pr = ep.argmax(1); b = [balanced_accuracy_score(Yte[i], pr[i]) for i in (rng.randint(0,len(Yte),len(Yte)) for _ in range(300))]
        r["bma_ens_95CI"] = f"[{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}]"; rows.append(r); pcl[label] = en["rec"]
S = pd.DataFrame(rows).set_index("method").sort_values("bma_ens", ascending=False); pd.set_option("display.width", 260)
print(S.to_string()); print(); print(pd.DataFrame(pcl, index=classes).T.loc[S.index].round(3).to_string())
S.to_csv(f"{OUT}/summary_{RES}.csv")

# %%
# ===================== CELL Z1: paired bootstrap, synthetic vs controls (CPU) =====================
import numpy as np, json
from sklearn.metrics import balanced_accuracy_score
OUT = "/kaggle/working/isic2019"; RUNS = f"{OUT}/runs_224"; C = f"{OUT}/cache"
K = len(json.load(open(f"{OUT}/meta.json"))["classes"]); Ytr, Yte = (np.load(f"{C}/224_squash_all_{s}_y.npy") for s in ("train", "test"))
prior = np.bincount(Ytr, minlength=K) / len(Ytr); sm = lambda z: (lambda e: e / e.sum(1, keepdims=True))(np.exp(z - z.max(1, keepdims=True)))
def ens(m, tau): return np.mean([sm(np.load(f"{RUNS}/{m}_s{s}_test.npy") - tau * np.log(prior)) for s in (0, 1, 2)], 0).argmax(1)
TAU = 1.25                                    # fixed for all rows so the comparison is fair
P = {m: ens(m, TAU) for m in ["cb", "syn_eq", "dup_eq", "vae_eq", "syn_dose", "syn_gate", "loop_final"] if all(__import__("os").path.exists(f"{RUNS}/{m}_s{s}_test.npy") for s in (0, 1, 2))}
rng = np.random.RandomState(0); idx = [rng.randint(0, len(Yte), len(Yte)) for _ in range(1000)]
print("BMA (tau=1.25):", {m: round(balanced_accuracy_score(Yte, p), 4) for m, p in P.items()})
for a, b in [("syn_eq", "dup_eq"), ("syn_eq", "vae_eq"), ("syn_eq", "cb"), ("syn_dose", "syn_eq"), ("syn_gate", "syn_eq"), ("loop_final", "cb"), ("loop_final", "syn_eq")]:
    if a in P and b in P:
        d = np.array([balanced_accuracy_score(Yte[i], P[a][i]) - balanced_accuracy_score(Yte[i], P[b][i]) for i in idx])
        print(f"{a} - {b}: {d.mean():+.4f}  95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
