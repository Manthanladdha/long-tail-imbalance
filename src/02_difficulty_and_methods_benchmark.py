# %%
# ===================== CELL D1: difficulty scores (out-of-fold) =====================
import os, json, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

DATASET, RES = "isic2019", 224
OUT = f"/kaggle/working/{DATASET}"; CACHE = f"{OUT}/cache"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
meta = json.load(open(f"{OUT}/meta.json")); classes, tail = meta["classes"], meta["tail"]; K = len(classes)
t0 = time.time()

Xs, rows = [], []
for s in ("train", "val"):
    Xs.append(np.load(f"{CACHE}/{RES}_squash_all_{s}_x.npy", mmap_mode="r")); rows.append(pd.read_csv(f"{CACHE}/{RES}_squash_all_{s}_rows.csv"))
rows = pd.concat(rows, ignore_index=True)
y = rows["class"].map({c: i for i, c in enumerate(classes)}).values; groups = rows["group"].astype(str).values
print("pool:", len(y), "images,", K, "classes")

FE = f"{OUT}/feats_dinov2s_{RES}.npy"
if os.path.exists(FE): Fe = np.load(FE)
else:
    from transformers import AutoModel
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval(); out = []
    with torch.no_grad():
        for X in Xs:
            for i in range(0, len(X), 128):
                x = torch.from_numpy(np.ascontiguousarray(X[i:i + 128])).to(DEV).permute(0, 3, 1, 2).float() / 255
                f = dino(pixel_values=(x - mean) / std).pooler_output.float()
                out.append(F.normalize(f, dim=1).cpu())
    Fe = torch.cat(out).numpy(); np.save(FE, Fe)
print("features", Fe.shape, f"| {time.time() - t0:.0f}s")

oof = np.zeros((len(y), K))
for k, (a, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Fe, y, groups)):
    clf = LogisticRegression(C=10, max_iter=500, class_weight="balanced").fit(Fe[a], y[a])
    p = np.zeros((len(b), K)); p[:, clf.classes_] = clf.predict_proba(Fe[b]); oof[b] = p
    print(f"fold {k + 1}/5 | {time.time() - t0:.0f}s")

pred = oof.argmax(1); n_c = np.bincount(y, minlength=K)
rec = np.array([(pred[y == c] == c).mean() for c in range(K)])
a_, b_ = rec * n_c + rec.mean() * 10, (1 - rec) * n_c + (1 - rec.mean()) * 10
shrunk, lo = a_ / (a_ + b_), beta_dist.ppf(0.10, a_, b_)
auc = np.array([roc_auc_score(y == c, oof[:, c]) for c in range(K)])
ll = np.array([-np.log(np.clip(oof[y == c, c], 1e-6, 1)).mean() for c in range(K)])
pur = np.zeros(K)
for r in range(5):
    rng = np.random.RandomState(100 + r); n = min(150, n_c.min())
    idx = np.concatenate([rng.choice(np.where(y == c)[0], n, replace=False) for c in range(K)])
    S = torch.from_numpy(Fe[idx]) @ torch.from_numpy(Fe[idx]).T; S.fill_diagonal_(-9)
    nb = S.topk(10, dim=1).indices.numpy(); yy = y[idx]
    for c in range(K): pur[c] += (yy[nb[yy == c]] == c).mean() / 5

mm = lambda v: (v - v.min()) / (v.max() - v.min() + 1e-9)
D = pd.DataFrame({"n": n_c, "tail": [c in tail for c in classes], "oof_recall": rec, "ovr_auc": auc,
                  "d_oof_recall": mm(1 - rec), "d_shrunk": mm(1 - shrunk), "d_recall_lo": mm(1 - lo),
                  "d_auc": mm(1 - auc), "d_logloss": mm(ll), "d_bal_purity": mm(1 - pur)}, index=classes)
D.to_csv(f"{OUT}/difficulty_scores.csv")
pd.set_option("display.width", 200); print("\n", D.round(2).to_string())
print(f"D1 DONE | {time.time() - t0:.0f}s")

# %%
# ===================== CELL 5: train methods x seeds (GPU) =====================
import os, gc, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from sklearn.metrics import balanced_accuracy_score
DATASET, RES, MODE = "isic2019", 224, "squash"
METHODS = ["ce", "cb", "cb+dup", "ce+dup", "wce", "bs", "focal", "ldam"]       # add: wce, focal, bs, ce+dup, ldam+dup
SEEDS, EPOCHS, BS, LR, WD, DUP_N = [0,1,2], 12, 64, 3e-4, 1e-4, 750
DEV = "cuda:0"; assert torch.cuda.is_available(), "turn on a GPU accelerator"
OUT = f"/kaggle/working/{DATASET}"; CACHE = f"{OUT}/cache"; RUNS = f"{OUT}/runs_{RES}"; os.makedirs(RUNS, exist_ok=True)
meta = json.load(open(f"{OUT}/meta.json")); classes, tail = meta["classes"], meta["tail"]; K = len(classes); tail_i = [classes.index(c) for c in tail]
ld = lambda s: (np.ascontiguousarray(np.load(f"{CACHE}/{RES}_{MODE}_all_{s}_x.npy", mmap_mode="r")), np.load(f"{CACHE}/{RES}_{MODE}_all_{s}_y.npy"))
(Xtr,Ytr),(Xva,Yva),(Xte,Yte) = ld("train"), ld("val"), ld("test")
counts = np.bincount(Ytr, minlength=K); prior = counts/counts.sum(); print(dict(zip(classes, counts.tolist())), "| tail", tail)
MEAN = torch.tensor([0.485,0.456,0.406], device=DEV).view(1,3,1,1); STD = torch.tensor([0.229,0.224,0.225], device=DEV).view(1,3,1,1)
to_gpu = lambda xb: torch.from_numpy(xb).to(DEV, non_blocking=True).permute(0,3,1,2).float()/255.0
def augment(x):
    n = x.shape[0]; r = lambda: torch.rand(n, device=DEV)
    ang = (r()*2-1)*np.pi; sc = 0.85+0.3*r(); flip = (r()<0.5).float()*2-1; tx, ty = (r()*2-1)*0.1, (r()*2-1)*0.1
    cos, sin = torch.cos(ang)/sc, torch.sin(ang)/sc
    th = torch.stack([torch.stack([cos*flip,-sin,tx],1), torch.stack([sin*flip,cos,ty],1)],1)
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode="reflection", align_corners=False)
    b = 1+(r().view(n,1,1,1)*2-1)*0.15; c = 1+(r().view(n,1,1,1)*2-1)*0.15; m = x.mean((1,2,3), keepdim=True)
    return ((x-m)*c+m*b).clamp(0,1)
def make_model():
    m = torchvision.models.efficientnet_b0(weights="DEFAULT"); m.classifier[1] = nn.Linear(1280, K); return m.to(DEV)
def make_loss(base):
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    cbw = (1-0.999)/(1-0.999**counts); cbw = t(cbw/cbw.sum()*K); ww = 1.0/counts; ww = t(ww/ww.sum()*K); lp = t(np.log(prior))
    if base=="ce": return lambda z,y,ep: F.cross_entropy(z,y)
    if base=="wce": return lambda z,y,ep: F.cross_entropy(z,y,weight=ww)
    if base=="cb": return lambda z,y,ep: F.cross_entropy(z,y,weight=cbw)
    if base=="bs": return lambda z,y,ep: F.cross_entropy(z+lp,y)
    if base=="focal":
        def f(z,y,ep): l = F.log_softmax(z,1).gather(1,y[:,None])[:,0]; return (-(1-l.exp())**2*l).mean()
        return f
    if base=="ldam":
        m = 1/np.sqrt(np.sqrt(counts)); m = t(m*0.5/m.max())
        return lambda z,y,ep: F.cross_entropy(z-F.one_hot(y,K)*m, y, weight=cbw if ep>=int(0.8*EPOCHS) else None)
    raise ValueError(base)
@torch.no_grad()
def logits_of(model, X):
    model.eval(); out = []
    for i in range(0, len(X), 128):
        x = (to_gpu(X[i:i+128])-MEAN)/STD
        with torch.autocast("cuda", dtype=torch.float16): out.append(((model(x).float()+model(x.flip(3)).float())/2).cpu())
    return torch.cat(out).numpy()
def run(method, seed):
    base, dup = method.replace("+dup",""), method.endswith("+dup"); name = f"{method}_s{seed}"; vf, tf = f"{RUNS}/{name}_val.npy", f"{RUNS}/{name}_test.npy"
    if os.path.exists(tf): print(name,"done"); return
    torch.manual_seed(seed); np.random.seed(seed); ids = np.arange(len(Ytr))
    if dup:
        rg = np.random.RandomState(seed); ids = np.concatenate([ids]+[rg.choice(np.where(Ytr==c)[0], DUP_N, replace=True) for c in tail_i])
    crit = make_loss(base); model = make_model(); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); spe = len(ids)//BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS*spe, pct_start=0.1); scaler = torch.amp.GradScaler("cuda"); best = -1
    print(f"{name} @ {RES}px: {len(ids)} training images")
    for ep in range(EPOCHS):
        t0 = time.time(); model.train(); perm = np.random.permutation(len(ids)); tot = 0.0
        for b in range(spe):
            bi = np.sort(ids[perm[b*BS:(b+1)*BS]]); x = (augment(to_gpu(Xtr[bi]))-MEAN)/STD; y = torch.from_numpy(Ytr[bi]).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16): loss = crit(model(x).float(), y, ep)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        zv = logits_of(model, Xva); vb = balanced_accuracy_score(Yva, zv.argmax(1))
        if vb > best: best = vb; np.save(vf, zv); np.save(tf, logits_of(model, Xte))
        print(f"  ep {ep+1}/{EPOCHS} loss {tot/spe:.3f} | val bma {vb:.3f} | {time.time()-t0:.0f}s")
    del model, opt; gc.collect(); torch.cuda.empty_cache()
for sd in SEEDS:
    for m in METHODS: run(m, sd)
print("CELL 5 DONE")

# %%
# ===================== CELL 5b: sampling, mixup, two-stage (cRT, LWS, tau-norm) =====================
METHODS2 = ["sqrt", "cbs", "mixup", "twostage"]     # twostage produces crt, lws, taunorm
def train_loop(name, seed, mode):
    vf, tf = f"{RUNS}/{name}_val.npy", f"{RUNS}/{name}_test.npy"
    torch.manual_seed(seed); np.random.seed(seed); model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); spe = len(Ytr)//BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS*spe, pct_start=0.1); scaler = torch.amp.GradScaler("cuda"); best = -1
    q = {"sqrt": 0.5, "cbs": 0.0}.get(mode, 1.0); pw = counts[Ytr].astype(float)**(q-1); pw /= pw.sum()
    print(f"{name} @ {RES}px")
    for ep in range(EPOCHS):
        t0 = time.time(); model.train(); tot = 0.0
        order = np.random.choice(len(Ytr), spe*BS, replace=True, p=pw) if mode in ("sqrt","cbs") else np.random.permutation(len(Ytr))
        for b in range(spe):
            bi = np.sort(order[b*BS:(b+1)*BS]); x = (augment(to_gpu(Xtr[bi]))-MEAN)/STD; y = torch.from_numpy(Ytr[bi]).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                if mode == "mixup":
                    lam = float(np.random.beta(0.4, 0.4)); pr_ = torch.randperm(len(bi), device=DEV); z = model(lam*x+(1-lam)*x[pr_]).float()
                    loss = lam*F.cross_entropy(z,y)+(1-lam)*F.cross_entropy(z,y[pr_])
                else: loss = F.cross_entropy(model(x).float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        zv = logits_of(model, Xva); vb = balanced_accuracy_score(Yva, zv.argmax(1))
        if mode != "twostage" and vb > best: best = vb; np.save(vf, zv); np.save(tf, logits_of(model, Xte))
        print(f"  ep {ep+1}/{EPOCHS} loss {tot/spe:.3f} | val bma {vb:.3f} | {time.time()-t0:.0f}s")
    return model

@torch.no_grad()
def feats_of(model, X):
    model.eval(); out = []
    for i in range(0, len(X), 128):
        x = (to_gpu(X[i:i+128])-MEAN)/STD
        with torch.autocast("cuda", dtype=torch.float16):
            f = (model.avgpool(model.features(x)).flatten(1).float()+model.avgpool(model.features(x.flip(3))).flatten(1).float())/2
        out.append(f.cpu())
    return torch.cat(out).to(DEV)

def twostage(seed):
    if all(os.path.exists(f"{RUNS}/{n}_s{seed}_test.npy") for n in ("crt","lws","taunorm")): print("twostage done", seed); return
    model = train_loop(f"stage1_s{seed}", seed, "twostage")
    Ftr, Fva, Fte = feats_of(model, Xtr), feats_of(model, Xva), feats_of(model, Xte)
    W, b = model.classifier[1].weight.detach().float(), model.classifier[1].bias.detach().float(); y = torch.from_numpy(Ytr).to(DEV)
    ww = torch.tensor(1.0/counts, dtype=torch.float32, device=DEV); ww = ww/ww.sum()*K; bm = lambda z: balanced_accuracy_score(Yva, z.argmax(1))
    def fit(params, fwd, lr, ep=40):
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(ep):
            perm = torch.randperm(len(y), device=DEV)
            for i in range(0, len(y), 512):
                j = perm[i:i+512]; loss = F.cross_entropy(fwd(Ftr[j]), y[j], weight=ww); opt.zero_grad(); loss.backward(); opt.step()
    h = nn.Linear(1280, K).to(DEV); fit(h.parameters(), lambda f: h(f), 1e-3)                       # cRT
    s = torch.ones(K, device=DEV, requires_grad=True); fit([s], lambda f: f@(W*s[:,None]).T+b, 1e-2)  # LWS
    best_t, best_v = 0, -1                                                                          # tau-norm
    for t in np.arange(0, 1.51, 0.25):
        Wn = W/(W.norm(dim=1, keepdim=True)**t); v = bm((Fva@Wn.T).cpu().numpy())
        if v > best_v: best_t, best_v = t, v
    Wn = W/(W.norm(dim=1, keepdim=True)**best_t); print("tau-norm tau =", best_t)
    heads = {"crt": lambda f: h(f), "lws": lambda f: f@(W*s[:,None]).T+b, "taunorm": lambda f: f@Wn.T}
    with torch.no_grad():
        for n, fn in heads.items(): np.save(f"{RUNS}/{n}_s{seed}_val.npy", fn(Fva).cpu().numpy()); np.save(f"{RUNS}/{n}_s{seed}_test.npy", fn(Fte).cpu().numpy())
    del model; gc.collect(); torch.cuda.empty_cache()

for sd in SEEDS:
    for m in METHODS2:
        if m == "twostage": twostage(sd)
        elif os.path.exists(f"{RUNS}/{m}_s{sd}_test.npy"): print(m, sd, "done")
        else: del_ = train_loop(f"{m}_s{sd}", sd, m); del del_; gc.collect(); torch.cuda.empty_cache()
print("CELL 5b DONE")

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
