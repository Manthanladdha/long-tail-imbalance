# %%
# ===================== CELL 0: environment check =====================
import os, shutil, torch
print("cuda:", torch.cuda.is_available(), "| free GB:", round(shutil.disk_usage("/kaggle/working").free/1e9,1))
for r, d, f in os.walk("/kaggle/input"):
    if r.count("/") <= 5: print(r, "|", len(f), "files")

# %%
# ===================== CELL 1: manifest (dataset adapter + split) =====================
import os, json
import numpy as np, pandas as pd
# ---------- settings: change only this block for a new dataset ----------
DATASET, KIND, ROOT = "isic2019", "isic2019", "/kaggle/input"   # KIND: isic2019 | folder | csv | medmnist
CSV_PATH, IMG_COL, LABEL_COL, GROUP_COL, IMG_DIR = None, "image", "label", None, None
TAIL_FRAC, VAL_HEAD, VAL_TAIL, TEST_FRAC, SEED = 0.10, 0.08, 0.20, 0.15, 42
# -------------------------------------------------------------------------
WORK = "/kaggle/working"; OUT = f"{WORK}/{DATASET}"; os.makedirs(OUT, exist_ok=True)
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
ISIC = ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC"]
imgs_of = lambda root: {os.path.splitext(f)[0]: os.path.join(r, f) for r,_,fs in os.walk(root) for f in fs if f.lower().endswith(EXT)}

def adapt_isic():
    imgs = imgs_of(ROOT); csvs = [os.path.join(r,f) for r,_,fs in os.walk(ROOT) for f in fs if f.endswith(".csv")]
    pick = lambda *m: [c for c in csvs if all(x in os.path.basename(c).lower() for x in m)][0]
    gt = pd.read_csv(pick("train","groundtruth")).merge(pd.read_csv(pick("train","metadata"))[["image","lesion_id"]], on="image", how="left")
    gt["group"] = gt["lesion_id"].fillna(gt["image"]); gt["split_hint"] = None
    te = pd.read_csv(pick("test","groundtruth")); te["group"] = te["image"]; te["split_hint"] = "test"
    strip = {k.replace("_downsampled",""): v for k,v in imgs.items()}; out = []
    for d in (gt, te):
        d = d[d[ISIC].sum(axis=1) == 1].copy(); d["class"] = d[ISIC].values.argmax(1); d["class"] = d["class"].map(dict(enumerate(ISIC)))
        d["path"] = d["image"].map(imgs).fillna(d["image"].str.replace("_downsampled","",regex=False).map(strip))
        assert d["path"].notna().all(), "csv rows without image files"
        out.append(d[["image","path","class","group","split_hint"]])
    return pd.concat(out, ignore_index=True), ISIC

def adapt_folder():
    tops = {d.lower() for d in os.listdir(ROOT) if os.path.isdir(f"{ROOT}/{d}")}
    has = len(tops & {"train","val","valid","validation","test"}) >= 2; rows = []
    for r,_,fs in os.walk(ROOT):
        rel = os.path.relpath(r, ROOT).split(os.sep)
        if rel == ["."]: continue
        hint = {"valid":"val","validation":"val"}.get(rel[0].lower(), rel[0].lower()) if has else None
        cls = (rel[1] if len(rel) > 1 else None) if has else rel[0]
        if cls is None: continue
        rows += [{"image": f"{cls}/{f}", "path": os.path.join(r,f), "class": cls, "group": f"{cls}/{f}", "split_hint": hint} for f in fs if f.lower().endswith(EXT)]
    d = pd.DataFrame(rows); return d, sorted(d["class"].unique())

def adapt_csv():
    d = pd.read_csv(CSV_PATH); im = imgs_of(IMG_DIR or ROOT); d["image"] = d[IMG_COL].astype(str); d["class"] = d[LABEL_COL].astype(str)
    d["path"] = d["image"].map(lambda s: im.get(os.path.splitext(os.path.basename(s))[0])); assert d["path"].notna().all()
    d["group"] = d[GROUP_COL].astype(str) if GROUP_COL else d["image"]; d["split_hint"] = None
    return d[["image","path","class","group","split_hint"]], sorted(d["class"].unique())

def adapt_medmnist():
    from PIL import Image
    z = np.load(ROOT); n = int(max(z[f"{s}_labels"].max() for s in ("train","val","test")) + 1); names = [f"c{i}" for i in range(n)]; rows = []
    for s in ("train","val","test"):
        X, Y = z[f"{s}_images"], z[f"{s}_labels"].reshape(-1); os.makedirs(f"{OUT}/png/{s}", exist_ok=True)
        for i in range(len(X)):
            p = f"{OUT}/png/{s}/{i}.png"
            if not os.path.exists(p): Image.fromarray(X[i]).save(p)
            rows.append({"image": f"{s}_{i}", "path": p, "class": names[int(Y[i])], "group": f"{s}_{i}", "split_hint": s})
    return pd.DataFrame(rows), names

df, classes = {"isic2019": adapt_isic, "folder": adapt_folder, "csv": adapt_csv, "medmnist": adapt_medmnist}[KIND]()
rng = np.random.RandomState(SEED)
def carve(sub, frac):
    gc = sub.groupby("group")["class"].agg(lambda s: s.mode().iat[0]); gs_ = sub.groupby("group").size(); take = set()
    for c in classes:
        g = gc.index[gc == c].to_numpy().copy(); rng.shuffle(g); n = int(gs_[g].sum()); acc = 0
        for x in g:
            if acc < frac(c) * n: take.add(x); acc += int(gs_[x])
    return take
df["split"] = df["split_hint"].replace({"valid": "val"})
if not (df["split"] == "test").any(): df.loc[df["group"].isin(carve(df, lambda c: TEST_FRAC)), "split"] = "test"
rest = df[df["split"].isna() | (df["split"] == "train")]
if not (df["split"] == "val").any():
    cnt = rest["class"].value_counts(); big = cnt.max()
    fv = lambda c: VAL_TAIL if cnt.get(c, 0) < TAIL_FRAC * big else VAL_HEAD
    df.loc[df["group"].isin(carve(rest, fv)) & df["split"].isna(), "split"] = "val"
df["split"] = df["split"].fillna("train")
lab = df[df.split.isin(["train","val"])]; assert lab.groupby("group")["split"].nunique().max() == 1 and df["image"].is_unique
cnt = df[df.split == "train"]["class"].value_counts().reindex(classes).fillna(0)
tail = [c for c in classes if cnt[c] < TAIL_FRAC * cnt.max()]; head = [c for c in classes if c not in tail]
df.to_csv(f"{OUT}/manifest.csv", index=False); json.dump({"dataset": DATASET, "classes": classes, "tail": tail, "head": head}, open(f"{OUT}/meta.json","w"), indent=2)
print(pd.crosstab(df["class"], df["split"]).reindex(classes).to_string()); print("tail:", tail)

# %%
# ===================== CELL 2: multi-resolution cache =====================
import os, json, shutil, time
import numpy as np, pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
DATASET = "isic2019"; CONFIGS = [(224,"all"), (320,"all"), (512,"tail")]; MODE = "squash"
OUT = f"/kaggle/working/{DATASET}"; CACHE = f"{OUT}/cache"; os.makedirs(CACHE, exist_ok=True)
df = pd.read_csv(f"{OUT}/manifest.csv"); meta = json.load(open(f"{OUT}/meta.json")); classes, tail = meta["classes"], meta["tail"]; cid = {c:i for i,c in enumerate(classes)}
need = sum(((df.split==s).sum() if o=="all" else df[df.split==s]["class"].isin(tail).sum())*r*r*3 for r,o in CONFIGS for s in ("train","val","test") if not (o=="tail" and s!="train"))/1e9
free = shutil.disk_usage("/kaggle/working").free/1e9; print(f"needs ~{need:.1f} GB | free {free:.1f} GB"); assert need < free-2, "not enough disk"
for split in ("train","val","test"):
    sub = df[df.split==split].reset_index(drop=True); n = len(sub); specs = []
    for r,o in CONFIGS:
        if o=="tail" and split!="train": continue
        inc = sub["class"].isin(tail).values if o=="tail" else np.ones(n,bool); tag = f"{r}_{MODE}_{o}_{split}"
        rows = sub[inc].reset_index(drop=True); rows.to_csv(f"{CACHE}/{tag}_rows.csv", index=False)
        np.save(f"{CACHE}/{tag}_y.npy", rows["class"].map(cid).values.astype(np.int64))
        specs.append({"r":r,"inc":inc,"pos":np.cumsum(inc)-1,"f":f"{CACHE}/{tag}_x.npy","n":int(inc.sum())})
    pf = f"{CACHE}/progress_{split}.txt"; done = int(open(pf).read()) if os.path.exists(pf) else 0
    if done >= n: print(split,"already cached"); continue
    for s in specs: s["mm"] = np.lib.format.open_memmap(s["f"], mode="r+" if (os.path.exists(s["f"]) and done>0) else "w+", dtype=np.uint8, shape=(s["n"],s["r"],s["r"],3))
    maxr = max(s["r"] for s in specs); paths = sub["path"].tolist()
    def work(i):
        nd = [s for s in specs if s["inc"][i]]
        if not nd: return i, []
        im = Image.open(paths[i]); im.draft("RGB",(maxr,maxr)); im = im.convert("RGB")
        return i, [(s, np.asarray(im.resize((s["r"],s["r"]), Image.LANCZOS))) for s in nd]
    with ThreadPoolExecutor(8) as ex:
        for b0 in tqdm(range(done, n, 256), desc=f"{split} ({n})"):
            b1 = min(b0+256, n)
            for i, outs in ex.map(work, range(b0,b1)):
                for s, a in outs: s["mm"][s["pos"][i]] = a
            for s in specs: s["mm"].flush()
            open(pf,"w").write(str(b1))
    for s in specs: del s["mm"]
print("cache GB:", round(sum(os.path.getsize(f"{CACHE}/{f}") for f in os.listdir(CACHE))/1e9,1), "| free GB:", round(shutil.disk_usage("/kaggle/working").free/1e9,1))

# %%
# ===================== CELL 3: verify cache =====================
import glob, os, numpy as np
C = "/kaggle/working/isic2019/cache"
for f in sorted(glob.glob(f"{C}/*_x.npy")):
    x = np.load(f, mmap_mode="r"); y = np.load(f.replace("_x.npy","_y.npy")); print(os.path.basename(f), x.shape, "OK" if len(x)==len(y) else "MISMATCH")
print({s: open(f"{C}/progress_{s}.txt").read() for s in ("train","val","test")})
