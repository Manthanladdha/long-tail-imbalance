# %%
import os, sys, json, shutil, zipfile, platform, collections, urllib.request, urllib.error
import importlib.metadata as md
import pandas as pd

INPUT, WORK = "/kaggle/input", "/kaggle/working"
os.makedirs(WORK, exist_ok=True)
report = {}

def ver(pkg):
    try:
        return md.version(pkg)
    except Exception:
        return "not installed"

print("=" * 60, "\nSYSTEM")
print("python", sys.version.split()[0], "|", platform.platform())
pk = ["torch", "torchvision", "torchao", "diffusers", "transformers", "huggingface_hub", "accelerate", "peft", "timm", "scikit-learn", "scikit-image", "scipy", "pandas", "numpy", "matplotlib", "pillow"]
report["packages"] = {p: ver(p) for p in pk}
for p in pk:
    print(f"  {p:18s}{report['packages'][p]}")

try:
    import torch
    report["cuda"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        report["gpu"] = {"name": pr.name, "memory_gb": round(pr.total_memory / 1e9, 1), "count": torch.cuda.device_count()}
        print(f"  GPU: {pr.name} | {report['gpu']['memory_gb']} GB | devices: {torch.cuda.device_count()}")
    else:
        print("  GPU: none (turn on an accelerator in the notebook settings)")
except Exception as e:
    print("  torch could not be imported:", e)

mem = {}
try:
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        if k in ("MemTotal", "MemAvailable"):
            mem[k] = round(int(v.split()[0]) / 1e6, 1)
except Exception:
    pass
du = shutil.disk_usage(WORK)
report["ram_gb"] = mem
report["disk_working_gb"] = {"free": round(du.free / 1e9, 1), "total": round(du.total / 1e9, 1)}
print(f"  RAM (GB): {mem} | disk in {WORK} (GB): free {report['disk_working_gb']['free']} of {report['disk_working_gb']['total']}")

print("=" * 60, "\nINTERNET")
def reach(url):
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=6)
        return "ok"
    except urllib.error.HTTPError as e:
        return f"ok (HTTP {e.code})"
    except Exception as e:
        return f"blocked ({type(e).__name__})"
report["internet"] = {u: reach(u) for u in ["https://huggingface.co", "https://download.pytorch.org", "https://pypi.org"]}
for u, r in report["internet"].items():
    print(f"  {u:32s}{r}")

print("=" * 60, "\nINPUT FILES")
if not os.path.exists(INPUT) or not os.listdir(INPUT):
    print("  nothing is attached: use Add Input in the right panel, then run this cell again")
    report["input"] = "empty"
else:
    exts, csvs, zips, dirs_seen = collections.Counter(), [], [], []
    for root, dirs, files in os.walk(INPUT):
        depth = root[len(INPUT):].count(os.sep)
        if depth <= 3:
            dirs_seen.append((depth, root, len(files), len(dirs)))
        for f in files:
            e = os.path.splitext(f)[1].lower()
            exts[e] += 1
            if e == ".csv":
                csvs.append(os.path.join(root, f))
            elif e == ".zip":
                zips.append(os.path.join(root, f))
    for depth, root, nf, nd in dirs_seen[:40]:
        print(f"  {'  ' * depth}{root}  [{nf} files, {nd} folders]")
    if len(dirs_seen) > 40:
        print(f"  ... {len(dirs_seen) - 40} more folders")
    print("  file types loose in the input folders:", dict(exts))
    report["input"] = {"extensions": dict(exts), "csv": [], "zip": []}

    print("\n  CSV files")
    for p in csvs:
        try:
            head = pd.read_csv(p, nrows=3)
            rows = sum(1 for _ in open(p, "rb")) - 1
            report["input"]["csv"].append({"path": p, "rows": rows, "columns": list(head.columns)})
            print(f"  {os.path.basename(p)}: {rows} rows | columns: {list(head.columns)}")
        except Exception as e:
            print(f"  {os.path.basename(p)}: could not be read ({e})")
    if not csvs:
        print("  none loose; they may be inside zip files, see below")

    print("\n  ZIP files")
    for p in zips:
        try:
            with zipfile.ZipFile(p) as z:
                names = [n for n in z.namelist() if not n.endswith("/")]
            z2 = zipfile.ZipFile(p)
            ec = collections.Counter(os.path.splitext(n)[1].lower() for n in names)
            info = {"path": p, "size_gb": round(os.path.getsize(p) / 1e9, 2), "members": len(names), "types": dict(ec), "sample": names[:3], "csv_members": [n for n in names if n.lower().endswith(".csv")][:10]}
            report["input"]["zip"].append(info)
            print(f"  {os.path.basename(p)}: {info['size_gb']} GB | {len(names)} members | types {dict(ec)}")
            print(f"    first members: {info['sample']}")
            for cm in info["csv_members"][:6]:
                with z2.open(cm) as fh:
                    head = pd.read_csv(fh, nrows=3)
                with z2.open(cm) as fh:
                    rows = sum(1 for _ in fh) - 1
                info.setdefault("csv_info", []).append({"name": cm, "rows": rows, "columns": list(head.columns)})
                print(f"    csv inside: {os.path.basename(cm)}: {rows} rows | columns: {list(head.columns)}")
            z2.close()
        except Exception as e:
            print(f"  {os.path.basename(p)}: could not be opened ({e})")
    if not zips:
        print("  none")

json.dump(report, open(f"{WORK}/env_report.json", "w"), indent=2, default=str)
print("=" * 60, "\nENV CHECK DONE | report saved to", f"{WORK}/env_report.json")

# %%
import sys, gc, json, time, copy, subprocess

WORK = "/kaggle/working"
REPOS = ["stable-diffusion-v1-5/stable-diffusion-inpainting", "benjamin-paine/stable-diffusion-v1-5-inpainting", "runwayml/stable-diffusion-inpainting"]
t_all = time.time()

r = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True, text=True)
print("torchao:", (r.stdout.strip().splitlines() or r.stderr.strip().splitlines() or ["no output"])[-1])

results = {}

def check(name, fn):
    t = time.time()
    try:
        out = fn()
        info = "" if out is None else str(out)[:400]
        results[name] = {"ok": True, "info": info}
        print(f"ok    {name} ({time.time() - t:.1f}s) {info}")
        return out
    except Exception as e:
        info = f"{type(e).__name__}: {str(e)[:400]}"
        results[name] = {"ok": False, "info": info}
        print(f"FAIL  {name}: {info}")
        return None

import torch
import torchvision

def gpus():
    out = []
    for i in range(torch.cuda.device_count()):
        x = torch.randn(512, 512, device=f"cuda:{i}")
        out.append(f"cuda:{i} {torch.cuda.get_device_name(i)} finite={bool(torch.isfinite(x @ x).all())}")
    assert out, "no GPU found: turn on an accelerator in the notebook settings"
    return out

def imp_diffusers():
    import diffusers
    from diffusers import UNet2DConditionModel, AutoencoderKL
    return diffusers.__version__

def imp_transformers():
    import transformers
    from transformers import AutoModel, CLIPTextModel, CLIPTokenizer
    return transformers.__version__

check("GPUs", gpus)
check("import diffusers", imp_diffusers)
check("import transformers", imp_transformers)

def pick_repo():
    from huggingface_hub import HfApi
    api = HfApi()
    for repo in REPOS:
        try:
            files = set(api.list_repo_files(repo))
        except Exception as e:
            print("   repo", repo, "not usable:", type(e).__name__)
            continue
        need = ["unet/config.json", "vae/config.json", "text_encoder/config.json", "tokenizer/vocab.json"]
        if all(f in files for f in need):
            st = {p: (f in files) for p, f in [("unet", "unet/diffusion_pytorch_model.safetensors"), ("vae", "vae/diffusion_pytorch_model.safetensors"), ("text_encoder", "text_encoder/model.safetensors")]}
            return repo, {k: ("safetensors" if v else "bin") for k, v in st.items()}
    raise RuntimeError("none of the Stable Diffusion inpainting repos could be reached")

picked = check("Stable Diffusion inpainting repo", pick_repo)
SD_REPO = picked[0] if picked else None

def unet_cfg():
    from diffusers import UNet2DConditionModel
    c = UNet2DConditionModel.load_config(SD_REPO, subfolder="unet")
    assert c["in_channels"] == 9, f"unexpected in_channels {c['in_channels']}"
    return f"in_channels {c['in_channels']} | cross_attention_dim {c['cross_attention_dim']} | sample_size {c['sample_size']}"

def unet_test():
    from diffusers import UNet2DConditionModel
    dv = "cuda:0"
    unet = UNet2DConditionModel.from_pretrained(SD_REPO, subfolder="unet").to(dv).eval()
    x = torch.randn(2, 9, 32, 32, device=dv)
    cond = torch.randn(2, 77, 768, device=dv)
    t = torch.tensor([500, 500], device=dv, dtype=torch.long)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        y = unet(x, t, encoder_hidden_states=cond).sample
    assert tuple(y.shape) == (2, 4, 32, 32) and bool(torch.isfinite(y).all())
    total = sum(p.numel() for p in unet.parameters()) / 1e6
    attn = sum(p.numel() for n, p in unet.named_parameters() if "attn1" in n or "attn2" in n) / 1e6
    uh = copy.deepcopy(unet).half()
    del unet
    xb = torch.randn(32, 9, 32, 32, device=dv, dtype=torch.half)
    cb = torch.randn(32, 77, 768, device=dv, dtype=torch.half)
    tb = torch.full((32,), 500, device=dv, dtype=torch.long)
    with torch.no_grad():
        for _ in range(2):
            uh(xb, tb, encoder_hidden_states=cb)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            uh(xb, tb, encoder_hidden_states=cb)
        torch.cuda.synchronize()
        per = (time.time() - t0) / 3
    del uh, xb, cb
    gc.collect()
    torch.cuda.empty_cache()
    results["_bench"] = {"unet_fp16_batch32_s": per, "images_per_s_25_steps": 32 / (25 * per)}
    return f"output {tuple(y.shape)} | params {total:.0f}M (attention {attn:.0f}M) | batch-32 fp16 forward {per:.2f}s -> about {32 / (25 * per):.2f} images/s with 25 steps, before VAE decode"

def vae_test():
    from diffusers import AutoencoderKL
    dv = "cuda:0"
    vae = AutoencoderKL.from_pretrained(SD_REPO, subfolder="vae").to(dv).eval()
    x = torch.randn(1, 3, 256, 256, device=dv).clamp(-1, 1)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.sample()
        y = vae.decode(z).sample
    assert tuple(z.shape) == (1, 4, 32, 32) and tuple(y.shape) == (1, 3, 256, 256) and bool(torch.isfinite(y).all())
    sf = vae.config.scaling_factor
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    return f"latent {tuple(z.shape)} | scaling_factor {sf}"

def text_test():
    from transformers import CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(SD_REPO, subfolder="tokenizer")
    txt = CLIPTextModel.from_pretrained(SD_REPO, subfolder="text_encoder").to("cuda:0").eval()
    ids = tok(["a dermoscopy image of dermatofibroma"], padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        h = txt(ids).last_hidden_state
    assert tuple(h.shape) == (1, 77, 768) and bool(torch.isfinite(h).all())
    del txt
    gc.collect()
    torch.cuda.empty_cache()
    return f"embedding {tuple(h.shape)}"

if SD_REPO:
    check("inpainting UNet config", unet_cfg)
    check("UNet load, forward, speed", unet_test)
    check("VAE load, encode, decode", vae_test)
    check("CLIP text encoder", text_test)
else:
    print("skipped the Stable Diffusion checks because no repo was reachable")

def dino_test():
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-small").to("cuda:0").eval()
    with torch.no_grad():
        o = m(pixel_values=torch.randn(2, 3, 224, 224, device="cuda:0")).pooler_output
    assert tuple(o.shape) == (2, 384) and bool(torch.isfinite(o).all())
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return f"embedding {tuple(o.shape)}"

def effnet_test():
    m = torchvision.models.efficientnet_b0(weights="DEFAULT").to("cuda:0").eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        o = m(torch.randn(8, 3, 224, 224, device="cuda:0"))
    assert tuple(o.shape) == (8, 1000)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return "pretrained weights downloaded and run"

check("DINOv2-small", dino_test)
check("EfficientNet-B0 pretrained", effnet_test)

bad = [k for k, v in results.items() if not k.startswith("_") and not v["ok"]]
json.dump({"sd_repo": SD_REPO, "results": results}, open(f"{WORK}/smoke_test.json", "w"), indent=2, default=str)
print("=" * 60)
if bad:
    print("PROBLEMS:", bad)
    if any("import" in b for b in bad):
        print("an import failed: restart the session (Run > Restart session), then run this cell again before anything else")
else:
    print(f"ENVIRONMENT READY | repo {SD_REPO} | {time.time() - t_all:.0f}s")

# %%
import os, gc, time, hashlib
import numpy as np, pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

INPUT, WORK = "/kaggle/input", "/kaggle/working"
SEED, S = 42, 224
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
TAIL = ["AK", "DF", "VASC", "SCC"]
K = len(CLASSES)
VAL_FRAC = {"tail": 0.20, "head": 0.08}
REBUILD_SPLITS = False
WORKERS = min(8, os.cpu_count() or 4)
CACHE, REPORT = f"{WORK}/cache", f"{WORK}/report"
for d in (CACHE, REPORT):
    os.makedirs(d, exist_ok=True)
np.random.seed(SEED)
t_start = time.time()

def load_resized(path, size):
    im = Image.open(path)
    im.draft("RGB", (2 * size, 2 * size))
    return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))

def find_csv(files, *must):
    hits = [f for f in files if all(m in os.path.basename(f).lower() for m in must)]
    assert hits, f"no csv with {must} in its name was found under {INPUT}"
    return hits[0]

def build_splits():
    images, csvs = {}, []
    for root, _, files in os.walk(INPUT):
        for f in files:
            low = f.lower()
            if low.endswith((".jpg", ".jpeg", ".png")):
                images[os.path.splitext(f)[0]] = os.path.join(root, f)
            elif low.endswith(".csv"):
                csvs.append(os.path.join(root, f))
    print("images found:", len(images), "| csv files:", [os.path.basename(c) for c in csvs])
    stripped = {k.replace("_downsampled", ""): v for k, v in images.items()}

    def prepare(gt):
        d = gt[gt[CLASSES].sum(axis=1) == 1].copy()
        d["label"] = d[CLASSES].values.argmax(1)
        d["class"] = d["label"].map(dict(enumerate(CLASSES)))
        d["path"] = d["image"].map(images)
        miss = d["path"].isna()
        d.loc[miss, "path"] = d.loc[miss, "image"].str.replace("_downsampled", "", regex=False).map(stripped)
        assert d["path"].notna().all(), f"{int(d['path'].isna().sum())} images listed in the csv have no file"
        return d[["image", "label", "class", "group", "path"]]

    gt_tr = pd.read_csv(find_csv(csvs, "train", "groundtruth"))
    mt_tr = pd.read_csv(find_csv(csvs, "train", "metadata"))
    gt_tr = gt_tr.merge(mt_tr[["image", "lesion_id"]], on="image", how="left")
    gt_tr["group"] = gt_tr["lesion_id"].fillna(gt_tr["image"])
    tr = prepare(gt_tr)
    gt_te = pd.read_csv(find_csv(csvs, "test", "groundtruth"))
    gt_te["group"] = gt_te["image"]
    n_unk = int((gt_te[CLASSES].sum(axis=1) != 1).sum())
    te = prepare(gt_te)
    print(f"official test rows: {len(gt_te)} | kept {len(te)} | dropped {n_unk} (no valid class label)")

    rng = np.random.RandomState(SEED)
    gcls = tr.groupby("group")["label"].agg(lambda s: int(s.mode().iat[0]))
    gsize = tr.groupby("group").size()
    assign = {}
    for ci, c in enumerate(CLASSES):
        gs = gcls.index[gcls == ci].to_numpy().copy()
        rng.shuffle(gs)
        n = int(gsize[gs].sum())
        fv = VAL_FRAC["tail" if c in TAIL else "head"]
        acc = 0
        for g in gs:
            assign[g] = "val" if acc < fv * n else "train"
            acc += int(gsize[g])
    tr["split"] = tr["group"].map(assign)
    te["split"] = "test"
    return pd.concat([tr, te], ignore_index=True)

SPLITS = f"{WORK}/splits.csv"
df = pd.read_csv(SPLITS) if (os.path.exists(SPLITS) and not REBUILD_SPLITS) else None
if df is not None and not all(os.path.exists(p) for p in df["path"].sample(min(15, len(df)), random_state=0)):
    print("saved split points to files that are gone, rebuilding it")
    df = None
if df is None:
    df = build_splits()
    df.to_csv(SPLITS, index=False)
    print("new split saved to", SPLITS)
else:
    print("loaded the saved split from", SPLITS)

assert df["image"].is_unique, "an image appears twice"
lab = df[df["split"].isin(["train", "val"])]
assert lab.groupby("group")["split"].nunique().max() == 1, "a lesion is split between train and val"
assert set(df.loc[df.split == "train", "image"]).isdisjoint(set(df.loc[df.split == "test", "image"])), "train and test overlap"
TAG = hashlib.md5(open(SPLITS, "rb").read()).hexdigest()[:8]

def cache_split(name):
    sub = df[df["split"] == name].reset_index(drop=True)
    n = len(sub)
    xf, yf, pf = (f"{CACHE}/{TAG}_{name}_x.npy", f"{CACHE}/{TAG}_{name}_y.npy", f"{CACHE}/{TAG}_{name}.progress")
    done = int(open(pf).read()) if os.path.exists(pf) else 0
    if not (os.path.exists(xf) and done >= n and os.path.exists(yf)):
        X = np.lib.format.open_memmap(xf, mode="r+" if (os.path.exists(xf) and done > 0) else "w+", dtype=np.uint8, shape=(n, S, S, 3))
        paths = sub["path"].tolist()
        starts = list(range(done, n, 512))
        with ThreadPoolExecutor(WORKERS) as ex:
            for b0 in tqdm(starts, desc=f"caching {name} ({n} images)"):
                b1 = min(b0 + 512, n)
                X[b0:b1] = np.stack(list(ex.map(lambda p: load_resized(p, S), paths[b0:b1])))
                X.flush()
                open(pf, "w").write(str(b1))
        del X
        np.save(yf, sub["label"].values.astype(np.int64))
    return np.load(xf, mmap_mode="r"), np.load(yf)

Xtr, Ytr = cache_split("train")
Xva, Yva = cache_split("val")
Xte, Yte = cache_split("test")
gc.collect()

tab = pd.crosstab(df["class"], df["split"]).reindex(CLASSES)[["train", "val", "test"]]
tab["total"] = tab.sum(axis=1)
tab["tail"] = ["yes" if c in TAIL else "" for c in CLASSES]
print("\nimages per class and split")
print(tab.to_string())
print(f"\nimbalance ratio in train (largest / smallest class): {tab['train'].max() / tab['train'].min():.0f} : 1")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
w = 0.27
for i, sp in enumerate(["train", "val", "test"]):
    ax[0].bar(np.arange(K) + (i - 1) * w, tab[sp].values, w, label=sp)
ax[0].set_yscale("log")
ax[0].set_xticks(range(K))
ax[0].set_xticklabels(CLASSES)
ax[0].set_title("images per class (log scale)")
ax[0].legend()
share = tab[["train", "val", "test"]].div(tab[["train", "val", "test"]].sum(axis=0), axis=1) * 100
ax[1].bar(range(K), share["test"].values, 0.4, label="test share %")
ax[1].bar(np.arange(K) + 0.4, share["train"].values, 0.4, label="train share %")
ax[1].set_xticks(range(K))
ax[1].set_xticklabels(CLASSES)
ax[1].set_title("class share: train vs test")
ax[1].legend()
plt.tight_layout()
plt.savefig(f"{REPORT}/data_overview.png", dpi=110)
plt.show()

rng = np.random.RandomState(1)
fig, ax = plt.subplots(K, 6, figsize=(12, 2 * K))
for ci in range(K):
    for j in range(6):
        if j < 3:
            idx = rng.choice(np.where(Ytr == ci)[0])
            img, ttl = Xtr[idx], f"{CLASSES[ci]} train"
        else:
            pool = np.where(Yte == ci)[0]
            idx = rng.choice(pool) if len(pool) else None
            img, ttl = (Xte[idx], f"{CLASSES[ci]} test") if idx is not None else (np.zeros((S, S, 3), np.uint8), "none")
        ax[ci, j].imshow(img)
        ax[ci, j].set_title(ttl, fontsize=8)
        ax[ci, j].axis("off")
plt.tight_layout()
plt.savefig(f"{REPORT}/data_samples.png", dpi=90)
plt.show()

cache_gb = sum(os.path.getsize(os.path.join(CACHE, f)) for f in os.listdir(CACHE)) / 1e9
print(f"\nready | train {len(Xtr)} | val {len(Xva)} | test {len(Xte)} | cache {cache_gb:.1f} GB | {time.time() - t_start:.0f}s")

# %%
import os, time, json, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

DEV = "cuda:0"
EPOCHS, BS, LR, WD = 12, 64, 3e-4, 1e-4
CK = f"{WORK}/ckpt"
os.makedirs(CK, exist_ok=True)
torch.manual_seed(42); np.random.seed(42)

Xtr_r, Xva_r, Xte_r = (np.ascontiguousarray(a) for a in (Xtr, Xva, Xte))
counts = np.bincount(Ytr, minlength=K)
print("train counts:", dict(zip(CLASSES, counts.tolist())))

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

def to_gpu(xb):
    return torch.from_numpy(xb).to(DEV, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0

def augment(x):
    n = x.shape[0]
    ang = (torch.rand(n, device=DEV) * 2 - 1) * np.pi
    sc = 0.85 + 0.3 * torch.rand(n, device=DEV)
    flip = (torch.rand(n, device=DEV) < 0.5).float() * 2 - 1
    tx, ty = [(torch.rand(n, device=DEV) * 2 - 1) * 0.1 for _ in range(2)]
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    b = 1 + (torch.rand(n, 1, 1, 1, device=DEV) * 2 - 1) * 0.15
    c = 1 + (torch.rand(n, 1, 1, 1, device=DEV) * 2 - 1) * 0.15
    m = x.mean((1, 2, 3), keepdim=True)
    return ((x - m) * c + m * b).clamp(0, 1)

def make_model():
    m = torchvision.models.efficientnet_b0(weights="DEFAULT")
    m.classifier[1] = nn.Linear(1280, K)
    return m.to(DEV)

@torch.no_grad()
def predict(model, X):
    model.eval()
    out = []
    for i in range(0, len(X), 128):
        x = (to_gpu(X[i:i + 128]) - MEAN) / STD
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.softmax(model(x).float(), 1).cpu())
    return torch.cat(out).numpy()

def metrics(probs, y):
    p = probs.argmax(1)
    rec = recall_score(y, p, labels=list(range(K)), average=None, zero_division=0)
    return {"bacc": balanced_accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
            "acc": float((p == y).mean()), "recall": rec.tolist(),
            "tail_recall": float(np.mean([rec[CLASSES.index(c)] for c in TAIL]))}

def class_weights(kind):
    if kind == "plain":
        return None
    beta = 0.999
    w = (1 - beta) / (1 - beta ** counts)
    return torch.tensor(w / w.sum() * K, dtype=torch.float32, device=DEV)

def run(name, kind):
    done_f = f"{CK}/{name}_done.json"
    if os.path.exists(done_f):
        print(name, "already finished, skipping")
        return json.load(open(done_f))
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps_per = len(Xtr_r) // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * steps_per, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss(weight=class_weights(kind))
    ep0, hist, best = 0, [], -1.0
    last = f"{CK}/{name}_last.pt"
    if os.path.exists(last):
        s = torch.load(last, map_location=DEV, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); scaler.load_state_dict(s["scaler"])
        ep0, hist, best = s["epoch"], s["hist"], s["best"]
        print(f"{name}: resumed from epoch {ep0}")
    for ep in range(ep0, EPOCHS):
        t0 = time.time(); model.train()
        perm = np.random.permutation(len(Xtr_r)); tot = 0.0
        for b in range(steps_per):
            idx = np.sort(perm[b * BS:(b + 1) * BS])
            x = augment(to_gpu(Xtr_r[idx])); x = (x - MEAN) / STD
            y = torch.from_numpy(Ytr[idx]).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        mv = metrics(predict(model, Xva_r), Yva)
        hist.append({"epoch": ep + 1, "loss": tot / steps_per, "val_bacc": mv["bacc"], "val_tail_recall": mv["tail_recall"]})
        if mv["bacc"] > best:
            best = mv["bacc"]; torch.save(model.state_dict(), f"{CK}/{name}_best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep + 1, "hist": hist, "best": best}, last)
        print(f"{name} ep {ep + 1}/{EPOCHS} loss {tot / steps_per:.3f} | val bacc {mv['bacc']:.3f} tail recall {mv['tail_recall']:.3f} | {time.time() - t0:.0f}s")
    model.load_state_dict(torch.load(f"{CK}/{name}_best.pt", map_location=DEV))
    pv, pt = predict(model, Xva_r), predict(model, Xte_r)
    np.save(f"{CK}/{name}_val_probs.npy", pv); np.save(f"{CK}/{name}_test_probs.npy", pt)
    res = {"name": name, "hist": hist, "val": metrics(pv, Yva), "test": metrics(pt, Yte)}
    json.dump(res, open(done_f, "w"))
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return res

results = [run("plain_ce", "plain"), run("class_balanced_ce", "cb")]

rows = [{"run": r["name"], "val_bacc": r["val"]["bacc"], "test_bacc": r["test"]["bacc"],
         "test_macro_f1": r["test"]["macro_f1"], "test_acc": r["test"]["acc"],
         "test_tail_recall": r["test"]["tail_recall"],
         **{f"test_recall_{c}": r["test"]["recall"][i] for i, c in enumerate(CLASSES)}} for r in results]
log = pd.DataFrame(rows)
log.to_csv(f"{WORK}/results_log.csv", index=False)
print("\n", log.round(3).T.to_string())

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for r in results:
    h = pd.DataFrame(r["hist"])
    ax[0].plot(h["epoch"], h["val_bacc"], label=r["name"])
    ax[1].plot(h["epoch"], h["val_tail_recall"], label=r["name"])
ax[0].set_title("val balanced accuracy"); ax[1].set_title("val tail-class recall")
for a in ax[:2]: a.set_xlabel("epoch"); a.legend()
w = 0.38
for i, r in enumerate(results):
    ax[2].bar(np.arange(K) + (i - 0.5) * w, r["test"]["recall"], w, label=r["name"])
ax[2].set_xticks(range(K)); ax[2].set_xticklabels(CLASSES); ax[2].set_title("test recall per class"); ax[2].legend()
plt.tight_layout(); plt.savefig(f"{REPORT}/baselines.png", dpi=110); plt.show()

# %%
import os, gc, time, json
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy import ndimage as ndi
from skimage import color, filters, morphology
from sklearn.decomposition import PCA
from transformers import AutoModel

DEV = "cuda:0"
CK = f"{WORK}/ckpt"
MASKS, FEATS = f"{WORK}/tail_masks.npz", f"{WORK}/dino_feats.npz"
S_ = Xtr.shape[1]
yy, xx = np.mgrid[:S_, :S_]
rr = np.hypot(yy - S_ / 2 + 0.5, xx - S_ / 2 + 0.5) / (S_ / 2)
VALID, RING = rr < 0.97, (rr > 0.80) & (rr < 0.97)
t0 = time.time()

def lesion_mask(img):
    lab = ndi.gaussian_filter(color.rgb2lab(img), (2, 2, 0))
    bg = np.median(lab[RING], axis=0)
    d = np.linalg.norm(lab - bg, axis=2)
    d[~VALID] = 0
    m = (d > filters.threshold_otsu(d[VALID])) & VALID
    m = morphology.opening(m, morphology.disk(3))
    m = morphology.closing(m, morphology.disk(5))
    m = ndi.binary_fill_holes(m)
    lbl, n = ndi.label(m)
    if n > 0:
        sizes = ndi.sum(m, lbl, range(1, n + 1))
        m = lbl == (int(np.argmax(sizes)) + 1)
    frac = float(m.mean())
    ok = 0.03 < frac < 0.80
    if not ok:
        m = rr < 0.35
    return m, ok, frac

# 1) masks for tail-class training images
tail_ids = np.where(np.isin(Ytr, [CLASSES.index(c) for c in TAIL]))[0]
if os.path.exists(MASKS):
    z = np.load(MASKS)
    mask_idx, mask_arr, mask_ok = z["idx"], z["masks"], z["ok"]
    print("loaded masks:", len(mask_idx))
else:
    ms, oks = [], []
    for i in tqdm(tail_ids, desc="lesion masks"):
        m, ok, _ = lesion_mask(np.asarray(Xtr[i]))
        ms.append(m.astype(np.uint8)); oks.append(ok)
    mask_idx, mask_arr, mask_ok = tail_ids, np.stack(ms), np.array(oks)
    np.savez_compressed(MASKS, idx=mask_idx, masks=mask_arr, ok=mask_ok)
print(f"masks: {len(mask_idx)} | automatic segmentation accepted: {mask_ok.mean() * 100:.1f}% | "
      f"mean lesion area {mask_arr.mean() * 100:.1f}% of image")

# 2) DINOv2 features
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

@torch.no_grad()
def embed(model, X):
    out = []
    for i in range(0, len(X), 128):
        x = torch.from_numpy(np.ascontiguousarray(X[i:i + 128])).to(DEV).permute(0, 3, 1, 2).float() / 255
        with torch.autocast("cuda", dtype=torch.float16):
            f = model(pixel_values=(x - MEAN) / STD).pooler_output.float()
        out.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(out).numpy()

if os.path.exists(FEATS):
    z = np.load(FEATS); Ftr, Fva = z["tr"], z["va"]
    print("loaded features", Ftr.shape, Fva.shape)
else:
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()
    Ftr, Fva = embed(dino, Xtr), embed(dino, Xva)
    np.savez_compressed(FEATS, tr=Ftr, va=Fva)
    del dino; gc.collect(); torch.cuda.empty_cache()

# 3) per-class statistics
Ft = torch.from_numpy(Ftr).to(DEV)
yt = torch.from_numpy(Ytr).to(DEV)
purity = torch.zeros(len(Ftr))
for i in range(0, len(Ftr), 2048):
    sim = Ft[i:i + 2048] @ Ft.T
    sim[torch.arange(sim.shape[0]), torch.arange(i, i + sim.shape[0])] = -9
    nb = sim.topk(10, dim=1).indices
    purity[i:i + 2048] = (yt[nb] == yt[i:i + 2048, None]).float().mean(1).cpu()
purity = purity.numpy()
del Ft; torch.cuda.empty_cache()

vp = {n: np.load(f"{CK}/{n}_val_probs.npy").argmax(1) for n in ("plain_ce", "class_balanced_ce")}
rows = []
for ci, c in enumerate(CLASSES):
    f = Ftr[Ytr == ci]
    cen = f.mean(0); cen_n = cen / np.linalg.norm(cen)
    spread = float((1 - f @ cen_n).mean())
    ev = np.clip(np.linalg.eigvalsh(np.cov(f.T)), 0, None)
    part = float(ev.sum() ** 2 / (ev ** 2).sum())
    vm = Yva == ci
    rows.append({"class": c, "tail": c in TAIL, "n_train": int((Ytr == ci).sum()), "n_val": int(vm.sum()),
                 "val_recall_plain": float((vp["plain_ce"][vm] == ci).mean()),
                 "val_recall_cb": float((vp["class_balanced_ce"][vm] == ci).mean()),
                 "knn_purity": float(purity[Ytr == ci].mean()), "spread": spread, "diversity": part})
st = pd.DataFrame(rows).set_index("class")
st["difficulty"] = 0.5 * (1 - st["val_recall_cb"]) + 0.5 * (1 - st["knn_purity"])
st.to_csv(f"{WORK}/class_stats.csv")
print("\n", st.round(3).to_string())

# 4) plots
fig, ax = plt.subplots(len(TAIL), 6, figsize=(12, 2 * len(TAIL)))
rng = np.random.RandomState(0)
for r, c in enumerate(TAIL):
    ids = np.where(Ytr[mask_idx] == CLASSES.index(c))[0]
    for j, k in enumerate(rng.choice(ids, 6, replace=False)):
        img = np.asarray(Xtr[mask_idx[k]]).copy()
        edge = mask_arr[k].astype(bool) ^ ndi.binary_erosion(mask_arr[k].astype(bool), iterations=2)
        img[edge] = [0, 255, 0]
        ax[r, j].imshow(img); ax[r, j].axis("off")
        ax[r, j].set_title(f"{c}{'' if mask_ok[k] else ' (fallback)'}", fontsize=8)
plt.tight_layout(); plt.savefig(f"{REPORT}/tail_masks.png", dpi=90); plt.show()

sub = rng.choice(len(Ftr), 6000, replace=False)
P = PCA(2, random_state=0).fit_transform(Ftr[sub])
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for ci, c in enumerate(CLASSES):
    m = Ytr[sub] == ci
    ax[0].scatter(P[m, 0], P[m, 1], s=4, alpha=0.5, label=c)
ax[0].legend(markerscale=3, fontsize=7); ax[0].set_title("DINOv2 feature space (PCA)")
x = np.arange(K); w = 0.27
ax[1].bar(x - w, 1 - st["val_recall_cb"], w, label="1 - val recall"); ax[1].bar(x, 1 - st["knn_purity"], w, label="1 - kNN purity")
ax[1].bar(x + w, st["spread"], w, label="spread"); ax[1].set_xticks(x); ax[1].set_xticklabels(CLASSES); ax[1].legend(); ax[1].set_title("difficulty and spread")
ax[2].bar(x, st["difficulty"], color=["tab:red" if t else "tab:gray" for t in st["tail"]])
ax[2].set_xticks(x); ax[2].set_xticklabels(CLASSES); ax[2].set_title("combined difficulty (red = tail)")
plt.tight_layout(); plt.savefig(f"{REPORT}/class_difficulty.png", dpi=110); plt.show()
print(f"done | {time.time() - t0:.0f}s")

# %%
import os, gc, time, random
import numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig

DEV = "cuda:0"
SD_REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"
STEPS, BS, LR, RES = 3000, 8, 1e-4, 256
SAVE_EVERY, PREVIEW_EVERY = 250, 1000
CK = f"{WORK}/ckpt"; os.makedirs(CK, exist_ok=True)
GEN_CK = f"{CK}/gen_last.pt"
NAMES = {"AK": "actinic keratosis", "DF": "dermatofibroma", "VASC": "vascular lesion", "SCC": "squamous cell carcinoma"}
torch.manual_seed(0); np.random.seed(0); random.seed(0)

# data: tail images + their lesion masks
z = np.load(f"{WORK}/tail_masks.npz")
m_idx, Mt = z["idx"], z["masks"]
Xt, Yt = np.ascontiguousarray(Xtr[m_idx]), Ytr[m_idx]
tail_cls = [CLASSES.index(c) for c in TAIL]
by_cls = {c: np.where(Yt == c)[0] for c in tail_cls}
print({CLASSES[c]: len(v) for c, v in by_cls.items()})

# frozen parts
tok = CLIPTokenizer.from_pretrained(SD_REPO, subfolder="tokenizer")
txt = CLIPTextModel.from_pretrained(SD_REPO, subfolder="text_encoder").half().to(DEV).eval()
@torch.no_grad()
def encode_text(p):
    ids = tok(p, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(DEV)
    return txt(ids).last_hidden_state
EMB = {c: encode_text([f"a dermoscopy image of {NAMES[CLASSES[c]]}"]) for c in tail_cls}
UNCOND = encode_text([""])
del txt; gc.collect(); torch.cuda.empty_cache()

vae = AutoencoderKL.from_pretrained(SD_REPO, subfolder="vae").half().to(DEV).eval()
SF = vae.config.scaling_factor
unet = UNet2DConditionModel.from_pretrained(SD_REPO, subfolder="unet").half().to(DEV)
unet.requires_grad_(False)
unet.add_adapter(LoraConfig(r=16, lora_alpha=16, init_lora_weights="gaussian",
                            target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
for p in unet.parameters():
    if p.requires_grad:
        p.data = p.data.float()
params = [p for p in unet.parameters() if p.requires_grad]
print(f"trainable LoRA parameters: {sum(p.numel() for p in params) / 1e6:.2f}M")
noise_sched = DDPMScheduler.from_pretrained(SD_REPO, subfolder="scheduler")
opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-2)
scaler = torch.amp.GradScaler("cuda")

step, hist = 0, []
if os.path.exists(GEN_CK):
    s = torch.load(GEN_CK, map_location=DEV, weights_only=False)
    named = dict(unet.named_parameters())
    for n, v in s["lora"].items():
        named[n].data.copy_(v.to(DEV))
    opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
    step, hist = s["step"], s["hist"]
    print("resumed from step", step)

def get_batch():
    cls = np.random.choice(tail_cls, BS)
    idx = np.array([np.random.choice(by_cls[c]) for c in cls])
    x = torch.from_numpy(Xt[idx]).to(DEV).permute(0, 3, 1, 2).float() / 127.5 - 1
    m = torch.from_numpy(Mt[idx]).to(DEV)[:, None].float()
    x = F.interpolate(x, size=RES, mode="bilinear", align_corners=False)
    m = F.interpolate(m, size=RES, mode="nearest")
    if random.random() < 0.5:
        x, m = x.flip(3), m.flip(3)
    k = random.randint(0, 3)
    x, m = torch.rot90(x, k, (2, 3)), torch.rot90(m, k, (2, 3))
    kk = random.choice([1, 5, 11, 17])
    m = F.max_pool2d(m, kk, 1, kk // 2)
    full = torch.rand(BS, device=DEV) < 0.1
    m = torch.where(full[:, None, None, None], torch.ones_like(m), m)
    return x, m, cls

@torch.no_grad()
def generate(x_src, mask, cls_ids, steps=25, guidance=5.0, seed=0):
    """x_src in [-1,1] (B,3,RES,RES), mask 1 = repaint. Returns images in [-1,1], real pixels kept outside the mask."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    sch = DDIMScheduler.from_config(noise_sched.config)
    sch.set_timesteps(steps, device=DEV)
    ml = vae.encode((x_src * (1 - mask)).half()).latent_dist.mode() * SF
    mm = F.interpolate(mask, size=ml.shape[-2:], mode="nearest").half()
    cond = torch.cat([EMB[c] for c in cls_ids])
    emb = torch.cat([UNCOND.expand(len(cond), -1, -1), cond])
    lat = torch.randn(ml.shape, generator=g, device=DEV, dtype=torch.float16)
    for t in sch.timesteps:
        inp = torch.cat([lat, mm, ml], 1)
        inp = torch.cat([inp, inp]).half()
        with torch.autocast("cuda", dtype=torch.float16):
            e = unet(inp, t, encoder_hidden_states=emb).sample
        eu, ec = e.chunk(2)
        lat = sch.step((eu + guidance * (ec - eu)).half(), t, lat).prev_sample
    out = vae.decode(lat / SF).sample.float().clamp(-1, 1)
    return x_src * (1 - mask) + out * mask

nv = np.where(Ytr == CLASSES.index("NV"))[0]
def preview(st):
    rng = np.random.RandomState(st)
    fig, ax = plt.subplots(len(tail_cls), 7, figsize=(14, 2 * len(tail_cls)))
    for r, c in enumerate(tail_cls):
        si = np.sort(rng.choice(nv, 4, replace=False)); mi = rng.choice(by_cls[c], 4, replace=False)
        x = torch.from_numpy(np.stack([Xtr[i] for i in si])).to(DEV).permute(0, 3, 1, 2).float() / 127.5 - 1
        x = F.interpolate(x, size=RES, mode="bilinear", align_corners=False)
        m = F.interpolate(torch.from_numpy(Mt[mi]).to(DEV)[:, None].float(), size=RES, mode="nearest")
        gen = generate(x, m, [c] * 4, seed=st)
        gen = ((gen + 1) * 127.5).permute(0, 2, 3, 1).byte().cpu().numpy()
        real = Xt[rng.choice(by_cls[c], 3, replace=False)]
        for j in range(7):
            ax[r, j].imshow(gen[j] if j < 4 else real[j - 4]); ax[r, j].axis("off")
            ax[r, j].set_title(f"{CLASSES[c]} generated" if j < 4 else f"{CLASSES[c]} real", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{REPORT}/gen_preview_{st}.png", dpi=90); plt.show()

def save_ck():
    lora = {n: p.detach().cpu() for n, p in unet.named_parameters() if p.requires_grad}
    torch.save({"lora": lora, "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step, "hist": hist}, GEN_CK)

t0, run_loss = time.time(), []
unet.train()
while step < STEPS:
    x, m, cls = get_batch()
    with torch.no_grad():
        lat = vae.encode(x.half()).latent_dist.sample().float() * SF
        ml = vae.encode((x * (1 - m)).half()).latent_dist.sample().float() * SF
        mm = F.interpolate(m, size=lat.shape[-2:], mode="nearest")
        cond = torch.cat([EMB[c] for c in cls])
        drop = torch.rand(BS, device=DEV) < 0.1
        cond = torch.where(drop[:, None, None], UNCOND.expand(BS, -1, -1), cond)
    noise = torch.randn_like(lat)
    t = torch.randint(0, noise_sched.config.num_train_timesteps, (BS,), device=DEV)
    noisy = noise_sched.add_noise(lat, noise, t)
    with torch.autocast("cuda", dtype=torch.float16):
        pred = unet(torch.cat([noisy, mm, ml], 1).half(), t, encoder_hidden_states=cond).sample
    loss = F.mse_loss(pred.float(), noise)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt); scaler.update()
    step += 1; run_loss.append(loss.item())
    if step % 50 == 0:
        hist.append((step, float(np.mean(run_loss)))); run_loss = []
        print(f"step {step}/{STEPS} | loss {hist[-1][1]:.4f} | {(time.time() - t0) / 50:.2f}s/step"); t0 = time.time()
    if step % SAVE_EVERY == 0:
        save_ck()
    if step % PREVIEW_EVERY == 0 or step == STEPS:
        unet.eval(); preview(step); unet.train()

save_ck()
h = np.array(hist)
plt.figure(figsize=(6, 3)); plt.plot(h[:, 0], h[:, 1]); plt.title("generator training loss"); plt.xlabel("step")
plt.tight_layout(); plt.savefig(f"{REPORT}/generator_loss.png", dpi=100); plt.show()
print("GENERATOR DONE | LoRA saved in", GEN_CK)

# %%
import os, gc, time, json, random
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, AutoModel
from peft import LoraConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score

DEV = "cuda:0"
SD_REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"
RES, N_PILOT, BATCH = 256, 150, 16
DOSES = [0, 25, 50, 100, 150]
BUDGET, SEEDS = 3000, [0, 1, 2]
CK = f"{WORK}/ckpt"
NAMES = {"AK": "actinic keratosis", "DF": "dermatofibroma", "VASC": "vascular lesion", "SCC": "squamous cell carcinoma"}
PILOT = f"{WORK}/pilot.npz"
t_start = time.time()

tail_cls = [CLASSES.index(c) for c in TAIL]
z = np.load(f"{WORK}/tail_masks.npz")
m_idx, Mt = z["idx"], z["masks"]
Yt = Ytr[m_idx]
by_cls = {c: np.where(Yt == c)[0] for c in tail_cls}
bg_pool = np.where(np.isin(Ytr, [CLASSES.index(c) for c in ("NV", "BKL", "MEL", "BCC")]))[0]

# ---------- 1) pilot generation ----------
if os.path.exists(PILOT):
    zz = np.load(PILOT); P_img, P_lab = zz["img"], zz["lab"]
    print("loaded pilot images:", len(P_img))
else:
    tok = CLIPTokenizer.from_pretrained(SD_REPO, subfolder="tokenizer")
    txt = CLIPTextModel.from_pretrained(SD_REPO, subfolder="text_encoder").half().to(DEV).eval()
    @torch.no_grad()
    def encode_text(p):
        ids = tok(p, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(DEV)
        return txt(ids).last_hidden_state
    EMB = {c: encode_text([f"a dermoscopy image of {NAMES[CLASSES[c]]}"]) for c in tail_cls}
    UNCOND = encode_text([""])
    del txt; gc.collect(); torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained(SD_REPO, subfolder="vae").half().to(DEV).eval()
    SF = vae.config.scaling_factor
    unet = UNet2DConditionModel.from_pretrained(SD_REPO, subfolder="unet").half().to(DEV)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=16, lora_alpha=16, init_lora_weights="gaussian",
                                target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    named = dict(unet.named_parameters())
    s = torch.load(f"{CK}/gen_last.pt", map_location=DEV, weights_only=False)
    for n, v in s["lora"].items():
        named[n].data = v.to(DEV).float()
    print("generator loaded, trained steps:", s["step"])
    unet.eval()
    noise_cfg = DDPMScheduler.from_pretrained(SD_REPO, subfolder="scheduler").config

    @torch.no_grad()
    def generate(x_src, mask, cls_ids, steps=25, guidance=5.0, seed=0):
        g = torch.Generator(device=DEV).manual_seed(seed)
        sch = DDIMScheduler.from_config(noise_cfg)
        sch.set_timesteps(steps, device=DEV)
        ml = vae.encode((x_src * (1 - mask)).half()).latent_dist.mode() * SF
        mm = F.interpolate(mask, size=ml.shape[-2:], mode="nearest").half()
        cond = torch.cat([EMB[c] for c in cls_ids])
        emb = torch.cat([UNCOND.expand(len(cond), -1, -1), cond])
        lat = torch.randn(ml.shape, generator=g, device=DEV, dtype=torch.float16)
        for t in sch.timesteps:
            inp = torch.cat([lat, mm, ml], 1)
            inp = torch.cat([inp, inp]).half()
            with torch.autocast("cuda", dtype=torch.float16):
                e = unet(inp, t, encoder_hidden_states=emb).sample
            eu, ec = e.chunk(2)
            lat = sch.step((eu + guidance * (ec - eu)).half(), t, lat).prev_sample
        out = vae.decode(lat / SF).sample.float().clamp(-1, 1)
        return x_src * (1 - mask) + out * mask

    imgs, labs = [], []
    for c in tail_cls:
        rng = np.random.RandomState(100 + c)
        for b0 in range(0, N_PILOT, BATCH):
            n = min(BATCH, N_PILOT - b0)
            bi = np.sort(rng.choice(bg_pool, n, replace=False)); mi = rng.choice(by_cls[c], n)
            x = torch.from_numpy(np.stack([Xtr[i] for i in bi])).to(DEV).permute(0, 3, 1, 2).float() / 127.5 - 1
            x = F.interpolate(x, size=RES, mode="bilinear", align_corners=False)
            m = F.interpolate(torch.from_numpy(Mt[mi]).to(DEV)[:, None].float(), size=RES, mode="nearest")
            if rng.rand() < 0.5: x, m = x.flip(3), m.flip(3)
            k = int(rng.randint(0, 4)); x, m = torch.rot90(x, k, (2, 3)), torch.rot90(m, k, (2, 3))
            kk = int(rng.choice([1, 5, 11])); m = F.max_pool2d(m, kk, 1, kk // 2)
            g = generate(x, m, [c] * n, seed=int(rng.randint(1_000_000)))
            g = F.interpolate(g, size=224, mode="bilinear", antialias=True, align_corners=False)
            imgs.append(((g + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy())
            labs.append(np.full(n, c))
        print(f"pilot {CLASSES[c]} done | {time.time() - t_start:.0f}s")
    P_img, P_lab = np.concatenate(imgs), np.concatenate(labs)
    np.savez_compressed(PILOT, img=P_img, lab=P_lab)
    del unet, vae; gc.collect(); torch.cuda.empty_cache()

# pilot grid
fig, ax = plt.subplots(len(tail_cls), 8, figsize=(16, 2 * len(tail_cls)))
for r, c in enumerate(tail_cls):
    gi = np.where(P_lab == c)[0][:8]
    for j in range(8):
        ax[r, j].imshow(P_img[gi[j]]); ax[r, j].axis("off"); ax[r, j].set_title(f"{CLASSES[c]} synthetic", fontsize=8)
plt.tight_layout(); plt.savefig(f"{REPORT}/pilot_grid.png", dpi=90); plt.show()

# ---------- 2) DINOv2 features of the pilot ----------
zf = np.load(f"{WORK}/dino_feats.npz"); Ftr, Fva = zf["tr"], zf["va"]
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()
@torch.no_grad()
def embed(arr):
    out = []
    for i in range(0, len(arr), 128):
        x = torch.from_numpy(np.ascontiguousarray(arr[i:i + 128])).to(DEV).permute(0, 3, 1, 2).float() / 255
        with torch.autocast("cuda", dtype=torch.float16):
            f = dino(pixel_values=(x - MEAN) / STD).pooler_output.float()
        out.append(F.normalize(f, dim=1).cpu())
    return torch.cat(out).numpy()
Fp = embed(P_img)
del dino; gc.collect(); torch.cuda.empty_cache()

# ---------- 3) probe ----------
def head_subset(seed):
    rng = np.random.RandomState(seed)
    return np.concatenate([rng.choice(np.where(Ytr == ci)[0], min(int((Ytr == ci).sum()), 1500), replace=False) for ci in range(K)])

def fit(seed, extra_F, extra_y):
    keep = head_subset(seed)
    Xf = np.concatenate([Ftr[keep]] + extra_F); yf = np.concatenate([Ytr[keep]] + extra_y)
    clf = LogisticRegression(C=10, max_iter=300, class_weight="balanced").fit(Xf, yf)
    return clf, clf.predict(Fva)

def score(p):
    rec = recall_score(Yva, p, labels=list(range(K)), average=None, zero_division=0)
    return balanced_accuracy_score(Yva, p), rec

rows, base_clf = [], None
for sd in SEEDS:
    clf0, p0 = fit(sd, [], [])
    if sd == SEEDS[0]: base_clf = clf0
    b0, r0 = score(p0)
    for c in tail_cls:
        pool = np.where(P_lab == c)[0]
        for d in DOSES:
            if d == 0:
                bb, rr = b0, r0
            else:
                pick = np.random.RandomState(1000 * sd + c + d).choice(pool, d, replace=False)
                _, p = fit(sd, [Fp[pick]], [P_lab[pick]])
                bb, rr = score(p)
            rows.append({"seed": sd, "class": CLASSES[c], "dose": d, "d_bacc": bb - b0, "d_recall_c": rr[c] - r0[c],
                         "d_head_recall": float(np.mean(np.delete(rr - r0, tail_cls)))})
    print(f"probe seed {sd} done | {time.time() - t_start:.0f}s")
dr = pd.DataFrame(rows)
dr.to_csv(f"{WORK}/dose_response.csv", index=False)
agg = dr.groupby(["class", "dose"])[["d_bacc", "d_recall_c", "d_head_recall"]].agg(["mean", "std"]).round(4)
print("\nDose-response (change vs no synthetic images)\n", agg.to_string())

# fidelity check
pp = base_clf.predict(Fp)
print("\nFidelity: probe trained on real data only")
for c in tail_cls:
    fid = float((pp[P_lab == c] == c).mean()); vr = float((base_clf.predict(Fva[Yva == c]) == c).mean())
    print(f"  {CLASSES[c]}: synthetic recognised as {CLASSES[c]} {fid:.2f} | real val images {vr:.2f}")

# ---------- 4) dose rule ----------
st = pd.read_csv(f"{WORK}/class_stats.csv").set_index("class")
gain = dr[dr["dose"] == max(DOSES)].groupby("class")["d_bacc"].mean().reindex(TAIL).clip(lower=0)
diff = st.loc[TAIL, "difficulty"]
gn = gain / gain.sum() if gain.sum() > 0 else diff / diff.sum()
w = np.sqrt(gn * (diff / diff.sum()))
share = (w / w.sum()).clip(0.10, 0.45); share = share / share.sum()
table = pd.DataFrame({"measured_gain": gain, "difficulty": diff, "share": share,
                      "dose_aware": (BUDGET * share).round().astype(int), "dose_equal": BUDGET // len(TAIL)})
table.to_csv(f"{WORK}/dose_table.csv")
print(f"\nDose table for a budget of {BUDGET} images\n", table.round(3).to_string())

fig, ax = plt.subplots(1, 3, figsize=(17, 4))
for c in TAIL:
    s_ = dr[dr["class"] == CLASSES[c]].groupby("dose")
    for a, col in zip(ax[:2], ["d_bacc", "d_recall_c"]):
        a.errorbar(DOSES, s_[col].mean(), yerr=s_[col].std(), marker="o", capsize=3, label=CLASSES[c])
ax[0].set_title("change in balanced accuracy vs dose"); ax[1].set_title("change in own-class recall vs dose")
for a in ax[:2]: a.axhline(0, color="gray", lw=0.8); a.set_xlabel("synthetic images added"); a.legend()
x = np.arange(len(TAIL)); ax[2].bar(x - 0.2, table["dose_equal"], 0.4, label="equal"); ax[2].bar(x + 0.2, table["dose_aware"], 0.4, label="gain + difficulty")
ax[2].set_xticks(x); ax[2].set_xticklabels(TAIL); ax[2].set_title(f"proposed dose ({BUDGET} images)"); ax[2].legend()
plt.tight_layout(); plt.savefig(f"{REPORT}/dose_rule.png", dpi=110); plt.show()
print(f"CELL 6 DONE | {time.time() - t_start:.0f}s")

# %%
import os, gc, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, AutoModel
from peft import LoraConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score

DEV = "cuda:0"
SD_REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"
RES, N, BATCH, SEEDS = 256, 100, 16, [0, 1, 2]
CK = f"{WORK}/ckpt"
NAMES = {"AK": "actinic keratosis", "DF": "dermatofibroma", "VASC": "vascular lesion", "SCC": "squamous cell carcinoma"}
MODES = {"own_lesion_repaint": 1.0, "sdedit_0.5": 0.5, "sdedit_0.75": 0.75}
OUT = f"{WORK}/pilot2.npz"
t_start = time.time()

tail_cls = [CLASSES.index(c) for c in TAIL]
z = np.load(f"{WORK}/tail_masks.npz")
m_idx, Mt = z["idx"], z["masks"]
Yt = Ytr[m_idx]
by_cls = {c: np.where(Yt == c)[0] for c in tail_cls}
picks = {c: np.random.RandomState(200 + c).choice(by_cls[c], N, replace=False) for c in tail_cls}

if os.path.exists(OUT):
    zz = np.load(OUT, allow_pickle=True)
    GEN = {m: (zz[m + "_img"], zz[m + "_lab"]) for m in MODES}
    print("loaded generated sets")
else:
    tok = CLIPTokenizer.from_pretrained(SD_REPO, subfolder="tokenizer")
    txt = CLIPTextModel.from_pretrained(SD_REPO, subfolder="text_encoder").half().to(DEV).eval()
    @torch.no_grad()
    def encode_text(p):
        ids = tok(p, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(DEV)
        return txt(ids).last_hidden_state
    EMB = {c: encode_text([f"a dermoscopy image of {NAMES[CLASSES[c]]}"]) for c in tail_cls}
    UNCOND = encode_text([""])
    del txt; gc.collect(); torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained(SD_REPO, subfolder="vae").half().to(DEV).eval()
    SF = vae.config.scaling_factor
    unet = UNet2DConditionModel.from_pretrained(SD_REPO, subfolder="unet").half().to(DEV)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=16, lora_alpha=16, init_lora_weights="gaussian",
                                target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    named = dict(unet.named_parameters())
    s = torch.load(f"{CK}/gen_last.pt", map_location=DEV, weights_only=False)
    for n, v in s["lora"].items():
        named[n].data = v.to(DEV).float()
    unet.eval()
    noise_cfg = DDPMScheduler.from_pretrained(SD_REPO, subfolder="scheduler").config

    @torch.no_grad()
    def generate(x_src, mask, cls_ids, strength=1.0, steps=25, guidance=5.0, seed=0):
        g = torch.Generator(device=DEV).manual_seed(seed)
        sch = DDIMScheduler.from_config(noise_cfg)
        sch.set_timesteps(steps, device=DEV)
        ts = sch.timesteps[int(round(steps * (1 - strength))):]
        ml = vae.encode((x_src * (1 - mask)).half()).latent_dist.mode() * SF
        mm = F.interpolate(mask, size=ml.shape[-2:], mode="nearest").half()
        cond = torch.cat([EMB[c] for c in cls_ids])
        emb = torch.cat([UNCOND.expand(len(cond), -1, -1), cond])
        noise = torch.randn(ml.shape, generator=g, device=DEV, dtype=torch.float16)
        if strength < 1.0:
            z0 = vae.encode(x_src.half()).latent_dist.mode() * SF
            lat = sch.add_noise(z0, noise, ts[:1].repeat(len(z0)))
        else:
            lat = noise
        for t in ts:
            inp = torch.cat([lat, mm, ml], 1)
            inp = torch.cat([inp, inp]).half()
            with torch.autocast("cuda", dtype=torch.float16):
                e = unet(inp, t, encoder_hidden_states=emb).sample
            eu, ec = e.chunk(2)
            lat = sch.step((eu + guidance * (ec - eu)).half(), t, lat).prev_sample
        out = vae.decode(lat / SF).sample.float().clamp(-1, 1)
        return x_src * (1 - mask) + out * mask

    GEN = {}
    for mode, strength in MODES.items():
        imgs, labs = [], []
        for c in tail_cls:
            for b0 in range(0, N, BATCH):
                ii = picks[c][b0:b0 + BATCH]; n = len(ii)
                x = torch.from_numpy(np.stack([Xtr[m_idx[i]] for i in ii])).to(DEV).permute(0, 3, 1, 2).float() / 127.5 - 1
                x = F.interpolate(x, size=RES, mode="bilinear", align_corners=False)
                if mode == "own_lesion_repaint":
                    m = F.interpolate(torch.from_numpy(Mt[ii]).to(DEV)[:, None].float(), size=RES, mode="nearest")
                    m = F.max_pool2d(m, 11, 1, 5)
                else:
                    m = torch.ones(n, 1, RES, RES, device=DEV)
                g = generate(x, m, [c] * n, strength=strength, seed=1000 * c + b0)
                g = F.interpolate(g, size=224, mode="bilinear", antialias=True, align_corners=False)
                imgs.append(((g + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy())
                labs.append(np.full(n, c))
        GEN[mode] = (np.concatenate(imgs), np.concatenate(labs))
        print(f"{mode} done | {time.time() - t_start:.0f}s")
    np.savez_compressed(OUT, **{k + s_: v[i] for k, v in GEN.items() for i, s_ in enumerate(["_img", "_lab"])})
    del unet, vae; gc.collect(); torch.cuda.empty_cache()

# grid: 3 images per mode per class
fig, ax = plt.subplots(len(tail_cls), 3 * len(MODES), figsize=(3 * len(MODES) * 1.8, 2 * len(tail_cls)))
for r, c in enumerate(tail_cls):
    for mi, mode in enumerate(MODES):
        im, lb = GEN[mode]
        for j in range(3):
            a = ax[r, mi * 3 + j]; a.imshow(im[np.where(lb == c)[0][j]]); a.axis("off")
            a.set_title(f"{CLASSES[c]} {mode[:12]}", fontsize=6)
plt.tight_layout(); plt.savefig(f"{REPORT}/modes_grid.png", dpi=90); plt.show()

# features
zf = np.load(f"{WORK}/dino_feats.npz"); Ftr, Fva = zf["tr"], zf["va"]
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()
@torch.no_grad()
def embed(arr):
    out = []
    for i in range(0, len(arr), 128):
        x = torch.from_numpy(np.ascontiguousarray(arr[i:i + 128])).to(DEV).permute(0, 3, 1, 2).float() / 255
        with torch.autocast("cuda", dtype=torch.float16):
            f = dino(pixel_values=(x - MEAN) / STD).pooler_output.float()
        out.append(F.normalize(f, dim=1).cpu())
    return torch.cat(out).numpy()

SETS = {}
zp = np.load(f"{WORK}/pilot.npz"); PI, PL = zp["img"], zp["lab"]
keep = np.concatenate([np.where(PL == c)[0][:N] for c in tail_cls])
SETS["A_old_pilot"] = (embed(PI[keep]), PL[keep])
for mode in MODES:
    SETS[mode] = (embed(GEN[mode][0]), GEN[mode][1])
dup_i = np.concatenate([m_idx[picks[c]] for c in tail_cls])
SETS["control_duplicate_real"] = (Ftr[dup_i], Ytr[dup_i])
del dino; gc.collect(); torch.cuda.empty_cache()

def head_subset(seed):
    rng = np.random.RandomState(seed)
    return np.concatenate([rng.choice(np.where(Ytr == ci)[0], min(int((Ytr == ci).sum()), 1500), replace=False) for ci in range(K)])

def fit(seed, Fx=None, yx=None):
    keep = head_subset(seed)
    Xf, yf = Ftr[keep], Ytr[keep]
    if Fx is not None:
        Xf, yf = np.concatenate([Xf, Fx]), np.concatenate([yf, yx])
    clf = LogisticRegression(C=10, max_iter=300, class_weight="balanced").fit(Xf, yf)
    return clf, clf.predict(Fva)

def score(p):
    return balanced_accuracy_score(Yva, p), recall_score(Yva, p, labels=list(range(K)), average=None, zero_division=0)

rows, base_clf = [], None
for sd in SEEDS:
    clf0, p0 = fit(sd); b0, r0 = score(p0)
    if sd == SEEDS[0]: base_clf = clf0
    for name, (Fx, yx) in SETS.items():
        _, p = fit(sd, Fx, yx); bb, rr = score(p)
        row = {"set": name, "seed": sd, "d_bacc": bb - b0, "d_head_recall": float(np.mean(np.delete(rr - r0, tail_cls)))}
        row.update({f"d_recall_{CLASSES[c]}": rr[c] - r0[c] for c in tail_cls})
        rows.append(row)
    print(f"probe seed {sd} done | {time.time() - t_start:.0f}s")
res = pd.DataFrame(rows)
res.to_csv(f"{WORK}/modes_probe.csv", index=False)
print("\nMean change vs no synthetic (100 images per tail class)")
print(res.groupby("set").mean(numeric_only=True).drop(columns="seed").round(4).to_string())
print("\nStd across seeds")
print(res.groupby("set").std(numeric_only=True).drop(columns="seed").round(4).to_string())

print("\nFidelity (probe trained on real data only)  | real val reference:",
      {CLASSES[c]: round(float((base_clf.predict(Fva[Yva == c]) == c).mean()), 2) for c in tail_cls})
for name, (Fx, yx) in SETS.items():
    pp = base_clf.predict(Fx)
    print(f"  {name:24s}", {CLASSES[c]: round(float((pp[yx == c] == c).mean()), 2) for c in tail_cls})
print(f"CELL 6b DONE | {time.time() - t_start:.0f}s")

# %%
import os, gc, time, glob
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, AutoModel
from peft import LoraConfig
from sklearn.linear_model import LogisticRegression

DEV = "cuda:0"
SD_REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"
RES, BATCH, CHUNK = 256, 16, 80
N_REPAINT, N_SDEDIT = 540, 260
CK, POOL = f"{WORK}/ckpt", f"{WORK}/pool"
os.makedirs(POOL, exist_ok=True)
NAMES = {"AK": "actinic keratosis", "DF": "dermatofibroma", "VASC": "vascular lesion", "SCC": "squamous cell carcinoma"}
t_start = time.time()

tail_cls = [CLASSES.index(c) for c in TAIL]
z = np.load(f"{WORK}/tail_masks.npz")
m_idx, Mt = z["idx"], z["masks"]
Yt = Ytr[m_idx]
by_cls = {c: np.where(Yt == c)[0] for c in tail_cls}

# job list: (class, mode, chunk number, size)
jobs = []
for c in tail_cls:
    for mode, n in (("repaint", N_REPAINT), ("sdedit", N_SDEDIT)):
        for ci, b0 in enumerate(range(0, n, CHUNK)):
            jobs.append((c, mode, ci, min(CHUNK, n - b0)))
fname = lambda c, mode, ci: f"{POOL}/{CLASSES[c]}_{mode}_{ci}.npz"
todo = [j for j in jobs if not os.path.exists(fname(*j[:3]))]
print(f"{len(jobs) - len(todo)}/{len(jobs)} chunks already done")

if todo:
    tok = CLIPTokenizer.from_pretrained(SD_REPO, subfolder="tokenizer")
    txt = CLIPTextModel.from_pretrained(SD_REPO, subfolder="text_encoder").half().to(DEV).eval()
    @torch.no_grad()
    def encode_text(p):
        ids = tok(p, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(DEV)
        return txt(ids).last_hidden_state
    EMB = {c: encode_text([f"a dermoscopy image of {NAMES[CLASSES[c]]}"]) for c in tail_cls}
    UNCOND = encode_text([""])
    del txt; gc.collect(); torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained(SD_REPO, subfolder="vae").half().to(DEV).eval()
    SF = vae.config.scaling_factor
    unet = UNet2DConditionModel.from_pretrained(SD_REPO, subfolder="unet").half().to(DEV)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=16, lora_alpha=16, init_lora_weights="gaussian",
                                target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    named = dict(unet.named_parameters())
    s = torch.load(f"{CK}/gen_last.pt", map_location=DEV, weights_only=False)
    for n, v in s["lora"].items():
        named[n].data = v.to(DEV).float()
    unet.eval()
    noise_cfg = DDPMScheduler.from_pretrained(SD_REPO, subfolder="scheduler").config

    @torch.no_grad()
    def generate(x_src, mask, cls_ids, strength=1.0, steps=25, guidance=5.0, seed=0):
        g = torch.Generator(device=DEV).manual_seed(seed)
        sch = DDIMScheduler.from_config(noise_cfg)
        sch.set_timesteps(steps, device=DEV)
        ts = sch.timesteps[int(round(steps * (1 - strength))):]
        ml = vae.encode((x_src * (1 - mask)).half()).latent_dist.mode() * SF
        mm = F.interpolate(mask, size=ml.shape[-2:], mode="nearest").half()
        cond = torch.cat([EMB[c] for c in cls_ids])
        emb = torch.cat([UNCOND.expand(len(cond), -1, -1), cond])
        noise = torch.randn(ml.shape, generator=g, device=DEV, dtype=torch.float16)
        if strength < 1.0:
            z0 = vae.encode(x_src.half()).latent_dist.mode() * SF
            lat = sch.add_noise(z0, noise, ts[:1].repeat(len(z0)))
        else:
            lat = noise
        for t in ts:
            inp = torch.cat([lat, mm, ml], 1)
            inp = torch.cat([inp, inp]).half()
            with torch.autocast("cuda", dtype=torch.float16):
                e = unet(inp, t, encoder_hidden_states=emb).sample
            eu, ec = e.chunk(2)
            lat = sch.step((eu + guidance * (ec - eu)).half(), t, lat).prev_sample
        out = vae.decode(lat / SF).sample.float().clamp(-1, 1)
        return x_src * (1 - mask) + out * mask

    for c, mode, ci, size in todo:
        t0 = time.time()
        rng = np.random.RandomState(7000 + 100 * c + 10 * (mode == "sdedit") + ci)
        order = np.random.RandomState(500 + c + 10 * (mode == "sdedit")).permutation(by_cls[c])
        n_mode = N_REPAINT if mode == "repaint" else N_SDEDIT
        srcs_all = np.resize(order, n_mode)
        srcs = srcs_all[ci * CHUNK: ci * CHUNK + size]
        imgs = []
        for b0 in range(0, size, BATCH):
            ii = srcs[b0:b0 + BATCH]; n = len(ii)
            x = torch.from_numpy(np.stack([Xtr[m_idx[i]] for i in ii])).to(DEV).permute(0, 3, 1, 2).float() / 127.5 - 1
            x = F.interpolate(x, size=RES, mode="bilinear", align_corners=False)
            if mode == "repaint":
                m = F.interpolate(torch.from_numpy(Mt[ii]).to(DEV)[:, None].float(), size=RES, mode="nearest")
                kk = int(rng.choice([5, 11, 17])); m = F.max_pool2d(m, kk, 1, kk // 2)
                strength = 1.0
            else:
                m = torch.ones(n, 1, RES, RES, device=DEV); strength = 0.5
            if rng.rand() < 0.5: x, m = x.flip(3), m.flip(3)
            k = int(rng.randint(0, 4)); x, m = torch.rot90(x, k, (2, 3)), torch.rot90(m, k, (2, 3))
            g = generate(x, m, [c] * n, strength=strength, seed=int(rng.randint(1_000_000_000)))
            g = F.interpolate(g, size=224, mode="bilinear", antialias=True, align_corners=False)
            imgs.append(((g + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy())
        np.savez(fname(c, mode, ci), img=np.concatenate(imgs), src=m_idx[srcs], lab=np.full(size, c))
        print(f"{CLASSES[c]} {mode} chunk {ci} ({size} imgs) | {time.time() - t0:.0f}s | total {time.time() - t_start:.0f}s")
    del unet, vae; gc.collect(); torch.cuda.empty_cache()

# ---------- load the pool and score it ----------
imgs, labs, srcs, modes = [], [], [], []
for c in tail_cls:
    for mode in ("repaint", "sdedit"):
        for f in sorted(glob.glob(f"{POOL}/{CLASSES[c]}_{mode}_*.npz")):
            d = np.load(f)
            imgs.append(d["img"]); labs.append(d["lab"]); srcs.append(d["src"]); modes += [mode] * len(d["lab"])
P_img, P_lab, P_src, P_mode = np.concatenate(imgs), np.concatenate(labs), np.concatenate(srcs), np.array(modes)
print("pool size:", len(P_img))

zf = np.load(f"{WORK}/dino_feats.npz"); Ftr = zf["tr"]
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()
@torch.no_grad()
def embed(arr):
    out = []
    for i in range(0, len(arr), 128):
        x = torch.from_numpy(np.ascontiguousarray(arr[i:i + 128])).to(DEV).permute(0, 3, 1, 2).float() / 255
        with torch.autocast("cuda", dtype=torch.float16):
            f = dino(pixel_values=(x - MEAN) / STD).pooler_output.float()
        out.append(F.normalize(f, dim=1).cpu())
    return torch.cat(out).numpy()
Fp = embed(P_img)
del dino; gc.collect(); torch.cuda.empty_cache()
np.save(f"{WORK}/pool_feats.npy", Fp)

rng = np.random.RandomState(0)
keep = np.concatenate([rng.choice(np.where(Ytr == ci)[0], min(int((Ytr == ci).sum()), 1500), replace=False) for ci in range(K)])
clf = LogisticRegression(C=10, max_iter=300, class_weight="balanced").fit(Ftr[keep], Ytr[keep])
prob = clf.predict_proba(Fp)
p_target = prob[np.arange(len(P_lab)), P_lab]

copy_score = np.zeros(len(P_lab))      # similarity to the closest real image of the same class (1 = near copy)
srcsim = np.zeros(len(P_lab))          # similarity to the image it was made from
for c in tail_cls:
    mk = P_lab == c
    sim = Fp[mk] @ Ftr[Ytr == c].T
    copy_score[mk] = sim.max(1)
    srcsim[mk] = np.einsum("ij,ij->i", Fp[mk], Ftr[P_src[mk]])
sc = pd.DataFrame({"class": [CLASSES[c] for c in P_lab], "mode": P_mode, "src": P_src,
                   "p_target": p_target, "correct": prob.argmax(1) == P_lab, "copy_score": copy_score, "src_sim": srcsim})
sc.to_csv(f"{WORK}/pool_scores.csv", index=False)
print("\nPool summary (fidelity = share recognised as the right class; copy_score near 1 means near-duplicate)")
print(sc.groupby(["class", "mode"]).agg(n=("correct", "size"), fidelity=("correct", "mean"),
      p_target=("p_target", "mean"), copy_score=("copy_score", "mean"), src_sim=("src_sim", "mean")).round(3).to_string())

fig, ax = plt.subplots(len(tail_cls), 8, figsize=(16, 2 * len(tail_cls)))
for r, c in enumerate(tail_cls):
    for j in range(8):
        pool_c = np.where((P_lab == c) & (P_mode == ("repaint" if j < 4 else "sdedit")))[0]
        i = pool_c[np.random.RandomState(j).randint(len(pool_c))]
        ax[r, j].imshow(P_img[i]); ax[r, j].axis("off")
        ax[r, j].set_title(f"{CLASSES[c]} {P_mode[i]}", fontsize=7)
plt.tight_layout(); plt.savefig(f"{REPORT}/pool_grid.png", dpi=90); plt.show()
print(f"CELL 7 DONE | {time.time() - t_start:.0f}s")

# %%
import os, gc, time, json, glob
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

DEV = "cuda:0"
SEEDS = [0]
CONDS = ["baseline", "equal", "control_real_dup", "synth_small", "dup_small", "dose_aware", "dose_aware_filtered"]
EPOCHS, BS, LR, WD = 12, 64, 3e-4, 1e-4
EQUAL_N, BUDGET, FLOOR, CAP = 750, 3000, 300, 800
CK = f"{WORK}/ckpt8"
os.makedirs(CK, exist_ok=True)
POOL = f"{WORK}/pool"
tail_cls = [CLASSES.index(c) for c in TAIL]
head_cls = [i for i in range(K) if i not in tail_cls]

# ---------- data: real train + synthetic pool ----------
Xr = np.ascontiguousarray(Xtr); Xva_r = np.ascontiguousarray(Xva); Xte_r = np.ascontiguousarray(Xte)
imgs, labs, srcs = [], [], []
for c in tail_cls:
    for mode in ("repaint", "sdedit"):
        for f in sorted(glob.glob(f"{POOL}/{CLASSES[c]}_{mode}_*.npz")):
            d = np.load(f); imgs.append(d["img"]); labs.append(d["lab"]); srcs.append(d["src"])
P_img, P_lab, P_src = np.concatenate(imgs), np.concatenate(labs), np.concatenate(srcs)
sc = pd.read_csv(f"{WORK}/pool_scores.csv")
assert len(sc) == len(P_lab) and (sc["src"].values == P_src).all(), "pool order does not match pool_scores.csv"
p_target = sc["p_target"].values
n_real = len(Xr)
X_all = np.concatenate([Xr, P_img]); Y_all = np.concatenate([Ytr, P_lab])
del Xr; gc.collect()
print("real train", n_real, "| pool", len(P_img))

# ---------- dose rule: difficulty x fidelity ----------
st = pd.read_csv(f"{WORK}/class_stats.csv").set_index("class")
fid = sc.groupby("class")["correct"].mean()
w = pd.Series({c: st.loc[c, "difficulty"] * fid[c] for c in TAIL}); share = w / w.sum()
dose_aware = {c: int(np.clip(round(BUDGET * share[c]), FLOOR, CAP)) for c in TAIL}
print("fidelity:", fid.round(3).to_dict())
print("dose_aware:", dose_aware, "| total", sum(dose_aware.values()), "| equal:", EQUAL_N * len(TAIL))

def build_index(cond, seed):
    base = np.arange(n_real)
    if cond == "baseline":
        return base
    rng = np.random.RandomState(seed)
    extra = []
    for c in TAIL:
        ci = CLASSES.index(c); pc = np.where(P_lab == ci)[0]
        if cond in ("equal", "control_real_dup"):
            sel = rng.choice(pc, EQUAL_N, replace=False)
        elif cond == "dose_aware":
            sel = rng.choice(pc, dose_aware[c], replace=False)
        else:
            sel = pc[np.argsort(-p_target[pc])[:dose_aware[c]]]
        extra.append(P_src[sel] if cond == "control_real_dup" else n_real + sel)
    return np.concatenate([base] + extra)

# ---------- training (same recipe as the baseline cell) ----------
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
def to_gpu(xb): return torch.from_numpy(xb).to(DEV, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0

def augment(x):
    n = x.shape[0]
    ang = (torch.rand(n, device=DEV) * 2 - 1) * np.pi
    sc_ = 0.85 + 0.3 * torch.rand(n, device=DEV)
    flip = (torch.rand(n, device=DEV) < 0.5).float() * 2 - 1
    tx, ty = [(torch.rand(n, device=DEV) * 2 - 1) * 0.1 for _ in range(2)]
    cos, sin = torch.cos(ang) / sc_, torch.sin(ang) / sc_
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="reflection", align_corners=False)
    b = 1 + (torch.rand(n, 1, 1, 1, device=DEV) * 2 - 1) * 0.15
    c = 1 + (torch.rand(n, 1, 1, 1, device=DEV) * 2 - 1) * 0.15
    m = x.mean((1, 2, 3), keepdim=True)
    return ((x - m) * c + m * b).clamp(0, 1)

def make_model():
    m = torchvision.models.efficientnet_b0(weights="DEFAULT")
    m.classifier[1] = nn.Linear(1280, K)
    return m.to(DEV)

@torch.no_grad()
def predict(model, X):
    model.eval(); out = []
    for i in range(0, len(X), 128):
        x = (to_gpu(X[i:i + 128]) - MEAN) / STD
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.softmax(model(x).float(), 1).cpu())
    return torch.cat(out).numpy()

def metrics(probs, y):
    p = probs.argmax(1)
    rec = recall_score(y, p, labels=list(range(K)), average=None, zero_division=0)
    return {"bacc": float(balanced_accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro")),
            "acc": float((p == y).mean()), "recall": rec.tolist(),
            "tail_recall": float(rec[tail_cls].mean()), "head_recall": float(rec[head_cls].mean())}

def run(cond, seed):
    name = f"{cond}_s{seed}"; done_f = f"{CK}/{name}_done.json"
    if os.path.exists(done_f):
        print(name, "already finished"); return json.load(open(done_f))
    torch.manual_seed(seed); np.random.seed(seed)
    ids = build_index(cond, seed); yi = Y_all[ids]
    counts = np.bincount(Ytr, minlength=K)   # original counts, so synthetic images don't change the loss weights
    wts = (1 - 0.999) / (1 - 0.999 ** counts); wts = torch.tensor(wts / wts.sum() * K, dtype=torch.float32, device=DEV)
    print(f"{name}: {len(ids)} training images | counts {dict(zip(CLASSES, counts.tolist()))}")
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps_per = len(ids) // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * steps_per, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda"); crit = nn.CrossEntropyLoss(weight=wts)
    ep0, hist, best = 0, [], -1.0
    last = f"{CK}/{name}_last.pt"
    if os.path.exists(last):
        s = torch.load(last, map_location=DEV, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"]); ep0, hist, best = s["epoch"], s["hist"], s["best"]
        print(f"  resumed from epoch {ep0}")
    for ep in range(ep0, EPOCHS):
        t0 = time.time(); model.train(); perm = np.random.permutation(len(ids)); tot = 0.0
        for b in range(steps_per):
            bi = np.sort(ids[perm[b * BS:(b + 1) * BS]])
            x = (augment(to_gpu(X_all[bi])) - MEAN) / STD
            y = torch.from_numpy(Y_all[bi]).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        mv = metrics(predict(model, Xva_r), Yva)
        hist.append({"epoch": ep + 1, "loss": tot / steps_per, "val_bacc": mv["bacc"], "val_tail_recall": mv["tail_recall"]})
        if mv["bacc"] > best:
            best = mv["bacc"]; torch.save(model.state_dict(), f"{CK}/{name}_best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep + 1, "hist": hist, "best": best}, last)
        print(f"  ep {ep + 1}/{EPOCHS} loss {tot / steps_per:.3f} | val bacc {mv['bacc']:.3f} tail {mv['tail_recall']:.3f} | {time.time() - t0:.0f}s")
    model.load_state_dict(torch.load(f"{CK}/{name}_best.pt", map_location=DEV))
    res = {"cond": cond, "seed": seed, "n_train": int(len(ids)), "hist": hist,
           "val": metrics(predict(model, Xva_r), Yva), "test": metrics(predict(model, Xte_r), Yte)}
    json.dump(res, open(done_f, "w"))
    for f in (last, f"{CK}/{name}_best.pt"):
        if os.path.exists(f): os.remove(f)
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return res

t_all = time.time()
for sd in SEEDS:
    for cond in CONDS:
        run(cond, sd)
        print(f"--- elapsed {(time.time() - t_all) / 60:.0f} min")

# ---------- summary over every finished run ----------
R = [json.load(open(f)) for f in sorted(glob.glob(f"{CK}/*_done.json"))]
rows = [{"cond": r["cond"], "seed": r["seed"], "n_train": r["n_train"], "val_bacc": r["val"]["bacc"],
         "test_bacc": r["test"]["bacc"], "test_f1": r["test"]["macro_f1"], "test_acc": r["test"]["acc"],
         "tail_recall": r["test"]["tail_recall"], "head_recall": r["test"]["head_recall"],
         **{f"rec_{c}": r["test"]["recall"][i] for i, c in enumerate(CLASSES)}} for r in R]
df8 = pd.DataFrame(rows); df8.to_csv(f"{WORK}/comparison_runs.csv", index=False)
order = [c for c in CONDS if c in set(df8["cond"])]
cols = ["test_bacc", "test_f1", "test_acc", "tail_recall", "head_recall"]
print("\nTest metrics: mean over seeds (n =", df8.groupby("cond").size().to_dict(), ")")
print(df8.groupby("cond")[cols].mean().reindex(order).round(4).to_string())
print("\nStd over seeds")
print(df8.groupby("cond")[cols].std().reindex(order).round(4).to_string())
piv = df8.pivot(index="seed", columns="cond", values="test_bacc")
if "baseline" in piv:
    print("\nPaired change in test balanced accuracy vs baseline (per seed)")
    print(piv.sub(piv["baseline"], axis=0).drop(columns="baseline").round(4).to_string())
print("\nPer-class test recall (mean over seeds)")
print(df8.groupby("cond")[[f"rec_{c}" for c in CLASSES]].mean().reindex(order).round(3).to_string())

fig, ax = plt.subplots(1, 3, figsize=(18, 4))
for a, col in zip(ax[:2], ["test_bacc", "tail_recall"]):
    m = df8.groupby("cond")[col].mean().reindex(order)
    a.bar(range(len(order)), m.values, alpha=0.6)
    for i, c in enumerate(order):
        a.scatter([i] * int((df8.cond == c).sum()), df8[df8.cond == c][col], color="k", s=14, zorder=3)
    a.set_xticks(range(len(order))); a.set_xticklabels(order, rotation=25, ha="right"); a.set_title(col)
    a.set_ylim(max(0, m.min() - 0.05), m.max() + 0.04)
base_rec = df8[df8.cond == "baseline"][[f"rec_{c}" for c in CLASSES]].mean()
w_ = 0.8 / max(1, len(order) - 1)
for i, c in enumerate([o for o in order if o != "baseline"]):
    d = df8[df8.cond == c][[f"rec_{k}" for k in CLASSES]].mean() - base_rec
    ax[2].bar(np.arange(K) + i * w_ - 0.4, d.values, w_, label=c)
ax[2].axhline(0, color="gray", lw=0.8); ax[2].set_xticks(range(K)); ax[2].set_xticklabels(CLASSES)
ax[2].set_title("per-class test recall change vs baseline"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{REPORT}/comparison.png", dpi=110); plt.show()
print(f"CELL 8 DONE | {(time.time() - t_all) / 60:.0f} min")

# %%
import os
W = "/kaggle/working"
need = ["splits.csv", "class_stats.csv", "tail_masks.npz", "dino_feats.npz", "pool_scores.csv", "ckpt/gen_last.pt", "pool", "ckpt8"]
for n in need:
    print(("ok      " if os.path.exists(f"{W}/{n}") else "MISSING ") + n)
print("pool chunks:", len(os.listdir(f"{W}/pool")) if os.path.exists(f"{W}/pool") else 0, "(expected 40)")
print("cache files:", len(os.listdir(f"{W}/cache")) if os.path.exists(f"{W}/cache") else 0)

# %%
import os, gc, time, hashlib
import numpy as np, pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

INPUT, WORK = "/kaggle/input", "/kaggle/working"
SEED, S = 42, 224
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
TAIL = ["AK", "DF", "VASC", "SCC"]
K = len(CLASSES)
VAL_FRAC = {"tail": 0.20, "head": 0.08}
REBUILD_SPLITS = False
WORKERS = min(8, os.cpu_count() or 4)
CACHE, REPORT = f"{WORK}/cache", f"{WORK}/report"
for d in (CACHE, REPORT):
    os.makedirs(d, exist_ok=True)
np.random.seed(SEED)
t_start = time.time()

def load_resized(path, size):
    im = Image.open(path)
    im.draft("RGB", (2 * size, 2 * size))
    return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))

def find_csv(files, *must):
    hits = [f for f in files if all(m in os.path.basename(f).lower() for m in must)]
    assert hits, f"no csv with {must} in its name was found under {INPUT}"
    return hits[0]

def build_splits():
    images, csvs = {}, []
    for root, _, files in os.walk(INPUT):
        for f in files:
            low = f.lower()
            if low.endswith((".jpg", ".jpeg", ".png")):
                images[os.path.splitext(f)[0]] = os.path.join(root, f)
            elif low.endswith(".csv"):
                csvs.append(os.path.join(root, f))
    print("images found:", len(images), "| csv files:", [os.path.basename(c) for c in csvs])
    stripped = {k.replace("_downsampled", ""): v for k, v in images.items()}

    def prepare(gt):
        d = gt[gt[CLASSES].sum(axis=1) == 1].copy()
        d["label"] = d[CLASSES].values.argmax(1)
        d["class"] = d["label"].map(dict(enumerate(CLASSES)))
        d["path"] = d["image"].map(images)
        miss = d["path"].isna()
        d.loc[miss, "path"] = d.loc[miss, "image"].str.replace("_downsampled", "", regex=False).map(stripped)
        assert d["path"].notna().all(), f"{int(d['path'].isna().sum())} images listed in the csv have no file"
        return d[["image", "label", "class", "group", "path"]]

    gt_tr = pd.read_csv(find_csv(csvs, "train", "groundtruth"))
    mt_tr = pd.read_csv(find_csv(csvs, "train", "metadata"))
    gt_tr = gt_tr.merge(mt_tr[["image", "lesion_id"]], on="image", how="left")
    gt_tr["group"] = gt_tr["lesion_id"].fillna(gt_tr["image"])
    tr = prepare(gt_tr)
    gt_te = pd.read_csv(find_csv(csvs, "test", "groundtruth"))
    gt_te["group"] = gt_te["image"]
    n_unk = int((gt_te[CLASSES].sum(axis=1) != 1).sum())
    te = prepare(gt_te)
    print(f"official test rows: {len(gt_te)} | kept {len(te)} | dropped {n_unk} (no valid class label)")

    rng = np.random.RandomState(SEED)
    gcls = tr.groupby("group")["label"].agg(lambda s: int(s.mode().iat[0]))
    gsize = tr.groupby("group").size()
    assign = {}
    for ci, c in enumerate(CLASSES):
        gs = gcls.index[gcls == ci].to_numpy().copy()
        rng.shuffle(gs)
        n = int(gsize[gs].sum())
        fv = VAL_FRAC["tail" if c in TAIL else "head"]
        acc = 0
        for g in gs:
            assign[g] = "val" if acc < fv * n else "train"
            acc += int(gsize[g])
    tr["split"] = tr["group"].map(assign)
    te["split"] = "test"
    return pd.concat([tr, te], ignore_index=True)

SPLITS = f"{WORK}/splits.csv"
df = pd.read_csv(SPLITS) if (os.path.exists(SPLITS) and not REBUILD_SPLITS) else None
if df is not None and not all(os.path.exists(p) for p in df["path"].sample(min(15, len(df)), random_state=0)):
    print("saved split points to files that are gone, rebuilding it")
    df = None
if df is None:
    df = build_splits()
    df.to_csv(SPLITS, index=False)
    print("new split saved to", SPLITS)
else:
    print("loaded the saved split from", SPLITS)

assert df["image"].is_unique, "an image appears twice"
lab = df[df["split"].isin(["train", "val"])]
assert lab.groupby("group")["split"].nunique().max() == 1, "a lesion is split between train and val"
assert set(df.loc[df.split == "train", "image"]).isdisjoint(set(df.loc[df.split == "test", "image"])), "train and test overlap"
TAG = hashlib.md5(open(SPLITS, "rb").read()).hexdigest()[:8]

def cache_split(name):
    sub = df[df["split"] == name].reset_index(drop=True)
    n = len(sub)
    xf, yf, pf = (f"{CACHE}/{TAG}_{name}_x.npy", f"{CACHE}/{TAG}_{name}_y.npy", f"{CACHE}/{TAG}_{name}.progress")
    done = int(open(pf).read()) if os.path.exists(pf) else 0
    if not (os.path.exists(xf) and done >= n and os.path.exists(yf)):
        X = np.lib.format.open_memmap(xf, mode="r+" if (os.path.exists(xf) and done > 0) else "w+", dtype=np.uint8, shape=(n, S, S, 3))
        paths = sub["path"].tolist()
        starts = list(range(done, n, 512))
        with ThreadPoolExecutor(WORKERS) as ex:
            for b0 in tqdm(starts, desc=f"caching {name} ({n} images)"):
                b1 = min(b0 + 512, n)
                X[b0:b1] = np.stack(list(ex.map(lambda p: load_resized(p, S), paths[b0:b1])))
                X.flush()
                open(pf, "w").write(str(b1))
        del X
        np.save(yf, sub["label"].values.astype(np.int64))
    return np.load(xf, mmap_mode="r"), np.load(yf)

Xtr, Ytr = cache_split("train")
Xva, Yva = cache_split("val")
Xte, Yte = cache_split("test")
gc.collect()

tab = pd.crosstab(df["class"], df["split"]).reindex(CLASSES)[["train", "val", "test"]]
print(tab.to_string())
print(f"ready | train {len(Xtr)} | val {len(Xva)} | test {len(Xte)} | {time.time() - t_start:.0f}s")

# %%
import glob, os
print("before:", sorted(os.listdir("/kaggle/working/ckpt8")))
for f in glob.glob("/kaggle/working/ckpt8/*"):
    if not os.path.basename(f).startswith("baseline"):
        os.remove(f); print("removed", os.path.basename(f))
print("after:", sorted(os.listdir("/kaggle/working/ckpt8")))

# %%
import os, gc, time, json, glob
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

DEV = "cuda:0"
SEEDS = [1, 2]
CONDS = ["baseline", "equal", "control_real_dup", "synth_small", "dose_aware"]
EPOCHS, BS, LR, WD = 12, 64, 3e-4, 1e-4
EQUAL_N, BUDGET, FLOOR, CAP, SMALL = 750, 3000, 300, 800, 100
CK = f"{WORK}/ckpt8"
os.makedirs(CK, exist_ok=True)
POOL = f"{WORK}/pool"
tail_cls = [CLASSES.index(c) for c in TAIL]
head_cls = [i for i in range(K) if i not in tail_cls]

# ---------- data: real train + synthetic pool ----------
Xr = np.ascontiguousarray(Xtr); Xva_r = np.ascontiguousarray(Xva); Xte_r = np.ascontiguousarray(Xte)
imgs, labs, srcs = [], [], []
for c in tail_cls:
    for mode in ("repaint", "sdedit"):
        for f in sorted(glob.glob(f"{POOL}/{CLASSES[c]}_{mode}_*.npz")):
            d = np.load(f); imgs.append(d["img"]); labs.append(d["lab"]); srcs.append(d["src"])
P_img, P_lab, P_src = np.concatenate(imgs), np.concatenate(labs), np.concatenate(srcs)
sc = pd.read_csv(f"{WORK}/pool_scores.csv")
assert len(sc) == len(P_lab) and (sc["src"].values == P_src).all(), "pool order does not match pool_scores.csv"
p_target = sc["p_target"].values
p_mode = sc["mode"].values
n_real = len(Xr)
X_all = np.concatenate([Xr, P_img]); Y_all = np.concatenate([Ytr, P_lab])
del Xr; gc.collect()
print("real train", n_real, "| pool", len(P_img))

# ---------- dose rule: difficulty x fidelity ----------
st = pd.read_csv(f"{WORK}/class_stats.csv").set_index("class")
fid = sc.groupby("class")["correct"].mean()
w = pd.Series({c: st.loc[c, "difficulty"] * fid[c] for c in TAIL}); share = w / w.sum()
dose_aware = {c: int(np.clip(round(BUDGET * share[c]), FLOOR, CAP)) for c in TAIL}
print("fidelity:", fid.round(3).to_dict())
print("dose_aware:", dose_aware, "| total", sum(dose_aware.values()), "| equal:", EQUAL_N * len(TAIL))

def build_index(cond, seed):
    base = np.arange(n_real)
    if cond == "baseline":
        return base
    rng = np.random.RandomState(seed)
    extra = []
    for c in TAIL:
        ci = CLASSES.index(c); pc = np.where(P_lab == ci)[0]
        if cond in ("equal", "control_real_dup"):
            sel = rng.choice(pc, EQUAL_N, replace=False)
        elif cond in ("synth_small", "dup_small"):
            pr = np.where((P_lab == ci) & (p_mode == "repaint"))[0]
            sel = rng.choice(pr, SMALL, replace=False)
        elif cond == "dose_aware":
            sel = rng.choice(pc, dose_aware[c], replace=False)
        else:
            sel = pc[np.argsort(-p_target[pc])[:dose_aware[c]]]
        is_real = cond in ("control_real_dup", "dup_small")
        extra.append(P_src[sel] if is_real else n_real + sel)
    return np.concatenate([base] + extra)

# ---------- training (same recipe as the baseline) ----------
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
def to_gpu(xb): return torch.from_numpy(xb).to(DEV, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0

def augment(x):
    n = x.shape[0]
    ang = (torch.rand(n, device=DEV) * 2 - 1) * np.pi
    sc_ = 0.85 + 0.3 * torch.rand(n, device=DEV)
    flip = (torch.rand(n, device=DEV) < 0.5).float() * 2 - 1
    tx, ty = [(torch.rand(n, device=DEV) * 2 - 1) * 0.1 for _ in range(2)]
    cos, sin = torch.cos(ang) / sc_, torch.sin(ang) / sc_
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="reflection", align_corners=False)
    b = 1 + (torch.rand(n, 1, 1, 1, device=DEV) * 2 - 1) * 0.15
    c = 1 + (torch.rand(n, 1, 1, 1, device=DEV) * 2 - 1) * 0.15
    m = x.mean((1, 2, 3), keepdim=True)
    return ((x - m) * c + m * b).clamp(0, 1)

def make_model():
    m = torchvision.models.efficientnet_b0(weights="DEFAULT")
    m.classifier[1] = nn.Linear(1280, K)
    return m.to(DEV)

@torch.no_grad()
def predict(model, X):
    model.eval(); out = []
    for i in range(0, len(X), 128):
        x = (to_gpu(X[i:i + 128]) - MEAN) / STD
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.softmax(model(x).float(), 1).cpu())
    return torch.cat(out).numpy()

def metrics(probs, y):
    p = probs.argmax(1)
    rec = recall_score(y, p, labels=list(range(K)), average=None, zero_division=0)
    return {"bacc": float(balanced_accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro")),
            "acc": float((p == y).mean()), "recall": rec.tolist(),
            "tail_recall": float(rec[tail_cls].mean()), "head_recall": float(rec[head_cls].mean())}

def run(cond, seed):
    name = f"{cond}_s{seed}"; done_f = f"{CK}/{name}_done.json"
    if os.path.exists(done_f):
        print(name, "already finished"); return json.load(open(done_f))
    torch.manual_seed(seed); np.random.seed(seed)
    ids = build_index(cond, seed)
    counts = np.bincount(Ytr, minlength=K)    # ORIGINAL counts: synthetic images must not change the loss weights
    wts = (1 - 0.999) / (1 - 0.999 ** counts); wts = torch.tensor(wts / wts.sum() * K, dtype=torch.float32, device=DEV)
    print(f"{name}: {len(ids)} training images | added per class {dict(zip(CLASSES, (np.bincount(Y_all[ids], minlength=K) - counts).tolist()))}")
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps_per = len(ids) // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * steps_per, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda"); crit = nn.CrossEntropyLoss(weight=wts)
    ep0, hist, best = 0, [], -1.0
    last = f"{CK}/{name}_last.pt"
    if os.path.exists(last):
        s = torch.load(last, map_location=DEV, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"]); ep0, hist, best = s["epoch"], s["hist"], s["best"]
        print(f"  resumed from epoch {ep0}")
    for ep in range(ep0, EPOCHS):
        t0 = time.time(); model.train(); perm = np.random.permutation(len(ids)); tot = 0.0
        for b in range(steps_per):
            bi = np.sort(ids[perm[b * BS:(b + 1) * BS]])
            x = (augment(to_gpu(X_all[bi])) - MEAN) / STD
            y = torch.from_numpy(Y_all[bi]).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        mv = metrics(predict(model, Xva_r), Yva)
        hist.append({"epoch": ep + 1, "loss": tot / steps_per, "val_bacc": mv["bacc"], "val_tail_recall": mv["tail_recall"]})
        if mv["bacc"] > best:
            best = mv["bacc"]; torch.save(model.state_dict(), f"{CK}/{name}_best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep + 1, "hist": hist, "best": best}, last)
        print(f"  ep {ep + 1}/{EPOCHS} loss {tot / steps_per:.3f} | val bacc {mv['bacc']:.3f} tail {mv['tail_recall']:.3f} | {time.time() - t0:.0f}s")
    model.load_state_dict(torch.load(f"{CK}/{name}_best.pt", map_location=DEV))
    res = {"cond": cond, "seed": seed, "n_train": int(len(ids)), "hist": hist,
           "val": metrics(predict(model, Xva_r), Yva), "test": metrics(predict(model, Xte_r), Yte)}
    json.dump(res, open(done_f, "w"))
    for f in (last, f"{CK}/{name}_best.pt"):
        if os.path.exists(f): os.remove(f)
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return res

t_all = time.time()
for sd in SEEDS:
    for cond in CONDS:
        run(cond, sd)
        print(f"--- elapsed {(time.time() - t_all) / 60:.0f} min")

# ---------- summary over every finished run ----------
R = [json.load(open(f)) for f in sorted(glob.glob(f"{CK}/*_done.json"))]
rows = [{"cond": r["cond"], "seed": r["seed"], "n_train": r["n_train"], "val_bacc": r["val"]["bacc"],
         "test_bacc": r["test"]["bacc"], "test_f1": r["test"]["macro_f1"], "test_acc": r["test"]["acc"],
         "tail_recall": r["test"]["tail_recall"], "head_recall": r["test"]["head_recall"],
         **{f"rec_{c}": r["test"]["recall"][i] for i, c in enumerate(CLASSES)}} for r in R]
df8 = pd.DataFrame(rows); df8.to_csv(f"{WORK}/comparison_runs.csv", index=False)
order = [c for c in CONDS if c in set(df8["cond"])]
cols = ["test_bacc", "test_f1", "test_acc", "tail_recall", "head_recall"]
print("\nTest metrics: mean over seeds (n =", df8.groupby("cond").size().to_dict(), ")")
print(df8.groupby("cond")[cols].mean().reindex(order).round(4).to_string())
print("\nStd over seeds")
print(df8.groupby("cond")[cols].std().reindex(order).round(4).to_string())
piv = df8.pivot(index="seed", columns="cond", values="test_bacc")
if "baseline" in piv:
    print("\nPaired change in test balanced accuracy vs baseline (per seed)")
    print(piv.sub(piv["baseline"], axis=0).drop(columns="baseline").round(4).to_string())
print("\nPer-class test recall (mean over seeds)")
print(df8.groupby("cond")[[f"rec_{c}" for c in CLASSES]].mean().reindex(order).round(3).to_string())

fig, ax = plt.subplots(1, 3, figsize=(18, 4))
for a, col in zip(ax[:2], ["test_bacc", "tail_recall"]):
    m = df8.groupby("cond")[col].mean().reindex(order)
    a.bar(range(len(order)), m.values, alpha=0.6)
    for i, c in enumerate(order):
        a.scatter([i] * int((df8.cond == c).sum()), df8[df8.cond == c][col], color="k", s=14, zorder=3)
    a.set_xticks(range(len(order))); a.set_xticklabels(order, rotation=25, ha="right"); a.set_title(col)
    a.set_ylim(max(0, m.min() - 0.05), m.max() + 0.04)
base_rec = df8[df8.cond == "baseline"][[f"rec_{c}" for c in CLASSES]].mean()
w_ = 0.8 / max(1, len(order) - 1)
for i, c in enumerate([o for o in order if o != "baseline"]):
    d = df8[df8.cond == c][[f"rec_{k}" for k in CLASSES]].mean() - base_rec
    ax[2].bar(np.arange(K) + i * w_ - 0.4, d.values, w_, label=c)
ax[2].axhline(0, color="gray", lw=0.8); ax[2].set_xticks(range(K)); ax[2].set_xticklabels(CLASSES)
ax[2].set_title("per-class test recall change vs baseline"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{REPORT}/comparison.png", dpi=110); plt.show()
print(f"CELL 8 DONE | {(time.time() - t_all) / 60:.0f} min")

# %%
import os, glob, hashlib
import numpy as np, pandas as pd

WORK = "/kaggle/working"
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
TAIL = ["AK", "DF", "VASC", "SCC"]
K = len(CLASSES)
TAG = hashlib.md5(open(f"{WORK}/splits.csv", "rb").read()).hexdigest()[:8]
Xtr = np.load(f"{WORK}/cache/{TAG}_train_x.npy", mmap_mode="r")
Ytr = np.load(f"{WORK}/cache/{TAG}_train_y.npy")
print("restored | train", Xtr.shape, "| labels", Ytr.shape)

# %%
import glob, numpy as np, pandas as pd
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

W = "/kaggle/working"
sc = pd.read_csv(f"{W}/pool_scores.csv")
Fp = np.load(f"{W}/pool_feats.npy")
Ftr = np.load(f"{W}/dino_feats.npz")["tr"]

print("real vs synthetic AUC (0.5 = indistinguishable, 1.0 = trivially separable)")
for c in TAIL:
    ci = CLASSES.index(c)
    for mode in ("repaint", "sdedit"):
        sm = ((sc["class"] == c) & (sc["mode"] == mode)).values
        rid = np.where(Ytr == ci)[0]
        X = np.concatenate([Ftr[rid], Fp[sm]])
        y = np.r_[np.zeros(len(rid)), np.ones(sm.sum())]
        g = np.r_[rid, sc["src"].values[sm]]
        oof = np.zeros(len(y))
        for a, b in GroupKFold(5).split(X, y, g):
            clf = LogisticRegression(C=10, max_iter=300, class_weight="balanced").fit(X[a], y[a])
            oof[b] = clf.decision_function(X[b])
        print(f"  {c:5s} {mode:8s} AUC {roc_auc_score(y, oof):.3f}")

# sharpness: variance of the Laplacian (higher = sharper)
def sharp(a):
    return float(ndi.laplace(a.astype(np.float32).mean(-1)).var())

m_idx = np.load(f"{W}/tail_masks.npz")["idx"]
imgs, labs, modes = [], [], []
for c in TAIL:
    for mode in ("repaint", "sdedit"):
        for f in sorted(glob.glob(f"{W}/pool/{c}_{mode}_*.npz")):
            d = np.load(f); imgs.append(d["img"]); labs += [c] * len(d["lab"]); modes += [mode] * len(d["lab"])
imgs, labs, modes = np.concatenate(imgs), np.array(labs), np.array(modes)
rng = np.random.RandomState(0)
print("\nmedian sharpness (real vs synthetic)")
for c in TAIL:
    rid = m_idx[Ytr[m_idx] == CLASSES.index(c)]
    real = np.median([sharp(np.asarray(Xtr[i])) for i in rng.choice(rid, 150, replace=False)])
    out = {"real": round(real, 1)}
    for mode in ("repaint", "sdedit"):
        ii = rng.choice(np.where((labs == c) & (modes == mode))[0], 150, replace=False)
        out[mode] = round(float(np.median([sharp(imgs[i]) for i in ii])), 1)
    print(f"  {c:5s}", out)

# %%
import os, hashlib
import numpy as np

WORK = "/kaggle/working"
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
TAIL = ["AK", "DF", "VASC", "SCC"]
K = len(CLASSES)
TAG = hashlib.md5(open(f"{WORK}/splits.csv", "rb").read()).hexdigest()[:8]
ld = lambda n: np.load(f"{WORK}/cache/{TAG}_{n}_x.npy", mmap_mode="r")
Xtr, Xva, Xte = ld("train"), ld("val"), ld("test")
Ytr, Yva, Yte = (np.load(f"{WORK}/cache/{TAG}_{n}_y.npy") for n in ("train", "val", "test"))
print("restored | train", Xtr.shape, "val", Xva.shape, "test", Xte.shape)

# %%
import os, gc, time, json, glob
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from sklearn.metrics import balanced_accuracy_score, recall_score

DEV, WORK = "cuda:0", "/kaggle/working"
SEEDS, EPOCHS, BS, LR, WD = [0, 1, 2], 12, 64, 3e-4, 1e-4
CK = f"{WORK}/ckpt9"; os.makedirs(CK, exist_ok=True)

# ---------- metadata aligned to the cached image order ----------
sp = pd.read_csv(f"{WORK}/splits.csv")
mfiles = [os.path.join(r, f) for r, _, fs in os.walk("/kaggle/input") for f in fs if "metadata" in f.lower() and f.endswith(".csv")]
M = pd.concat([pd.read_csv(f) for f in mfiles]).drop_duplicates("image").set_index("image")
SITES = sorted(M["anatom_site_general"].dropna().unique())
tr_ages = M.reindex(sp[sp.split == "train"]["image"])["age_approx"]
AM, AS = tr_ages.mean(), tr_ages.std()
def meta_feats(split):
    m = M.reindex(sp[sp.split == split]["image"])
    age = m["age_approx"]
    cols = [((age.fillna(AM) - AM) / AS).values, age.isna().values.astype(float),
            (m["sex"] == "male").values.astype(float), (m["sex"] == "female").values.astype(float)]
    cols += [(m["anatom_site_general"] == s).values.astype(float) for s in SITES]
    return np.stack(cols, 1).astype(np.float32)
Mtr, Mva, Mte = meta_feats("train"), meta_feats("val"), meta_feats("test")
assert len(Mtr) == len(Xtr) and len(Mva) == len(Xva) and len(Mte) == len(Xte)
print("metadata features:", Mtr.shape[1])

Xr, Xv, Xt = (np.ascontiguousarray(a) for a in (Xtr, Xva, Xte))
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
to_gpu = lambda xb: torch.from_numpy(xb).to(DEV).permute(0, 3, 1, 2).float() / 255.0

def augment(x):
    n = x.shape[0]
    ang = (torch.rand(n, device=DEV) * 2 - 1) * np.pi
    sc_ = 0.85 + 0.3 * torch.rand(n, device=DEV)
    flip = (torch.rand(n, device=DEV) < 0.5).float() * 2 - 1
    tx, ty = [(torch.rand(n, device=DEV) * 2 - 1) * 0.1 for _ in range(2)]
    cos, sin = torch.cos(ang) / sc_, torch.sin(ang) / sc_
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="reflection", align_corners=False)
    b = 1 + (torch.rand(n, 1, 1, 1, device=DEV) * 2 - 1) * 0.15
    c = 1 + (torch.rand(n, 1, 1, 1, device=DEV) * 2 - 1) * 0.15
    m = x.mean((1, 2, 3), keepdim=True)
    return ((x - m) * c + m * b).clamp(0, 1)

class Net(nn.Module):
    def __init__(self, use_meta, nm):
        super().__init__()
        self.b = torchvision.models.efficientnet_b0(weights="DEFAULT"); self.b.classifier = nn.Identity()
        self.use_meta = use_meta
        self.mlp = nn.Sequential(nn.Linear(nm, 64), nn.ReLU(), nn.Dropout(0.2)) if use_meta else None
        self.head = nn.Linear(1280 + (64 if use_meta else 0), K)
    def forward(self, x, m):
        f = self.b(x)
        if self.use_meta: f = torch.cat([f, self.mlp(m)], 1)
        return self.head(f)

@torch.no_grad()
def predict(model, X, Mx):
    model.eval(); out = []
    for i in range(0, len(X), 128):
        x = (to_gpu(X[i:i + 128]) - MEAN) / STD; m = torch.from_numpy(Mx[i:i + 128]).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.softmax(model(x, m).float(), 1) + torch.softmax(model(x.flip(3), m).float(), 1)  # flip TTA
        out.append((p / 2).cpu())
    return torch.cat(out).numpy()

def run(use_meta, seed):
    name = f"{'meta' if use_meta else 'nometa'}_s{seed}"
    if os.path.exists(f"{CK}/{name}_test.npy"): print(name, "done"); return
    torch.manual_seed(seed); np.random.seed(seed)
    counts = np.bincount(Ytr, minlength=K)
    w = (1 - 0.999) / (1 - 0.999 ** counts); w = torch.tensor(w / w.sum() * K, dtype=torch.float32, device=DEV)
    model = Net(use_meta, Mtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    spe = len(Xr) // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * spe, pct_start=0.1)
    scaler, crit, best = torch.amp.GradScaler("cuda"), nn.CrossEntropyLoss(weight=w), -1
    for ep in range(EPOCHS):
        t0 = time.time(); model.train(); perm = np.random.permutation(len(Xr))
        for b in range(spe):
            bi = np.sort(perm[b * BS:(b + 1) * BS])
            x = (augment(to_gpu(Xr[bi])) - MEAN) / STD
            y = torch.from_numpy(Ytr[bi]).to(DEV); m = torch.from_numpy(Mtr[bi]).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x, m).float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        pv = predict(model, Xv, Mva); vb = balanced_accuracy_score(Yva, pv.argmax(1))
        if vb > best:
            best = vb; np.save(f"{CK}/{name}_val.npy", pv); np.save(f"{CK}/{name}_test.npy", predict(model, Xt, Mte))
        print(f"{name} ep {ep + 1}/{EPOCHS} val bacc {vb:.3f} | {time.time() - t0:.0f}s")
    del model, opt; gc.collect(); torch.cuda.empty_cache()

for sd in SEEDS:
    for um in (False, True):
        run(um, sd)

# ---------- results ----------
bacc = lambda p: balanced_accuracy_score(Yte, p.argmax(1))
for tag in ("nometa", "meta"):
    P = [np.load(f"{CK}/{tag}_s{s}_test.npy") for s in SEEDS if os.path.exists(f"{CK}/{tag}_s{s}_test.npy")]
    singles = [bacc(p) for p in P]
    ens = np.mean(P, 0)
    rec = recall_score(Yte, ens.argmax(1), average=None, labels=list(range(K)), zero_division=0)
    print(f"{tag:7s} single models {np.round(singles, 4)} mean {np.mean(singles):.4f} std {np.std(singles, ddof=1):.4f} | ensemble of {len(P)}: {bacc(ens):.4f}")
    print("        ensemble per-class recall:", dict(zip(CLASSES, rec.round(3))))

# %%
import os, hashlib
import numpy as np

WORK = "/kaggle/working"
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
TAIL = ["AK", "DF", "VASC", "SCC"]
K = len(CLASSES)
REPORT = f"{WORK}/report"; os.makedirs(REPORT, exist_ok=True)
TAG = hashlib.md5(open(f"{WORK}/splits.csv", "rb").read()).hexdigest()[:8]
ld = lambda n: np.load(f"{WORK}/cache/{TAG}_{n}_x.npy", mmap_mode="r")
Xtr, Xva, Xte = ld("train"), ld("val"), ld("test")
Ytr, Yva, Yte = (np.load(f"{WORK}/cache/{TAG}_{n}_y.npy") for n in ("train", "val", "test"))
print("restored | train", Xtr.shape, "val", Xva.shape, "test", Xte.shape)

# %%
import gc, numpy as np, torch, torch.nn.functional as F
from diffusers import AutoencoderKL
from scipy import ndimage as ndi

DEV, W = "cuda:0", "/kaggle/working"
SD_REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"
m_idx = np.load(f"{W}/tail_masks.npz")["idx"]
vae = AutoencoderKL.from_pretrained(SD_REPO, subfolder="vae").half().to(DEV).eval()
out = []
with torch.no_grad():
    for i in range(0, len(m_idx), 16):
        ii = m_idx[i:i + 16]
        x = torch.from_numpy(np.stack([Xtr[j] for j in ii])).to(DEV).permute(0, 3, 1, 2).float() / 127.5 - 1
        x = F.interpolate(x, size=256, mode="bilinear", align_corners=False)
        y = vae.decode(vae.encode(x.half()).latent_dist.mode()).sample.float().clamp(-1, 1)
        y = F.interpolate(y, size=224, mode="bilinear", antialias=True, align_corners=False)
        out.append(((y + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy())
RT = np.concatenate(out)
np.savez(f"{W}/vae_roundtrip.npz", img=RT, idx=m_idx)
del vae; gc.collect(); torch.cuda.empty_cache()

sharp = lambda a: float(ndi.laplace(a.astype(np.float32).mean(-1)).var())
for c in TAIL:
    k = np.where(Ytr[m_idx] == CLASSES.index(c))[0][:100]
    print(c, "real", round(np.median([sharp(np.asarray(Xtr[m_idx[j]])) for j in k]), 1),
          "| round-trip", round(np.median([sharp(RT[j]) for j in k]), 1))
print("saved", RT.shape)

# %%
import os, gc, time, json, glob
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

DEV = "cuda:0"
SEEDS = [0, 1, 2]
CONDS = ["baseline", "equal", "control_real_dup", "synth_small", "dose_aware", "control_real_vae"]
EPOCHS, BS, LR, WD = 12, 64, 3e-4, 1e-4
EQUAL_N, BUDGET, FLOOR, CAP, SMALL = 750, 3000, 300, 800, 100
CK = f"{WORK}/ckpt8"
os.makedirs(CK, exist_ok=True)
POOL = f"{WORK}/pool"
tail_cls = [CLASSES.index(c) for c in TAIL]
head_cls = [i for i in range(K) if i not in tail_cls]

# ---------- data: real train + synthetic pool + round-tripped real ----------
Xr = np.ascontiguousarray(Xtr); Xva_r = np.ascontiguousarray(Xva); Xte_r = np.ascontiguousarray(Xte)
imgs, labs, srcs = [], [], []
for c in tail_cls:
    for mode in ("repaint", "sdedit"):
        for f in sorted(glob.glob(f"{POOL}/{CLASSES[c]}_{mode}_*.npz")):
            d = np.load(f); imgs.append(d["img"]); labs.append(d["lab"]); srcs.append(d["src"])
P_img, P_lab, P_src = np.concatenate(imgs), np.concatenate(labs), np.concatenate(srcs)
sc = pd.read_csv(f"{WORK}/pool_scores.csv")
assert len(sc) == len(P_lab) and (sc["src"].values == P_src).all(), "pool order does not match pool_scores.csv"
p_target = sc["p_target"].values
p_mode = sc["mode"].values
rt = np.load(f"{WORK}/vae_roundtrip.npz"); RT_img, RT_idx = rt["img"], rt["idx"]
rt_pos = {int(i): k for k, i in enumerate(RT_idx)}
n_real = len(Xr)
X_all = np.concatenate([Xr, P_img, RT_img]); Y_all = np.concatenate([Ytr, P_lab, Ytr[RT_idx]])
RT_BASE = n_real + len(P_img)
del Xr; gc.collect()
print("real train", n_real, "| pool", len(P_img), "| round-trip real", len(RT_img))

# ---------- dose rule ----------
st = pd.read_csv(f"{WORK}/class_stats.csv").set_index("class")
fid = sc.groupby("class")["correct"].mean()
w = pd.Series({c: st.loc[c, "difficulty"] * fid[c] for c in TAIL}); share = w / w.sum()
dose_aware = {c: int(np.clip(round(BUDGET * share[c]), FLOOR, CAP)) for c in TAIL}

def build_index(cond, seed):
    base = np.arange(n_real)
    if cond == "baseline":
        return base
    rng = np.random.RandomState(seed)
    extra = []
    for c in TAIL:
        ci = CLASSES.index(c); pc = np.where(P_lab == ci)[0]
        if cond in ("equal", "control_real_dup", "control_real_vae"):
            sel = rng.choice(pc, EQUAL_N, replace=False)
        elif cond in ("synth_small", "dup_small"):
            pr = np.where((P_lab == ci) & (p_mode == "repaint"))[0]
            sel = rng.choice(pr, SMALL, replace=False)
        elif cond == "dose_aware":
            sel = rng.choice(pc, dose_aware[c], replace=False)
        else:
            sel = pc[np.argsort(-p_target[pc])[:dose_aware[c]]]
        if cond == "control_real_vae":
            extra.append(RT_BASE + np.array([rt_pos[int(i)] for i in P_src[sel]]))
        elif cond in ("control_real_dup", "dup_small"):
            extra.append(P_src[sel])
        else:
            extra.append(n_real + sel)
    return np.concatenate([base] + extra)

# ---------- training (same recipe as before) ----------
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
def to_gpu(xb): return torch.from_numpy(xb).to(DEV, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0

def augment(x):
    n = x.shape[0]
    ang = (torch.rand(n, device=DEV) * 2 - 1) * np.pi
    sc_ = 0.85 + 0.3 * torch.rand(n, device=DEV)
    flip = (torch.rand(n, device=DEV) < 0.5).float() * 2 - 1
    tx, ty = [(torch.rand(n, device=DEV) * 2 - 1) * 0.1 for _ in range(2)]
    cos, sin = torch.cos(ang) / sc_, torch.sin(ang) / sc_
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="reflection", align_corners=False)
    b = 1 + (torch.rand(n, 1, 1, 1, device=DEV) * 2 - 1) * 0.15
    c = 1 + (torch.rand(n, 1, 1, 1, device=DEV) * 2 - 1) * 0.15
    m = x.mean((1, 2, 3), keepdim=True)
    return ((x - m) * c + m * b).clamp(0, 1)

def make_model():
    m = torchvision.models.efficientnet_b0(weights="DEFAULT")
    m.classifier[1] = nn.Linear(1280, K)
    return m.to(DEV)

@torch.no_grad()
def predict(model, X):
    model.eval(); out = []
    for i in range(0, len(X), 128):
        x = (to_gpu(X[i:i + 128]) - MEAN) / STD
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.softmax(model(x).float(), 1).cpu())
    return torch.cat(out).numpy()

def metrics(probs, y):
    p = probs.argmax(1)
    rec = recall_score(y, p, labels=list(range(K)), average=None, zero_division=0)
    return {"bacc": float(balanced_accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro")),
            "acc": float((p == y).mean()), "recall": rec.tolist(),
            "tail_recall": float(rec[tail_cls].mean()), "head_recall": float(rec[head_cls].mean())}

def run(cond, seed):
    name = f"{cond}_s{seed}"; done_f = f"{CK}/{name}_done.json"
    if os.path.exists(done_f):
        print(name, "already finished"); return json.load(open(done_f))
    torch.manual_seed(seed); np.random.seed(seed)
    ids = build_index(cond, seed)
    counts = np.bincount(Ytr, minlength=K)    # original counts: added images must not change loss weights
    wts = (1 - 0.999) / (1 - 0.999 ** counts); wts = torch.tensor(wts / wts.sum() * K, dtype=torch.float32, device=DEV)
    print(f"{name}: {len(ids)} training images | added per class {dict(zip(CLASSES, (np.bincount(Y_all[ids], minlength=K) - counts).tolist()))}")
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps_per = len(ids) // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * steps_per, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda"); crit = nn.CrossEntropyLoss(weight=wts)
    ep0, hist, best = 0, [], -1.0
    last = f"{CK}/{name}_last.pt"
    if os.path.exists(last):
        s = torch.load(last, map_location=DEV, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"]); ep0, hist, best = s["epoch"], s["hist"], s["best"]
        print(f"  resumed from epoch {ep0}")
    for ep in range(ep0, EPOCHS):
        t0 = time.time(); model.train(); perm = np.random.permutation(len(ids)); tot = 0.0
        for b in range(steps_per):
            bi = np.sort(ids[perm[b * BS:(b + 1) * BS]])
            x = (augment(to_gpu(X_all[bi])) - MEAN) / STD
            y = torch.from_numpy(Y_all[bi]).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        mv = metrics(predict(model, Xva_r), Yva)
        hist.append({"epoch": ep + 1, "loss": tot / steps_per, "val_bacc": mv["bacc"], "val_tail_recall": mv["tail_recall"]})
        if mv["bacc"] > best:
            best = mv["bacc"]; torch.save(model.state_dict(), f"{CK}/{name}_best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep + 1, "hist": hist, "best": best}, last)
        print(f"  ep {ep + 1}/{EPOCHS} loss {tot / steps_per:.3f} | val bacc {mv['bacc']:.3f} tail {mv['tail_recall']:.3f} | {time.time() - t0:.0f}s")
    model.load_state_dict(torch.load(f"{CK}/{name}_best.pt", map_location=DEV))
    res = {"cond": cond, "seed": seed, "n_train": int(len(ids)), "hist": hist,
           "val": metrics(predict(model, Xva_r), Yva), "test": metrics(predict(model, Xte_r), Yte)}
    json.dump(res, open(done_f, "w"))
    for f in (last, f"{CK}/{name}_best.pt"):
        if os.path.exists(f): os.remove(f)
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return res

t_all = time.time()
for sd in SEEDS:
    for cond in CONDS:
        run(cond, sd)
        print(f"--- elapsed {(time.time() - t_all) / 60:.0f} min")

# ---------- summary ----------
R = [json.load(open(f)) for f in sorted(glob.glob(f"{CK}/*_done.json"))]
rows = [{"cond": r["cond"], "seed": r["seed"], "n_train": r["n_train"], "val_bacc": r["val"]["bacc"],
         "test_bacc": r["test"]["bacc"], "test_f1": r["test"]["macro_f1"], "test_acc": r["test"]["acc"],
         "tail_recall": r["test"]["tail_recall"], "head_recall": r["test"]["head_recall"],
         **{f"rec_{c}": r["test"]["recall"][i] for i, c in enumerate(CLASSES)}} for r in R]
df8 = pd.DataFrame(rows); df8.to_csv(f"{WORK}/comparison_runs.csv", index=False)
order = [c for c in CONDS if c in set(df8["cond"])]
cols = ["test_bacc", "test_f1", "test_acc", "tail_recall", "head_recall"]
print("\nMean over seeds (n =", df8.groupby("cond").size().reindex(order).to_dict(), ")")
print(df8.groupby("cond")[cols].mean().reindex(order).round(4).to_string())
print("\nStd over seeds")
print(df8.groupby("cond")[cols].std().reindex(order).round(4).to_string())
piv = df8.pivot(index="seed", columns="cond", values="test_bacc")[order]
print("\nTest balanced accuracy per seed")
print(piv.round(4).to_string())
print("\nPaired change vs baseline (per seed)")
print(piv.sub(piv["baseline"], axis=0).drop(columns="baseline").round(4).to_string())
print("\nPer-class test recall (mean over seeds)")
print(df8.groupby("cond")[[f"rec_{c}" for c in CLASSES]].mean().reindex(order).round(3).to_string())
print(f"CELL 9b DONE | {(time.time() - t_all) / 60:.0f} min")

# %%
import os, gc, random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, f1_score, recall_score,
    balanced_accuracy_score, accuracy_score, confusion_matrix
)
from sklearn.manifold import TSNE

# Reuse Xtr, Xva, Xte, Ytr, Yva, Yte and CLASSES already loaded
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = len(CLASSES)
BATCH_SIZE = 16
EPOCHS = 5
SAVE_DIR = "/kaggle/working/efficientnet_results"
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_PATH = os.path.join(SAVE_DIR, "best_model.pt")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1)

def labels_array(y):
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[-1] > 1:
        y = y.argmax(axis=-1)
    return y.reshape(-1).astype(np.int64)

Ytr_r, Yva_r, Yte_r = map(labels_array, (Ytr, Yva, Yte))

def batch_images(x):
    # Convert only the current batch, not the whole dataset.
    x = np.asarray(x)
    if x.shape[-1] in (1, 3, 4):
        x = x[..., :3].transpose(0, 3, 1, 2)
    elif x.shape[1] in (1, 3, 4):
        x = x[:, :3]
    else:
        raise ValueError(f"Unexpected image shape: {x.shape}")

    x = torch.from_numpy(np.ascontiguousarray(x)).to(
        DEVICE, dtype=torch.float32
    )
    if x.max() > 1:
        x = x / 255.0
    return (x - MEAN) / STD

def make_model():
    try:
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        net = torchvision.models.efficientnet_b0(weights=weights)
    except Exception as e:
        print("Pretrained weights unavailable; using random initialization:", e)
        net = torchvision.models.efficientnet_b0(weights=None)

    net.classifier[1] = nn.Linear(net.classifier[1].in_features, K)
    return net.to(DEVICE)

def forward_amp(model, xb):
    if DEVICE.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(xb)
    return model(xb)

# -------- Training --------
model = make_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
best_score = -1.0
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    order = np.random.permutation(len(Xtr))
    for start in range(0, len(order), BATCH_SIZE):
        ids = order[start:start + BATCH_SIZE]
        xb = batch_images(Xtr[ids])
        yb = torch.as_tensor(Ytr_r[ids], device=DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda")
        ):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(ids)

        del xb, yb, logits, loss

    # Validation in batches
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for start in range(0, len(Xva), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(Xva))
            xb = batch_images(Xva[start:end])
            logits = forward_amp(model, xb)
            val_pred.extend(logits.argmax(1).cpu().numpy())
            val_true.extend(Yva_r[start:end])
            del xb, logits

    score = balanced_accuracy_score(val_true, val_pred)
    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"loss={total_loss / len(Xtr):.4f} | val balanced accuracy={score:.4f}"
    )

    if score > best_score:
        best_score = score
        torch.save(model.state_dict(), BEST_PATH)

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

# Restore best validation checkpoint
model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# -------- Test predictions and small feature sample for t-SNE --------
test_probs_parts = []
feature_sample, label_sample = [], []
rng = np.random.default_rng(42)
MAX_TSNE = 1000

with torch.no_grad():
    for start in range(0, len(Xte), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(Xte))
        xb = batch_images(Xte[start:end])

        # Penultimate-layer features for t-SNE
        feat = model.features(xb)
        feat = model.avgpool(feat).flatten(1)
        logits = model.classifier(feat)
        probs = torch.softmax(logits.float(), dim=1)

        test_probs_parts.append(probs.cpu().numpy())

        # Keep at most MAX_TSNE feature vectors in memory.
        f = feat.cpu().numpy()
        labels_batch = Yte_r[start:end]
        for row, label in zip(f, labels_batch):
            if len(feature_sample) < MAX_TSNE:
                feature_sample.append(row)
                label_sample.append(label)
            else:
                j = rng.integers(0, start + len(f))
                if j < MAX_TSNE:
                    feature_sample[j] = row
                    label_sample[j] = label

        del xb, feat, logits, probs

test_probs = np.concatenate(test_probs_parts, axis=0)
y_true = Yte_r
y_pred = test_probs.argmax(axis=1)

np.save(os.path.join(SAVE_DIR, "test_probs.npy"), test_probs)
np.save(os.path.join(SAVE_DIR, "y_true.npy"), y_true)
np.save(os.path.join(SAVE_DIR, "y_pred.npy"), y_pred)

print(classification_report(
    y_true, y_pred, labels=np.arange(K),
    target_names=CLASSES, zero_division=0
))

macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
recalls = recall_score(
    y_true, y_pred, labels=np.arange(K),
    average=None, zero_division=0
)
g_mean = float(np.prod(np.maximum(recalls, 0)) ** (1 / K))

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Balanced accuracy:", balanced_accuracy_score(y_true, y_pred))
print("Macro-F1:", macro_f1)
print("G-mean:", g_mean)

# Per-class recall
plt.figure(figsize=(9, 4))
plt.bar(CLASSES, recalls)
plt.ylim(0, 1)
plt.ylabel("Recall")
plt.title(f"Per-class Recall | G-mean = {g_mean:.3f}")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "recall.png"), dpi=150)
plt.show()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=np.arange(K))
plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.xticks(range(K), CLASSES, rotation=45, ha="right")
plt.yticks(range(K), CLASSES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=150)
plt.show()

# t-SNE on a limited feature sample
if len(feature_sample) >= 4:
    feats = np.asarray(feature_sample)
    labs = np.asarray(label_sample)
    perplexity = min(30, max(2, (len(feats) - 1) // 3))
    emb = TSNE(
        n_components=2, perplexity=perplexity,
        init="pca", learning_rate="auto", random_state=42
    ).fit_transform(feats)

    plt.figure(figsize=(8, 6))
    for i, name in enumerate(CLASSES):
        mask = labs == i
        if mask.any():
            plt.scatter(emb[mask, 0], emb[mask, 1], s=12, alpha=0.7, label=name)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.title("t-SNE of Test Features")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "tsne.png"), dpi=150)
    plt.show()

print("Saved results to:", SAVE_DIR)
