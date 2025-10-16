# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB
from loader_wildfire_inputs import load_inputs, compute_d_param
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

# 1) Excel'i okuyun (sizin klasörünüz)
excel_path = r"C:\Users\Mert\deprem\wildfire\inputs_to_load-5x5_burcu.xlsx"

data = load_inputs(excel_path)   # tüm set/parametreler burada
# print(demo_print_summary(data))  # hızlı bütünlük kontrolü (opsiyonel)

N          = data["N"]
K          = data["K"]                 # araç seti (parameters sayfasından)
coords     = data["coords"]
pi         = data["v0"]
degrade    = data["degrade"]
ameliorate = data["ameliorate"]
state      = data["state"]
neighbors  = data["neighbors"]
A          = data["A"]
P          = data["P"]
beta=data["beta"]

d = compute_d_param(coords, speed_km_per_min=3.83, K=K)

Na = [i for i in N if i in {8, 11, 20}]
e = {i: (1 if i in Na else 0) for i in N}
alpha = {i: 5 for i in N}
omega = {i: 25000 for i in N}  #25 km2 alan için 250,000,000 litre su gerekli
mu_default = 1667  # birim: litre/dakika
mu = {k: mu_default for k in K}

M = 1e6  # Big-M
epsilon=0.0000000000000000000001 #????
# 2) Gurobi modeli (örnek iskelet)
m = gp.Model("wildfire_model")


p = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="p")  # p_i ≥ 0 : Collected reward at node i
y = m.addVars(N, vtype=GRB.BINARY, name="y")  # y_i ∈ {0,1} : 1 if a fire occurs at node i
ts = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="ts")   # t_i^s ≥ 0 : Fire start time at node i
tm = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="tm")   # t_i^m ≥ 0 : Spread threshold (midpoint) time at node i
te = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="te")   # t_i^e ≥ 0 : Natural end (burnout) time at node i
tc = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="tc")  # t_i^c ≥ 0 : Time when fire at node i is brought under control
u_pre = m.addVars(N, vtype=GRB.BINARY, name="u_pre")  # u_i^pre ∈ {0,1} : fire at node i controlled before tm
u_post = m.addVars(N, vtype=GRB.BINARY, name="u_post")  # u_i^post ∈ {0,1} : fire at node i controlled after tm but before te
x = m.addVars(N, K, vtype=GRB.BINARY, name="x")  # x_ik ∈ {0,1} : team k assigned to node i
t = m.addVars(N, K, vtype=GRB.CONTINUOUS, lb=0, name="t")   # t_ik ≥ 0 : dispatch time of team k assigned to node i
s = m.addVars(N, K, vtype=GRB.CONTINUOUS, lb=0, name="s")  # s_ik ≥ 0 : processing (suppression) duration of team k at node i
z = m.addVars(N, N, vtype=GRB.BINARY, name="z")   # z_ij ∈ {0,1} : fire spreads from node i to node j
q = m.addVars(N, N, vtype=GRB.BINARY, name="q")  # q_ij ∈ {0,1} : fire at node j ignited by node i


m.addConstrs(( p[i] <= pi[i] -beta[i]*(tc[i]-ts[i]) for i in N),
    name="cons 2")

m.addConstrs(( p[i] <= pi[i]*(u_pre[i]+u_post[i]+1-y[i]) for i in N),
    name="cons 3")

m.addConstrs((gp.quicksum(mu[k]*s[i,k] for k in K) >= omega[i]*y[i] for i in N),
    name="cons 4")

m.addConstrs((t[i,k]+s[i,k] <=M*x[i,k] for i in N for k in K),
    name="cons 5")

m.addConstrs((t[i,k] + d[(i,k)] + s[i,k] <= tc[i] + M*(1 - x[i,k]) for i in N for k in K),
             name="cons 6")

m.addConstrs((t[i,k] + d[(i,k)] >= ts[i] - M*(1 - x[i,k]) for i in N for k in K),
             name="cons 7")


m.addConstrs((x[i,k]<=y[i] for i in N for k in K),
    name="cons 8")

m.addConstrs((u_pre[i]+u_post[i] >= x[i,k] for i in N for k in K),
    name="cons 9")

m.addConstrs((u_pre[i]+u_post[i] <= gp.quicksum(x[i,k] for k in K) for i in N ),
    name="cons 10")

m.addConstrs((gp.quicksum(x[i,k] for i in N) <= 1 for k in K ),
             name="11")


m.addConstrs((tc[i]<=te[i] for i in N ),
    name="cons 12")

m.addConstrs((tc[i]>=ts[i] - M*(1-u_post[i]-u_pre[i]) for i in N ),
    name="cons 13")

m.addConstrs((tc[i]<=M*(u_post[i]+u_pre[i]) for i in N ),
    name="cons 14")

m.addConstrs((tc[i]<= tm[i] +epsilon+ M *(1-u_pre[i]) for i in N ),
    name="cons 15")

m.addConstrs((tc[i]>=tm[i]  - M *(1-u_post[i]) for i in N ),
    name="cons 16")

m.addConstrs((u_pre[i]+u_post[i] <=1 for i in N ),
    name="cons 17")
m.addConstrs((gp.quicksum(z[i, j] for j in neighbors[i]) == len(neighbors[i]) * (y[i] - u_pre[i]) for i in N),
                 name="cons 18")

m.addConstrs((len(neighbors[j]) * y[j] >= gp.quicksum(z[i, j] for i in neighbors[j]) for j in N),
                 name="cons 19")

m.addConstrs((q[i,j]<=z[i,j] for i in N for j in neighbors[i]),
                 name="cons 20")

m.addConstrs((gp.quicksum(q[i, j] for i in neighbors[j]) == y[j]-e[j] for j in N ),
                 name="cons 21")


m.addConstrs((ts[j]==gp.quicksum(tm[i]*q[i,j] for i in neighbors[j])  for j in N ),
                 name="cons 22")

m.addConstrs((ts[j]<= tm[i]+ M*(1-z[i,j]) for j in N for i in neighbors[j] ),
                 name="cons 23")

m.addConstr(gp.quicksum(y[i] for i in Na) == len(Na),
    name="cons 24")

m.addConstr( gp.quicksum(ts[i] for i in Na) == 0,
    name="cons 25")



m.addConstrs(( tm[i] == ts[i] + alpha[i] / degrade[i] for i in N ),
    name="cons 26")


m.addConstrs(( te[i] == tm[i] + alpha[i] / ameliorate[i] for i in N ),
    name="cons 27")



# preset_values = {
#     # 8: 1,  #
#     # 11: 1,  #
#     # 20: 1   #
# }
# for i, val in preset_values.items():
#     m.addConstr(u_pre[i] == val, name=f"fix_u_b_{i}")



# objective function
m.setObjective(gp.quicksum(p[i] for i in N), GRB.MAXIMIZE)

m.optimize()

# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def _single_index_to_df(var, index_name="i", value_name=None):
    """Gurobi VarDict (tek indeksli) -> DataFrame."""
    if value_name is None:
        value_name = var._name
    rows = []
    for i in var.keys():
        v = var[i].X if var[i].X is not None else float("nan")
        rows.append({index_name: i, value_name: v})
    return pd.DataFrame(rows)

def _double_index_to_df(var, index_names=("i", "k"), value_name=None):
    """Gurobi VarDict (çift indeksli) -> DataFrame."""
    if value_name is None:
        value_name = var._name
    i_name, j_name = index_names
    rows = []
    for key in var.keys():
        i, j = key
        v = var[key].X if var[key].X is not None else float("nan")
        rows.append({i_name: i, j_name: j, value_name: v})
    return pd.DataFrame(rows)

def _safe_write(writer, sheet_name, df):
    """Boş/None ise sayfayı atla (robust yazım)."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return
    df.to_excel(writer, index=False, sheet_name=sheet_name)

def _read_params(excel_inputs):
    """
    inputs Excel'inin 'parameters' sayfasından n_nodes ve n_vehicles'ı çek.
    Yoksa None döndür.
    """
    try:
        p = pd.read_excel(excel_inputs, sheet_name="parameters")
        p = p.set_index("parameter")["value"].to_dict()
        n_nodes = int(p.get("n_nodes")) if "n_nodes" in p else None
        n_vehicles = int(p.get("n_vehicles")) if "n_vehicles" in p else None
        return n_nodes, n_vehicles, p
    except Exception:
        return None, None, {}

def write_results_to_excel(model: gp.Model,
                           vars_dict: dict,
                           excel_inputs_path,                # <-- yeni: girdi dosyası (adı için ve parametre okumak için)
                           out_dir=r"C:\Users\Mert\deprem\wildfire\result",
                           file_prefix="results"):

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        raise RuntimeError(f"Çözülebilir bir çözüm bulunamadı (Status={model.Status}).")

    # Girdi parametrelerini oku
    excel_inputs_path = Path(excel_inputs_path)
    n_nodes, n_vehicles, params_all = _read_params(excel_inputs_path)

    # Dosya adı: <prefix>__<girdi_adi>__YYYYmmdd_HHMMSS.xlsx
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    in_stem = excel_inputs_path.stem  # örn: inputs_to_load-5x5_burcu
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    xlsx = out_path / f"{file_prefix}__{in_stem}__{ts_str}.xlsx"

    # Gurobi meta
    status = model.Status
    objval = getattr(model, "ObjVal", float("nan"))
    mipgap = getattr(model, "MIPGap", float("nan"))
    runtime = getattr(model, "Runtime", float("nan"))          # saniye
    nodecnt = getattr(model, "NodeCount", float("nan"))
    nvars   = getattr(model, "NumVars", float("nan"))
    ncons   = getattr(model, "NumConstrs", float("nan"))
    nbin    = getattr(model, "NumBinVars", float("nan"))
    nint    = getattr(model, "NumIntVars", float("nan"))
    nqc     = getattr(model, "NumQConstrs", float("nan"))
    ngen    = getattr(model, "NumGenConstrs", float("nan"))

    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
        # --- Meta sayfası ---
        meta_rows = [
            ("InputFile",             str(excel_inputs_path)),
            ("Status",                status),
            ("Objective",             objval),
            ("MIPGap",                mipgap),
            ("Runtime_sec",           runtime),
            ("NodeCount",             nodecnt),
            ("NumVars",               nvars),
            ("NumConstrs",            ncons),
            ("NumBinVars",            nbin),
            ("NumIntVars",            nint),
            ("NumQConstrs",           nqc),
            ("NumGenConstrs",         ngen),
            ("n_nodes (from input)",  n_nodes),
            ("n_vehicles (from input)", n_vehicles),
            ("ExportTime",            ts_str),
        ]
        meta = pd.DataFrame(meta_rows, columns=["metric", "value"])
        _safe_write(writer, "meta", meta)

        # --- (opsiyonel) parameters snapshot ---
        if params_all:
            params_df = pd.DataFrame(
                [{"parameter": k, "value": v} for k, v in params_all.items()]
            )
            _safe_write(writer, "parameters_inherited", params_df)

        # --- Tek indeksli değişkenler (varsa yazar) ---
        for name, (sheet, idx, colname) in {
            "p": ("p", "i", "p_i"),
            "y": ("y", "i", "y_i"),
            "ts": ("ts", "i", "t_i_s"),
            "tm": ("tm", "i", "t_i_m"),
            "te": ("te", "i", "t_i_e"),
            "tc": ("tc", "i", "t_i_c"),
            "u_pre": ("u_pre", "i", "u_i_pre"),
            "u_post": ("u_post", "i", "u_i_post"),
        }.items():
            if name in vars_dict and vars_dict[name] is not None:
                df = _single_index_to_df(vars_dict[name], idx, colname)
                _safe_write(writer, sheet, df)

        # --- Çift indeksli değişkenler (varsa yazar) ---
        for name, (sheet, idx_names, colname) in {
            "x": ("x", ("i","k"), "x_ik"),
            "t": ("t", ("i","k"), "t_ik"),
            "s": ("s", ("i","k"), "s_ik"),
            "z": ("z", ("i","j"), "z_ij"),
            "q": ("q", ("i","j"), "q_ij"),
            # İsterseniz "a","b","c" gibi diğer çift indekslileri de buraya ekleyin:
            "a": ("a", ("k","t"), "a_kt"),  # örnek şema; gerçek indekslerinize göre değiştirin
            "b": ("b", ("i","t"), "b_it"),
            "c": ("c", ("i","t"), "c_it"),
        }.items():
            if name in vars_dict and vars_dict[name] is not None:
                df = _double_index_to_df(vars_dict[name], idx_names, colname)
                _safe_write(writer, sheet, df)

    print(f"Sonuçlar Excel'e yazıldı: {xlsx}")
    return str(xlsx)
# vars_dict örneği (sizde hangi değişkenler varsa onları ekleyin)
vars_dict = {
    "p": p, "y": y, "ts": ts, "tm": tm, "te": te, "tc": tc,
    "u_pre": u_pre, "u_post": u_post,
    "x": x, "t": t, "s": s, "z": z, "q": q,
    # "a": a, "b": b, "c": c,  # sizde varsa açın
}

out_file = write_results_to_excel(
    model=m,
    vars_dict=vars_dict,
    excel_inputs_path=r"C:\Users\Mert\deprem\wildfire\inputs_to_load-5x5_burcu.xlsx",
    out_dir=r"C:\Users\Mert\deprem\wildfire\result",
    file_prefix="results"
)



# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Excel dosya yolu
excel_path = r"C:\Users\Mert\deprem\wildfire\inputs_to_load-5x5_burcu.xlsx"

# Çıktı klasörü
out_dir = Path("maps")
out_dir.mkdir(parents=True, exist_ok=True)

# Veriyi oku
df = pd.read_excel(excel_path, sheet_name="inputs_df")

# Grid boyutu
nrows, ncols = 5, 5

# Node → değerleri 5x5 matrise çevir
def to_matrix(values):
    mat = []
    for r in range(nrows):
        row_vals = []
        for c in range(ncols):
            node_id = r * ncols + (c + 1)
            row_vals.append(values.get(node_id, 0))
        mat.append(row_vals)
    return pd.DataFrame(mat[::-1])  # üstten alta ters

# Dict’ler
value_map   = dict(zip(df["node_id"], df["value_at_start"]))
degrade_map = dict(zip(df["node_id"], df["fire_degradation_rate"]))

# Matrisler
value_matrix   = to_matrix(value_map)
degrade_matrix = to_matrix(degrade_map)

# Timestamp
ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------- Initial Situation (yeşil + değerler yazılı) ----------------
plt.figure(figsize=(8, 6))
ax = sns.heatmap(value_matrix, annot=False, cmap="Greens", cbar=True,
                 linewidths=0.5, linecolor="gray", square=True)

for r in range(nrows):
    for c in range(ncols):
        node_id = (nrows - r - 1) * ncols + (c + 1)
        val = value_matrix.iloc[r, c]
        ax.text(c + 0.5, r + 0.5, f"{node_id}\n{val:.1f}",
                ha="center", va="center", color="black", fontsize=9)

plt.title("Initial Situation", fontsize=14, weight="bold")
plt.savefig(out_dir / f"initial_situation_{ts_str}.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------- Spread Rate (kırmızı + sadece node id) ----------------
plt.figure(figsize=(8, 6))
ax = sns.heatmap(degrade_matrix, annot=False, cmap="Reds", cbar=True,
                 linewidths=0.5, linecolor="gray", square=True)

for r in range(nrows):
    for c in range(ncols):
        node_id = (nrows - r - 1) * ncols + (c + 1)
        ax.text(c + 0.5, r + 0.5, f"{node_id}",
                ha="center", va="center", color="black", fontsize=9)

plt.title("Spread Rate", fontsize=14, weight="bold")
plt.savefig(out_dir / f"spread_rate_{ts_str}.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Haritalar kaydedildi: {out_dir}")

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from pathlib import Path
from datetime import datetime
import textwrap

# === GİRİŞ ===
excel_inputs = r"C:\Users\Mert\deprem\wildfire\inputs_to_load-5x5_burcu.xlsx"
results_dir  = Path(r"C:\Users\Mert\deprem\wildfire\result")
out_dir      = Path(r"C:\Users\Mert\deprem\wildfire\maps")
out_dir.mkdir(parents=True, exist_ok=True)

# === PARAMETRELERİ OKU ===
try:
    params = pd.read_excel(excel_inputs, sheet_name="parameters")
    params = params.set_index("parameter")["value"].to_dict()
    n_vehicles = int(params.get("n_vehicles", 0))
except Exception:
    n_vehicles = 0  # yoksa 0 kabul et

# Girdi adı ve zaman etiketi
in_stem = Path(excel_inputs).stem
ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# === BAŞLANGIÇ YANGINLARI (Na) ===
Na = {8, 11, 20}

# === VERİYİ OKU ===
idf = pd.read_excel(excel_inputs, sheet_name="inputs_df")

# === GRID BOYUTU OTO ===
x_vals = sorted(idf["x_coordinate"].unique())
y_vals = sorted(idf["y_coordinate"].unique())
nrows, ncols = len(y_vals), len(x_vals)
print(f"Grid boyutu: {nrows}x{ncols}")

# --- Yardımcı fonksiyon: değerleri matrise çevir ---
def to_matrix(values):
    mat = []
    for r in range(nrows):
        row_vals = []
        for c in range(ncols):
            node_id = r * ncols + (c + 1)
            row_vals.append(values.get(node_id, 0))
        mat.append(row_vals)
    return pd.DataFrame(mat[::-1])  # üstten alta ters

# Dict’ler
value_map   = dict(zip(idf["node_id"], idf["value_at_start"]))
degrade_map = dict(zip(idf["node_id"], idf["fire_degradation_rate"]))

value_matrix   = to_matrix(value_map)
degrade_matrix = to_matrix(degrade_map)

# === 1) INITIAL SITUATION ===
plt.figure(figsize=(8, 6))
ax = sns.heatmap(value_matrix, annot=False, cmap="Greens", cbar=True,
                 linewidths=0.5, linecolor="gray", square=True)

for r in range(nrows):
    for c in range(ncols):
        node_id = (nrows - r - 1) * ncols + (c + 1)
        val = value_matrix.iloc[r, c]
        ax.text(c + 0.5, r + 0.5, f"{node_id}\n{val:.1f}",
                ha="center", va="center", color="black", fontsize=9)

plt.title(f"Initial Situation ({nrows}×{ncols})", fontsize=14, weight="bold")
out_initial = out_dir / f"initial_situation__{in_stem}__veh{n_vehicles}__{ts_str}.png"
plt.savefig(out_initial, dpi=300, bbox_inches="tight")
plt.close()

# === 2) SPREAD RATE ===
plt.figure(figsize=(8, 6))
ax = sns.heatmap(degrade_matrix, annot=False, cmap="Reds", cbar=True,
                 linewidths=0.5, linecolor="gray", square=True)
for r in range(nrows):
    for c in range(ncols):
        node_id = (nrows - r - 1) * ncols + (c + 1)
        ax.text(c + 0.5, r + 0.5, f"{node_id}",
                ha="center", va="center", color="black", fontsize=9)

plt.title(f"Spread Rate ({nrows}×{ncols})", fontsize=14, weight="bold")
out_spread = out_dir / f"spread_rate__{in_stem}__veh{n_vehicles}__{ts_str}.png"
plt.savefig(out_spread, dpi=300, bbox_inches="tight")
plt.close()

print(f"Kaydedildi:\n - {out_initial}\n - {out_spread}")

# === 3) MODEL SONUÇ HARİTALARI ===
# Sonuç dosyasını bul
cand = sorted(results_dir.glob("results_*.xlsx"))
if not cand:
    raise FileNotFoundError(f"results_*.xlsx bulunamadı: {results_dir}")
res_path = max(cand, key=lambda p: p.stat().st_mtime)
print(f"Kullanılan sonuç dosyası: {res_path}")

# Yardımcı okuma fonksiyonları
def read_sheet_lastcol(path, sheet, key="i"):
    df = pd.read_excel(path, sheet_name=sheet)
    val_col = df.columns[-1]
    return dict(zip(df[key], df[val_col]))

def read_x_sheet_sum(path):
    df = pd.read_excel(path, sheet_name="x")
    val_col = df.columns[-1]
    return df.groupby("i")[val_col].sum().to_dict()

def read_q_ones(path):
    df = pd.read_excel(path, sheet_name="q")
    val_col = df.columns[-1]
    df["is_one"] = (df[val_col] >= 0.5).astype(int)
    return df[df["is_one"]==1].groupby("i")["j"].apply(list).to_dict()

# Oku
p_map  = read_sheet_lastcol(res_path, "p")
ts_map = read_sheet_lastcol(res_path, "ts")
tm_map = read_sheet_lastcol(res_path, "tm")
tc_map = read_sheet_lastcol(res_path, "tc")
te_map = read_sheet_lastcol(res_path, "te")
sumx   = read_x_sheet_sum(res_path)
q_ones = read_q_ones(res_path)

# Grid oluştur
x_idx = {x:i for i,x in enumerate(x_vals)}
y_idx = {y:i for i,y in enumerate(y_vals)}
p_grid = np.full((nrows, ncols), np.nan)
nodeid_grid = np.full((nrows, ncols), np.nan)
for _, r in idf.iterrows():
    i = int(r["node_id"])
    xi = x_idx[r["x_coordinate"]]
    yi = y_idx[r["y_coordinate"]]
    nodeid_grid[yi, xi] = i
    p_grid[yi, xi] = p_map.get(i, np.nan)

finite_p = p_grid[np.isfinite(p_grid)]
vmin, vmax = (0.0, 1.0) if finite_p.size == 0 else (float(np.nanmin(finite_p)), float(np.nanmax(finite_p)))
norm = Normalize(vmin=vmin, vmax=vmax)
extent = [min(x_vals)-2.5, max(x_vals)+2.5, min(y_vals)-2.5, max(y_vals)+2.5]

def fnum(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "NA"
    return f"{v:.1f}"

def draw_Na_marks(ax, offset=1.0):
    for yi, y in enumerate(y_vals):
        for xi, x in enumerate(x_vals):
            nid = nodeid_grid[yi, xi]
            if np.isnan(nid): continue
            i = int(nid)
            if i in Na:
                ax.text(x, y + offset, "×", color="red",
                        ha="center", va="center", fontsize=14, fontweight="bold")

# --- p-map ---
fig1, ax1 = plt.subplots(figsize=(9,9))
im1 = ax1.imshow(p_grid, origin="lower", cmap="Greens", norm=norm,
                 extent=extent, interpolation="none")
cb1 = fig1.colorbar(im1, ax=ax1)
cb1.set_label("p_i")
ax1.set_title(f"p-map ({in_stem}, veh={n_vehicles})")
ax1.set_aspect("equal")
for yi, y in enumerate(y_vals):
    for xi, x in enumerate(x_vals):
        nid = nodeid_grid[yi, xi]
        if np.isnan(nid): continue
        i = int(nid)
        ax1.text(x, y, f"{i}\n{fnum(p_map.get(i))}",
                 ha="center", va="center", fontsize=7, color="black",
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"))
draw_Na_marks(ax1)
out_pmap = out_dir / f"p_map__{in_stem}__veh{n_vehicles}__{ts_str}.png"
fig1.savefig(out_pmap, dpi=240, bbox_inches="tight")
plt.close(fig1)

# --- detail-map ---
fig2, ax2 = plt.subplots(figsize=(10,10))
im2 = ax2.imshow(p_grid, origin="lower", cmap="Greens", norm=norm,
                 extent=extent, interpolation="none")
cb2 = fig2.colorbar(im2, ax=ax2)
cb2.set_label("p_i (shared scale)")
ax2.set_title(f"detail-map ({in_stem}, veh={n_vehicles})")
ax2.set_aspect("equal")
for yi, y in enumerate(y_vals):
    for xi, x in enumerate(x_vals):
        nid = nodeid_grid[yi, xi]
        if np.isnan(nid): continue
        i = int(nid)
        ts_i, tm_i, tc_i, te_i = ts_map.get(i, np.nan), tm_map.get(i, np.nan), tc_map.get(i, np.nan), te_map.get(i, np.nan)
        sx = sumx.get(i, 0.0)
        js = q_ones.get(i, [])
        q_text = ", ".join([f"q({i},{int(j)})=1" for j in js]) if js else "q(i,j)=∅"
        q_wrapped = "\n".join(textwrap.wrap(q_text, width=30))
        lines = [f"{i}",
                 f"({fnum(ts_i)},{fnum(tm_i)},{fnum(tc_i)},{fnum(te_i)})",
                 f"∑x_ik={sx:.2f}",
                 q_wrapped]
        ax2.text(x, y, "\n".join(lines),
                 ha="center", va="center", fontsize=6, color="black",
                 bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.2"))
draw_Na_marks(ax2)
out_detail = out_dir / f"detail_map__{in_stem}__veh{n_vehicles}__{ts_str}.png"
fig2.savefig(out_detail, dpi=240, bbox_inches="tight")
plt.close(fig2)

print(f"Kaydedildi:\n - {out_pmap}\n - {out_detail}")

