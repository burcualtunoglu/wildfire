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
omega = {i: 25000 for i in N}  #25 km2 alan için 250000000 litre su gerekli
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

m.addConstrs((tc[i]<=te[i] for i in N ),
    name="cons 11")

m.addConstrs((tc[i]>=ts[i] - M*(1-u_post[i]-u_pre[i]) for i in N ),
    name="cons 12")

m.addConstrs((tc[i]<=M*(u_post[i]-u_pre[i]) for i in N ),
    name="cons 13")

m.addConstrs((tc[i]<= tm[i] +epsilon+ M *(1-u_pre[i]) for i in N ),
    name="cons 14")

m.addConstrs((tc[i]>=tm[i]  - M *(1-u_post[i]) for i in N ),
    name="cons 15")

m.addConstrs((u_pre[i]+u_post[i] <=1 for i in N ),
    name="cons 16")
m.addConstrs((gp.quicksum(z[i, j] for j in neighbors[i]) == len(neighbors[i]) * (y[i] - u_pre[i]) for i in N),
                 name="cons 17")

m.addConstrs((len(neighbors[j]) * y[j] >= gp.quicksum(z[i, j] for i in neighbors[j]) for j in N),
                 name="cons 18")

m.addConstrs((q[i,j]<=z[i,j] for i in N for j in neighbors[i]),
                 name="cons 19")

m.addConstrs((gp.quicksum(q[i, j] for i in neighbors[j]) == y[j]-e[j] for j in N ),
                 name="cons 20")


m.addConstrs((ts[j]==gp.quicksum(tm[i]*q[i,j] for i in neighbors[j])  for j in N ),
                 name="cons 21")

m.addConstrs((ts[j]<= tm[i]+ M*(1-z[i,j]) for j in N for i in neighbors[j] ),
                 name="cons 22")

m.addConstr(gp.quicksum(y[i] for i in Na) == len(Na),
    name="cons 23")

m.addConstr( gp.quicksum(ts[i] for i in Na) == 0,
    name="cons 24")



m.addConstrs(( tm[i] == ts[i] + alpha[i] / degrade[i] for i in N ),
    name="cons 25")


m.addConstrs(( te[i] == tm[i] + alpha[i] / ameliorate[i] for i in N ),
    name="cons 26")
import math
# R_i hesaplama: gerekli iş yükü / araç kapasitesi: gereken araç sayısı
R = {}
max_mu = 5000  # tüm araçların kapasitesi aynı ise bu yeterli
for i in N:
    R[i] = math.ceil(omega[i] / max_mu)

# Kısıtı ekle
m.addConstrs((gp.quicksum(x[i,k] for k in K) <= R[i] for i in N),
             name="node_cap")


# objective function
m.setObjective(gp.quicksum(p[i] for i in N), GRB.MAXIMIZE)

# m.update()
# m.optimize()


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

def write_results_to_excel(model: gp.Model,
                           vars_dict: dict,
                           out_dir=r"C:\Users\Mert\deprem\wildfire\result",
                           file_prefix="results"):
    """
    Model sonuçlarını Excel'e yazar.
    vars_dict = {
        "p": p, "y": y, "ts": ts, "tm": tm, "te": te, "tc": tc,
        "u_pre": u_pre, "u_post": u_post,
        "x": x, "t": t, "s": s, "z": z, "q": q,
        "a": a, "b": b, "c": c
    }
    """
    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        raise RuntimeError(f"Çözülebilir bir çözüm bulunamadı (Status={model.Status}).")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    xlsx = out_path / f"{file_prefix}_{ts_str}.xlsx"

    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
        # Meta sayfası
        meta = pd.DataFrame({
            "metric": ["Status", "Objective", "MIPGap"],
            "value":  [model.Status,
                       getattr(model, "ObjVal", float("nan")),
                       getattr(model, "MIPGap", float("nan"))]
        })
        meta.to_excel(writer, index=False, sheet_name="meta")

        # Tek indeksli değişkenler
        _single_index_to_df(vars_dict["p"],  "i", "p_i").to_excel(writer, index=False, sheet_name="p")
        _single_index_to_df(vars_dict["y"],  "i", "y_i").to_excel(writer, index=False, sheet_name="y")
        _single_index_to_df(vars_dict["ts"], "i", "t_i_s").to_excel(writer, index=False, sheet_name="ts")
        _single_index_to_df(vars_dict["tm"], "i", "t_i_m").to_excel(writer, index=False, sheet_name="tm")
        _single_index_to_df(vars_dict["te"], "i", "t_i_e").to_excel(writer, index=False, sheet_name="te")
        _single_index_to_df(vars_dict["tc"], "i", "t_i_c").to_excel(writer, index=False, sheet_name="tc")
        _single_index_to_df(vars_dict["u_pre"],  "i", "u_i_pre").to_excel(writer, index=False, sheet_name="u_pre")
        _single_index_to_df(vars_dict["u_post"], "i", "u_i_post").to_excel(writer, index=False, sheet_name="u_post")

        # Çift indeksli değişkenler
        _double_index_to_df(vars_dict["x"], ("i","k"), "x_ik").to_excel(writer, index=False, sheet_name="x")
        _double_index_to_df(vars_dict["t"], ("i","k"), "t_ik").to_excel(writer, index=False, sheet_name="t")
        _double_index_to_df(vars_dict["s"], ("i","k"), "s_ik").to_excel(writer, index=False, sheet_name="s")
        _double_index_to_df(vars_dict["z"], ("i","j"), "z_ij").to_excel(writer, index=False, sheet_name="z")
        _double_index_to_df(vars_dict["q"], ("i","j"), "q_ij").to_excel(writer, index=False, sheet_name="q")



    print(f"Sonuçlar Excel'e yazıldı: {xlsx}")
    return str(xlsx)





# Kullanım:
vars_dict = {
    "p": p, "y": y, "ts": ts, "tm": tm, "te": te, "tc": tc,
    "u_pre": u_pre, "u_post": u_post,
    "x": x, "t": t, "s": s, "z": z, "q": q
}

m.optimize()

for k in K:
    tot = sum(x[i,k].X for i in N)
    print(f"Vehicle {k}: sum_i x[i,k] = {tot:.6f}")
out_file = write_results_to_excel(m, vars_dict)


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
from matplotlib.colors import Normalize
from pathlib import Path
from datetime import datetime
import textwrap

# === YOLLAR ===
excel_inputs = r"C:\Users\Mert\deprem\wildfire\inputs_to_load-5x5_burcu.xlsx"
results_dir  = Path(r"C:\Users\Mert\deprem\wildfire\result")  # sonuç xlsx'lerin olduğu klasör
out_dir      = Path(r"C:\Users\Mert\deprem\wildfire\maps")
out_dir.mkdir(parents=True, exist_ok=True)

# === BAŞLANGIÇ YANGINLARI (Na) ===
# İsterseniz Excel'den de çekebiliriz; şimdilik açıkça veriyoruz:
Na = {8, 11, 20}

# === GÜNCEL SONUÇ DOSYASINI BUL ===
cand = sorted(results_dir.glob("results_*.xlsx"))
if not cand:
    raise FileNotFoundError(f"results_*.xlsx bulunamadı: {results_dir}")
res_path = max(cand, key=lambda p: p.stat().st_mtime)  # en yeni dosya
print(f"Kullanılan sonuç dosyası: {res_path}")

# === GİRDİLER: koordinatlar ve node_id ===
idf = pd.read_excel(excel_inputs, sheet_name="inputs_df")
need_cols = ["node_id","x_coordinate","y_coordinate"]
missing = [c for c in need_cols if c not in idf.columns]
if missing:
    raise KeyError(f"inputs_df sayfasında eksik kolon(lar): {missing}")

x_vals = sorted(idf["x_coordinate"].unique())
y_vals = sorted(idf["y_coordinate"].unique())
x_idx  = {x:i for i,x in enumerate(x_vals)}
y_idx  = {y:i for i,y in enumerate(y_vals)}

# 5 km kareler için extent (hücre merkezine ±2.5 km)
extent = [min(x_vals)-2.5, max(x_vals)+2.5, min(y_vals)-2.5, max(y_vals)+2.5]

# === YARDIMCI OKUMA (sonuç sayfaları) ===
def read_sheet_lastcol(path, sheet, key="i"):
    """Sayfayı oku; son kolonu değer olarak döndür."""
    df = pd.read_excel(path, sheet_name=sheet)
    val_col = df.columns[-1]
    return dict(zip(df[key], df[val_col]))

def read_x_sheet_sum(path):
    """x sayfasından ∑_k x_{ik} değerini döndür (float toplam)."""
    df = pd.read_excel(path, sheet_name="x")  # i, k, x_ik
    val_col = df.columns[-1]
    s = df.groupby("i")[val_col].sum()
    return s.to_dict()

def read_q_ones(path):
    """q sayfasından q(i,j)=1 olan j listelerini i'ye göre döndür (eşik 0.5)."""
    df = pd.read_excel(path, sheet_name="q")  # i, j, q_ij
    val_col = df.columns[-1]
    df["is_one"] = (df[val_col] >= 0.5).astype(int)
    sub = df[df["is_one"]==1]
    return sub.groupby("i")["j"].apply(list).to_dict()

# === SONUÇLARI OKU ===
p_map  = read_sheet_lastcol(res_path, "p",  key="i")   # p_i
ts_map = read_sheet_lastcol(res_path, "ts", key="i")   # t_i^s
tm_map = read_sheet_lastcol(res_path, "tm", key="i")   # t_i^m
tc_map = read_sheet_lastcol(res_path, "tc", key="i")   # t_i^c
te_map = read_sheet_lastcol(res_path, "te", key="i")   # t_i^e
sumx   = read_x_sheet_sum(res_path)                    # ∑_k x_{ik}
q_ones = read_q_ones(res_path)                         # {i: [j,...]}

# === GRIDLERİ OLUŞTUR (p için) ===
ny, nx = len(y_vals), len(x_vals)
p_grid       = np.full((ny, nx), np.nan)
nodeid_grid  = np.full((ny, nx), np.nan)

for _, r in idf.iterrows():
    i  = int(r["node_id"])
    xi = x_idx[r["x_coordinate"]]
    yi = y_idx[r["y_coordinate"]]
    nodeid_grid[yi, xi] = i
    p_grid[yi, xi]      = p_map.get(i, np.nan)

# P ortak renk skalası
finite_p = p_grid[np.isfinite(p_grid)]
if finite_p.size == 0:
    vmin, vmax = 0.0, 1.0
else:
    vmin, vmax = float(np.nanmin(finite_p)), float(np.nanmax(finite_p))
norm = Normalize(vmin=vmin, vmax=vmax)

# Zamanları yazarken biçim
def fnum(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    return f"{v:.1f}"

# Hücre merkezlerine “X” (Na) işareti basan yardımcı
def draw_Na_marks(ax, offset=1.0):
    """Na kümesindeki hücrelerin üst tarafına kırmızı çarpı ekler."""
    for yi, y in enumerate(y_vals):
        for xi, x in enumerate(x_vals):
            nid = nodeid_grid[yi, xi]
            if np.isnan(nid):
                continue
            i = int(nid)
            if i in Na:
                ax.text(x, y + offset, "×",
                        color="red", ha="center", va="center",
                        fontsize=14, fontweight="bold")


# === HARİTA 1: p-map ===
ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
fig1, ax1 = plt.subplots(figsize=(9,9))
im1 = ax1.imshow(p_grid, origin="lower", cmap="Greens", norm=norm,
                 extent=extent, interpolation="none")
cb1 = fig1.colorbar(im1, ax=ax1)
cb1.set_label("p_i")

ax1.set_title("p-map (node id & p_i)")
ax1.set_xlabel("X (km)")
ax1.set_ylabel("Y (km)")
ax1.set_aspect('equal')

# Etiketler: node_id ve p_i
for yi, y in enumerate(y_vals):
    for xi, x in enumerate(x_vals):
        nid = nodeid_grid[yi, xi]
        if np.isnan(nid):
            continue
        i = int(nid)
        pval = p_map.get(i, np.nan)
        txt  = f"{i}\n{fnum(pval)}"
        ax1.text(x, y, txt, ha="center", va="center", fontsize=7, color="black",
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"))

# Na işaretleri
draw_Na_marks(ax1)

out1 = out_dir / f"p_map_{ts_str}.png"
fig1.tight_layout()
fig1.savefig(out1, dpi=220)
plt.close(fig1)

# === HARİTA 2: detay-map (aynı renk sklası, p ile boyalı) ===
fig2, ax2 = plt.subplots(figsize=(10,10))
im2 = ax2.imshow(p_grid, origin="lower", cmap="Greens", norm=norm,
                 extent=extent, interpolation="none")
cb2 = fig2.colorbar(im2, ax=ax2)
cb2.set_label("p_i (shared scale)")

ax2.set_title("detail-map (node, (ts,tm,tc,te), ∑k x_ik, q(i,j)=1)")
ax2.set_xlabel("X (km)")
ax2.set_ylabel("Y (km)")
ax2.set_aspect('equal')

for yi, y in enumerate(y_vals):
    for xi, x in enumerate(x_vals):
        nid = nodeid_grid[yi, xi]
        if np.isnan(nid):
            continue
        i = int(nid)
        ts_i = ts_map.get(i, np.nan)
        tm_i = tm_map.get(i, np.nan)
        tc_i = tc_map.get(i, np.nan)
        te_i = te_map.get(i, np.nan)
        sx   = sumx.get(i, 0.0)
        js   = q_ones.get(i, [])

        # q(i,j)=1 listesi
        if js:
            q_items = [f"q({i},{int(j)})=1" for j in js]
            q_text  = ", ".join(q_items)
            q_text_wrapped = "\n".join(textwrap.wrap(q_text, width=32))
        else:
            q_text_wrapped = "q(i,j)=∅"

        lines = [
            f"{i}",
            f"({fnum(ts_i)},{fnum(tm_i)},{fnum(tc_i)},{fnum(te_i)})",
            f"∑ x_ik = {sx:.2f}",
            q_text_wrapped
        ]
        txt = "\n".join(lines)
        ax2.text(x, y, txt, ha="center", va="center", fontsize=6, color="black",
                 bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.2"))

# Na işaretleri
draw_Na_marks(ax2)

out2 = out_dir / f"detail_map_{ts_str}.png"
fig2.tight_layout()
fig2.savefig(out2, dpi=220)
plt.close(fig2)

print(f"Kaydedildi:\n - {out1}\n - {out2}")
