# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB
from loader_wildfire_inputs import load_inputs, compute_d_param
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

# 1) Excel'i okuyun (sizin klasörünüz)
excel_path = r"C:\Users\baltunoglu\PycharmProjects\fire\inputs_to_load-5x5_yyo.xlsx"

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
alpha = data["P"]["region_side_length"]

su=data["su"]

d = compute_d_param(coords, speed_km_per_min=3.83, K=K)

Na = [i for i in N if i in {3,6,9,12,15}]
e = {i: (1 if i in Na else 0) for i in N}

# omega = {i: 581152 for i in N}  #km2 başına 36322 litre 16 km2 alan için 581152 litre su gerekli
mu_default = 1667  # birim: litre/dakika
mu = {k: mu_default for k in K}

M = 1e6
OMEGA_MAX =908050
epsilon = 1e-12
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
omega = m.addVars(N,  vtype=GRB.CONTINUOUS,lb=0.0, name="omega")  # w_i ≥ 0 :the required water for node i
v=m.addVars(N,K, vtype=GRB.CONTINUOUS, lb=0, name="v")  # v_ik ≥ 0 :arrival time of team k at node i
a = m.addVars(N,  vtype=GRB.CONTINUOUS, name="a")   #ilk varış zamanı = min_k v[i,k] (yalnızca atanmış k'lar arasında)

m.addConstrs(( p[i] <= pi[i] -beta[i]*(tc[i]-ts[i]) for i in N),
    name="cons 2")

m.addConstrs(( p[i] <= pi[i]*(u_pre[i]+u_post[i]+1-y[i]) for i in N),
    name="cons 3")


delta = m.addVars(N, K, vtype=GRB.CONTINUOUS, lb=0.0, name="delta")



for i in N:
    for k in K:
        # x[i,k] = 1 iken: delta[i,k] = su[i]*(v[i,k] - ts[i])
        # x[i,k] = 0 iken: delta[i,k] = 0

        m.addConstr(
            delta[i, k] >= su[i] * (v[i, k] - ts[i]) - M * (1 - x[i, k]),
            name=f"delta_lo[{i},{k}]"
        )
#         m.addConstr(
#             delta[i, k] <= su[i] * (v[i, k] - ts[i]) + M * (1 - x[i, k]),
#             name=f"delta_up[{i},{k}]"
#         )
        # # Atama yoksa (x=0) delta[i,k] <= 0, lb=0 olduğu için delta[i,k]=0
        # m.addConstr(
        #     delta[i, k] <= OMEGA_MAX * x[i, k],
        #     name=f"delta_cap[{i},{k}]"
        # )

# OMEGA_MAX zaten yukarıda omega ve delta için kullandığınız güvenli üst sınır

# # omega_i = max_k delta_{i,k} (sadece atanmış ekiplerin etkisi)
# for i in N:
#     m.addGenConstrMax(
#         omega[i],
#         [delta[i, k] for k in K],
#         name=f"omega_max[{i}]"
#     )
z_max  = m.addVars(N, K, vtype=GRB.BINARY,     name="z_max")


# --- 2) omega[i] = max_{k:x[i,k]=1} delta[i,k] ---

for i in N:
    for k in K:
        # (a) omega[i] tüm delta[i,k]'lerin ÜSTÜNDE olsun
        #     (atanmış olmayan ekipler için delta = 0 zaten)
        m.addConstr(
            omega[i] >= delta[i, k],
            name=f"omega_ge_delta[{i},{k}]"
        )

        # (b) z_max[i,k] = 1 ise: omega[i] ≤ delta[i,k]
        m.addConstr(
            omega[i] <= delta[i, k] + OMEGA_MAX * (1 - z_max[i, k]),
            name=f"omega_le_delta_plusM[{i},{k}]"
        )

        # (c) z_max sadece atanmış ekipler arasından seçilebilir
        m.addConstr(
            z_max[i, k] <= x[i, k],
            name=f"zmax_le_x[{i},{k}]"
        )

    # (d) Kontrol fazı aktifse (u_pre+u_post = 1) tam BİR ekip seç,
    #     aktif değilse (0) hiç ekip seçme.
    m.addConstr(
        gp.quicksum(z_max[i, k] for k in K) == u_pre[i] + u_post[i],
        name=f"zmax_sum[{i}]"
    )



# A) su talebi – u_pre/u_post’a bağla
for i in N:
    m.addGenConstrIndicator(u_pre[i], 1,
        gp.quicksum(mu[k]*s[i,k] for k in K) >= omega[i], name=f"dem_suppress_pre[{i}]")
    m.addGenConstrIndicator(u_post[i], 1,
        gp.quicksum(mu[k]*s[i,k] for k in K) >= omega[i], name=f"dem_suppress_post[{i}]")
    m.addConstrs(
        (s[i, k] <= M * (u_pre[i] + u_post[i]) for k in K),
        name=f"no_water_if_no_control[{i}]"
    )

for i in N:
    for k in K:
        # x[i,k] = 1 ise: v[i,k] = t[i,k] + d[i,k]
        m.addConstr(
            v[i, k] >= t[i, k] + d[i, k] - M * (1 - x[i, k]),
            name=f"v_eq_td_lo[{i},{k}]"
        )
        m.addConstr(
            v[i, k] <= t[i, k] + d[i, k] + M * (1 - x[i, k]),
            name=f"v_eq_td_up[{i},{k}]"
        )


m.addConstrs((t[i,k]+s[i,k] +v[i,k]<=M*x[i,k] for i in N for k in K),
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

m.addConstrs((tc[i]>=a[i] - M*(1-u_post[i]-u_pre[i]) for i in N ),
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



m.addConstrs(( tm[i] == ts[i] + alpha / degrade[i] for i in N ),
    name="cons 26")


m.addConstrs(( te[i] == tm[i] + alpha / ameliorate[i] for i in N ),
    name="cons 27")

# objective function
m.setObjective(gp.quicksum(p[i] for i in N), GRB.MAXIMIZE)

# m.setParam("MIPGap", 0.00925)
# --- Optimize et ---
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
                           out_dir=r"C:\Users\baltunoglu\PycharmProjects\fire\result",
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
            # >>> EKLE: omega_i
            "omega": ("omega", "i", "omega_i"),
            "a": ("a", "i", "a_i"),
        }.items():
            if name in vars_dict and vars_dict[name] is not None:
                df = _single_index_to_df(vars_dict[name], idx, colname)
                _safe_write(writer, sheet, df)

        # --- Çift indeksli değişkenler (varsa yazar) ---
        for name, (sheet, idx_names, colname) in {
            "x": ("x", ("i", "k"), "x_ik"),
            "t": ("t", ("i", "k"), "t_ik"),
            "s": ("s", ("i", "k"), "s_ik"),
            "z": ("z", ("i", "j"), "z_ij"),
            "q": ("q", ("i", "j"), "q_ij"),
            "v": ("v", ("i", "k"), "v_ik"),
            "delta": ("delta", ("i", "k"), "delta_ik"),
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
    # >>> EKLE
    "omega": omega,   # tek indeksli: i ↦ omega_i
    "v": v,           # çift indeksli: (i,k) ↦ v_ik
    "a": a,
    "delta": delta,   # çift indeksli: (i,k) ↦ delta_ik

    # "a": a, "b": b, "c": c,
}

out_file = write_results_to_excel(
    model=m,
    vars_dict=vars_dict,
    excel_inputs_path=r"C:\Users\baltunoglu\PycharmProjects\fire\inputs_to_load-5x5_yyo.xlsx",
    out_dir=r"C:\Users\baltunoglu\PycharmProjects\fire\result",
    file_prefix="results"
)



# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Excel dosya yolu
excel_path = r"C:\Users\baltunoglu\PycharmProjects\fire\inputs_to_load-5x5_yyo.xlsx"

# Çıktı klasörü
out_dir = Path("maps")
out_dir.mkdir(parents=True, exist_ok=True)

# Veriyi oku
df = pd.read_excel(excel_path, sheet_name="inputs_df")

# Grid boyutu
nrows, ncols = 5,5

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
excel_inputs = r"C:\Users\baltunoglu\PycharmProjects\fire\inputs_to_load-5x5_yyo.xlsx"
results_dir = Path(r"C:\Users\baltunoglu\PycharmProjects\fire\result") # sonuç xlsx'lerin olduğu klasör
out_dir      = Path(r"C:\Users\baltunoglu\PycharmProjects\fire\maps")
out_dir.mkdir(parents=True, exist_ok=True)

# === BAŞLANGIÇ YANGINLARI (Na) ===
# İsterseniz Excel'den de çekebiliriz; şimdilik açıkça veriyoruz:
# Na = {8, 11, 20}

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
extent = [min(x_vals)-2, max(x_vals)+2, min(y_vals)-2, max(y_vals)+2]

# === YARDIMCI OKUMA (sonuç sayfaları) ===
def read_sheet_lastcol(path, sheet, key="i"):
    """Sayfayı oku; son kolonu değer olarak döndür."""
    df = pd.read_excel(path, sheet_name=sheet)
    val_col = df.columns[-1]
    return dict(zip(df[key], df[val_col]))
def read_first_available(path, sheet_candidates, key="i"):
    """Verilen sayfa adları listesinden ilk bulunanı okuyup {i: son_kolon} döndürür."""
    for sh in sheet_candidates:
        try:
            df = pd.read_excel(path, sheet_name=sh)
            val_col = df.columns[-1]
            return dict(zip(df[key], df[val_col]))
        except Exception:
            continue
    # hiçbiri yoksa boş sözlük döndür
    return {}

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
# --- ek okuma: omega ve alpha (a) ---
omega_map = read_first_available(res_path, ["omega", "w", "omega_i"], key="i")
a_map     = read_first_available(res_path, ["a", "alpha", "alpha_i"], key="i")

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
        omega_i = omega_map.get(i, np.nan)
        a_i = a_map.get(i, np.nan)

        # q(i,j)=1 listesi (kırma)
        if js:
            q_items = [f"q({i},{int(j)})=1" for j in js]
            q_text = ", ".join(q_items)
            q_text_wrapped = "\n".join(textwrap.wrap(q_text, width=32))
        else:
            q_text_wrapped = "q(i,j)=∅"

        lines = [
            f"{i}",
            f"({fnum(ts_i)},{fnum(tm_i)},{fnum(tc_i)},{fnum(te_i)})",
            f"ω_i={fnum(omega_i)}, α_i={fnum(a_i)}",  # <-- EKLENDİ
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
