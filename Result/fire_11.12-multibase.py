# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB
from loader_wildfire_inputs import load_inputs, distribute_vehicles_to_bases,compute_d_param_multibase



# 1) Excel'i okuyun (sizin klasörünüz)
excel_path = r"C:\Users\Mert\deprem\wildfire\inputs_to_load-5x5_yyo.xlsx"

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
coords     = data["coords"]
su=data["su"]



Na = [i for i in N if state[i] == 1]
e = {i: (1 if i in Na else 0) for i in N}
BASES = {
    "A":    (0.0, 0.0),
    "B": (20.0, 10.0),
    "C":     (5.0, 25.0)
}

# 2. Araçları Üslere Otomatik Dağıt
vehicle_home = distribute_vehicles_to_bases(K, BASES)

# (Opsiyonel) Dağılımı Kontrol Etmek İçin Yazdır
print("--- Araç Dağılım Özeti ---")
from collections import Counter
summary = Counter(vehicle_home.values())
for base, count in summary.items():
    print(f"{base}: {count} adet araç")
print("--------------------------")

# 3. Mesafeleri Hesapla (d matrisi)
d = compute_d_param_multibase(
    coords=coords,
    speed_km_per_hour=120,
    K=K,
    base_locations=BASES,
    vehicle_assignments=vehicle_home
)
# mu_default  araç 230 km/saat iken 3 tur atıyor. 60 km/saat hıza düşersek 1 turu 1 saat 16dk 40 saniye olur. 1 tam tur atar diyelim. bu tam su kapasitesi yapsın. 120 km/saat dersek 1 turu 38.33 dk olur 2 tur varsayalım
mu_default = 10000  # birim: litre/saat
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

z_max  = m.addVars(N, K, vtype=GRB.BINARY,     name="z_max")

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
    excel_inputs_path=r"C:\Users\Mert\deprem\wildfire\inputs_to_load-5x5_yyo.xlsx",
    out_dir=r"C:\Users\Mert\deprem\wildfire\result",
    file_prefix="results"
)

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from matplotlib.colors import to_rgb, Normalize

# === AYARLAR: AKADEMİK RENK PALETİ ===
COLOR_BLACK = "#2c3e50"  # Never Controlled (Koyu Lacivert/Siyah)
COLOR_ORANGE = "#d35400"  # Controlled - Partial Loss (Kiremit)
COLOR_YELLOW = "#f1c40f"  # Controlled - No Loss (Amber)
COLOR_GREEN = "#27ae60"  # No Fire (Orman Yeşili)
COLOR_BASE = "#2980b9"  # Base Station (Mavi)

# === TANIYICILAR (BASES) ===
# Koordinatlar tam sayısal değerlerdir.
BASES = {
    "A": (0.0, 0.0),
    "B": (20.0, 10.0),
    "C": (5.0, 25.0)
}

# === YOLLAR ===
excel_inputs = r"C:\Users\Mert\deprem\wildfire\inputs_to_load-5x5_yyo.xlsx"
results_dir = Path(r"C:\Users\Mert\deprem\wildfire\result")
out_dir = Path(r"C:\Users\Mert\deprem\wildfire\maps")
out_dir.mkdir(parents=True, exist_ok=True)

# === SONUÇ DOSYASI SEÇİMİ ===
cand = sorted(results_dir.glob("results_*.xlsx"))
if not cand:
    raise FileNotFoundError(f"results_*.xlsx bulunamadı: {results_dir}")
res_path = max(cand, key=lambda p: p.stat().st_mtime)
print(f"Kullanılan sonuç dosyası: {res_path.name}")


# === OKUMA FONKSİYONLARI ===
def read_sheet(path, sheet, key="i"):
    try:
        df = pd.read_excel(path, sheet_name=sheet)
        val_col = df.columns[-1]
        return dict(zip(df[key], df[val_col]))
    except:
        return {}


def read_x_sum(path):
    try:
        df = pd.read_excel(path, sheet_name="x")
        return df.groupby("i")[df.columns[-1]].sum().to_dict()
    except:
        return {}


def read_v_max(path):
    try:
        df = pd.read_excel(path, sheet_name="v")
        return df.groupby("i")[df.columns[-1]].max().to_dict()
    except:
        return {}


def read_q(path):
    try:
        df = pd.read_excel(path, sheet_name="q")
        val = df.columns[-1]
        sub = df[df[val] >= 0.5]
        return sub.groupby("i")["j"].apply(list).to_dict()
    except:
        return {}


def read_objective_from_meta(path):
    try:
        df = pd.read_excel(path, sheet_name="meta")
        row = df[df.iloc[:, 0].astype(str).str.strip() == "Objective"]
        if not row.empty:
            val = row.iloc[0, 1]
            if isinstance(val, str): val = val.replace(',', '.')
            return float(val)
    except:
        pass
    return None


# === VERİLERİ YÜKLEME ===
idf = pd.read_excel(excel_inputs, sheet_name="inputs_df")
if "value_at_start" not in idf.columns: idf["value_at_start"] = 0
start_val_map = dict(zip(idf["node_id"], idf["value_at_start"]))

if "status" in idf.columns:
    Na_set = set(idf[idf["state"] == 1]["node_id"].astype(int))
else:
    Na_set = {3, 6, 9, 12, 15}

# === KOORDİNAT SİSTEMİ HESAPLAMALARI (CRITICAL FIX) ===
x_vals = sorted(idf["x_coordinate"].unique())
y_vals = sorted(idf["y_coordinate"].unique())

# Izgara aralığını bul (örn: 2.5, 7.5 ise aralık 5'tir)
dx = x_vals[1] - x_vals[0] if len(x_vals) > 1 else 5.0
dy = y_vals[1] - y_vals[0] if len(y_vals) > 1 else 5.0

# Extent: [x_min_edge, x_max_edge, y_min_edge, y_max_edge]
# Merkez noktaları (vals) kullanarak kenarları buluyoruz.
extent = [
    min(x_vals) - dx / 2,
    max(x_vals) + dx / 2,
    min(y_vals) - dy / 2,
    max(y_vals) + dy / 2
]

# Matris indekslemesi için map
coord_x_to_idx = {x: i for i, x in enumerate(x_vals)}
coord_y_to_idx = {y: i for i, y in enumerate(y_vals)}
nx, ny = len(x_vals), len(y_vals)

# Sonuç Değişkenleri
p_map = read_sheet(res_path, "p")
ts_map = read_sheet(res_path, "ts")
tm_map = read_sheet(res_path, "tm")
tc_map = read_sheet(res_path, "tc")
te_map = read_sheet(res_path, "te")
fire_status_map = read_sheet(res_path, "y")
sumx = read_x_sum(res_path)
max_v = read_v_max(res_path)
q_ones = read_q(res_path)
omega_map = read_sheet(res_path, "omega")
if not omega_map: omega_map = read_sheet(res_path, "w")
z_val = read_objective_from_meta(res_path)


def fnum(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "-"
    return f"{v:.1f}"


file_ts = datetime.now().strftime('%Y%m%d_%H%M%S')


# ==========================================
# HARİTA 1: INITIAL SITUATION
# ==========================================
def plot_initial_map():
    vals = [v for v in start_val_map.values() if not np.isnan(v)]
    vmin, vmax = (min(vals), max(vals)) if vals else (0, 100)
    cmap = plt.cm.Greens
    norm = Normalize(vmin=vmin, vmax=vmax)

    rgb_grid = np.zeros((ny, nx, 3))
    # Matrisi doldurma
    for _, row in idf.iterrows():
        nid = int(row["node_id"])
        xi = coord_x_to_idx[row["x_coordinate"]]
        yi = coord_y_to_idx[row["y_coordinate"]]
        s_val = start_val_map.get(nid, 0.0)
        rgb_grid[yi, xi] = cmap(norm(s_val))[:3]

    fig, ax = plt.subplots(figsize=(10, 10))

    # --- DÜZELTME: EXTENT KULLANIMI ---
    # Bu parametre sayesinde eksenler 0, 1, 2 değil; 0, 5, 10, 15... olur.
    ax.imshow(rgb_grid, origin="lower", aspect='equal', interpolation='nearest', extent=extent)

    # Grid (Gerçek koordinatlara göre)
    # Major ticks (0, 5, 10...) kenarlarda olmalı
    ax.set_xticks(np.arange(extent[0], extent[1] + 0.1, dx))
    ax.set_yticks(np.arange(extent[2], extent[3] + 0.1, dy))
    ax.grid(which='major', color='white', linestyle='-', linewidth=2)

    ax.set_xlabel("X Coordinate (km)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Y Coordinate (km)", fontsize=11, fontweight='bold')
    ax.set_title(f"Initial Situation (Value Gradient)\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 fontsize=16, fontweight='bold', pad=15)

    # Base Stationları Gerçek Koordinatlarına Çiz
    for label, (bx, by) in BASES.items():
        # zorder yüksek tutarak en üste çiziyoruz, clip_on=False ile sınırdaysa kesilmesini önlüyoruz
        ax.scatter(bx, by, s=350, marker='s', color=COLOR_BASE, edgecolors='white', linewidth=1.5, zorder=30,
                   clip_on=False)
        ax.text(bx, by, label, color='white', fontweight='bold', fontsize=10, ha='center', va='center', zorder=31,
                clip_on=False)

    # Hücre İçi Metinler (Merkez Koordinatlarını Kullanarak)
    for _, row in idf.iterrows():
        nid = int(row["node_id"])
        cx, cy = row["x_coordinate"], row["y_coordinate"]  # Hücre merkezi

        s_val = start_val_map.get(nid, 0.0)
        text_color = "white" if s_val > (vmax + vmin) / 2 else "black"

        # Kırmızı Çerçeve (Initial Fire)
        if nid in Na_set:
            # Dikdörtgen sol alt köşeden başlar: Merkez - dx/2
            rect = mpatches.Rectangle((cx - dx / 2 + 0.1, cy - dy / 2 + 0.1), dx - 0.2, dy - 0.2,
                                      fill=False, edgecolor='red', linewidth=3, zorder=20)
            ax.add_patch(rect)

        ax.text(cx, cy + dy * 0.15, str(nid), ha='center', va='center', fontsize=12, color=text_color,
                fontweight='bold', zorder=15)
        ax.text(cx, cy - dy * 0.15, f"Val: {fnum(s_val)}", ha='center', va='center', fontsize=9, color=text_color,
                zorder=15)

    patches = [
        mpatches.Patch(facecolor='none', edgecolor='red', linewidth=2, label='Initial Fire Source'),
        mpatches.Patch(color=COLOR_BASE, label='Base Station')
    ]
    ax.legend(handles=patches, loc='upper left', fontsize=10, frameon=True)

    plt.tight_layout()
    save_path = out_dir / f"initial_situation_{file_ts}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Initial Map kaydedildi: {save_path}")


# ==========================================
# HARİTA 2: DETAIL MAP (AKADEMİK)
# ==========================================
def plot_detail_map():
    rgb_grid = np.zeros((ny, nx, 3))

    # Renkleri Doldur
    for _, row in idf.iterrows():
        nid = int(row["node_id"])
        xi = coord_x_to_idx[row["x_coordinate"]]
        yi = coord_y_to_idx[row["y_coordinate"]]

        p_val = p_map.get(nid, 0.0)
        s_val = start_val_map.get(nid, 0.0)
        y_val = fire_status_map.get(nid, 0.0)

        if p_val <= 0.001:
            color = COLOR_BLACK
        elif p_val < (s_val - 0.001):
            color = COLOR_ORANGE
        elif y_val >= 0.5:
            color = COLOR_YELLOW
        else:
            color = COLOR_GREEN
        rgb_grid[yi, xi] = to_rgb(color)

    fig, ax = plt.subplots(figsize=(10, 10))
    # EXTENT kullanarak gerçek koordinatlara oturtuyoruz
    ax.imshow(rgb_grid, origin="lower", aspect='equal', interpolation='nearest', extent=extent)

    # Grid
    ax.set_xticks(np.arange(extent[0], extent[1] + 0.1, dx))
    ax.set_yticks(np.arange(extent[2], extent[3] + 0.1, dy))
    ax.grid(which='major', color='white', linestyle='-', linewidth=2)

    ax.set_xlabel("X Coordinate (km)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Y Coordinate (km)", fontsize=11, fontweight='bold')

    z_str = f"{z_val:,.2f}" if z_val is not None else "N/A"
    title_str = f"Optimization Result\nObjective Value (Z): {z_str}\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ax.set_title(title_str, fontsize=16, fontweight='bold', pad=15)

    # Base Stationları Çiz (Gerçek Koordinat)
    for label, (bx, by) in BASES.items():
        ax.scatter(bx, by, s=350, marker='s', color=COLOR_BASE, edgecolors='white', linewidth=1.5, zorder=30,
                   clip_on=False)
        ax.text(bx, by, label, color='white', fontweight='bold', fontsize=10, ha='center', va='center', zorder=31,
                clip_on=False)

    # Hücre İçi Metinler (Merkez Koordinat)
    for _, row in idf.iterrows():
        nid = int(row["node_id"])
        cx, cy = row["x_coordinate"], row["y_coordinate"]

        p_val = p_map.get(nid, 0.0)
        s_val = start_val_map.get(nid, 0.0)
        y_val = fire_status_map.get(nid, 0.0)

        # Text Rengi
        if p_val <= 0.001:
            text_color = "white"
        elif p_val < (s_val - 0.001):
            text_color = "white"
        elif y_val >= 0.5:
            text_color = "black"
        else:
            text_color = "white"

        # Kırmızı Çerçeve
        if nid in Na_set:
            rect = mpatches.Rectangle((cx - dx / 2 + 0.1, cy - dy / 2 + 0.1), dx - 0.2, dy - 0.2,
                                      fill=False, edgecolor='red', linewidth=3, zorder=20)
            ax.add_patch(rect)

        # ID
        ax.text(cx, cy + dy * 0.25, str(nid), ha='center', va='center',
                fontsize=12, color=text_color, fontweight='bold', zorder=15)

        # İstatistikler
        lines = []
        if y_val >= 0.5:
            ts_i = ts_map.get(nid, np.nan)
            tm_i = tm_map.get(nid, np.nan)
            tc_i = tc_map.get(nid, np.nan)
            te_i = te_map.get(nid, np.nan)
            sx = sumx.get(nid, 0.0)
            mv = max_v.get(nid, np.nan)
            om = omega_map.get(nid, np.nan)
            js = q_ones.get(nid, [])

            lines.append(f"Val: {fnum(s_val)}→{fnum(p_val)}")
            lines.append(f"T: ({fnum(ts_i)},{fnum(tm_i)},{fnum(te_i)})→{fnum(tc_i)}")
            lines.append(f"Water: {fnum(om)}")
            lines.append(f"Team: ∑x={sx:.1f} | max(v)={fnum(mv)}")

            if js:
                j_str = ",".join([str(int(j)) for j in js])
                lines.append(f"Ignite: {j_str[:10]}..." if len(j_str) > 12 else f"Ignite: {j_str}")
            else:
                lines.append("Ignite: -")
        else:
            lines.append(f"Val: {fnum(s_val)}")

        full_text = "\n".join(lines)
        ax.text(cx, cy - dy * 0.1, full_text, ha='center', va='center',
                fontsize=6, color=text_color, fontweight='normal', linespacing=1.2, zorder=15)

    patches = [
        mpatches.Patch(color=COLOR_BLACK, label='Never Controlled'),
        mpatches.Patch(color=COLOR_ORANGE, label='Controlled (Partial Loss)'),
        mpatches.Patch(color=COLOR_YELLOW, label='Controlled (No Loss)'),
        mpatches.Patch(color=COLOR_GREEN, label='No Fire'),
        mpatches.Patch(facecolor='none', edgecolor='red', linewidth=2, label='Initial Fire Source'),
        mpatches.Patch(color=COLOR_BASE, label='Base Station')
    ]
    ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.08),
              fancybox=True, shadow=False, ncol=6, fontsize=10, frameon=False)

    plt.tight_layout()
    save_path = out_dir / f"detail_map_{file_ts}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Detail Map kaydedildi: {save_path}")


if __name__ == "__main__":
    plot_initial_map()
    plot_detail_map()
