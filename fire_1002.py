# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB
from load_inputs import load_wildfire_data_final
import pandas as pd
from pathlib import Path
BASE_DIR = Path(r"C:\Users\Mert\deprem\wildfire")
# BASE_DIR = Path(r"C:\Users\Mert\deprem\wildfire")
EXCEL_PATH = BASE_DIR / "inputs" / "inputs_7x7.xlsx"
RESULT_DIR=BASE_DIR / "result"
OUT_DIR=BASE_DIR / "maps"

try:
    data = load_wildfire_data_final(EXCEL_PATH)
except Exception as e:
    print(f"Veri yüklenirken hata oluştu: {e}")
    raise


# --- 2. PARAMETRELERİN MODELLENMESİ ---
N          = data["N"]
K          = data["K"]
state      = data["state"]    # Excel'deki state sütunu (0 veya 1)
v0         = data["v0"]       # Yangın yükü
mu         = data["mu"]       # Araç kapasiteleri
degrade    = data["degrade"]     # Yayılma hızı
ameliorate = data["ameliorate"]  # İyileşme/Sönümleme hızı
su         = data["su"]          # Su ihtiyacı / Kaynak gereksinimi
beta       = data["beta"]        # Önem katsayısı
d          = gp.tupledict(data["d_matrix"])
neighbors = data["neighbors"]
P          = data["P"]           # Global parametreler sözlüğü
alpha = P.get("region_side_length", 1.0)
# DOĞRU YAKLAŞIM: Na kümesi state sütununa göre belirlenir
# state[i] == 1 olan düğümler aktif yangın bölgeleridir (Na)
Na = [i for i in N if state[i] == 1]
# Su kaynakları hücreleri
Ns = [i for i in N if state[i] == 2]

# Yanabilir hücreler (0 ve 1)
Nf = [i for i in N if state[i] in (0, 1)]

# Modelde kullanılacak parametrik e_i (indicative) değerleri
# i düğümü aktif yangın bölgesindeyse 1, değilse 0
e = {i: (1 if i in Na else 0) for i in N}

M = 1e6

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

m.addConstrs(( p[i] <= v0[i] -beta[i]*(tc[i]-ts[i]) for i in Nf),
    name="cons 2")

m.addConstrs(( p[i] <= v0[i]*(u_pre[i]+u_post[i]+1-y[i]) for i in Nf),
    name="cons 3")


delta = m.addVars(N, K, vtype=GRB.CONTINUOUS, lb=0.0, name="delta")



for i in Nf:
    for k in K:
               m.addConstr(
            delta[i, k] >= su[i] * (v[i, k] - ts[i]) - M * (1 - x[i, k]),
            name=f"delta_lo[{i},{k}]")

# Yeni değişkenler
omega_max = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0.0, name="omega_max")
omega_min = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0.0, name="omega_min")
omega_avg = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0.0, name="omega_avg")

z_max = m.addVars(N, K, vtype=GRB.BINARY, name="z_max")
z_min = m.addVars(N, K, vtype=GRB.BINARY, name="z_min")

for i in Nf:
    u = u_pre[i] + u_post[i]  # faz aktivasyonu (0/1)

    for k in K:
        # ---------- MAX ----------
        m.addConstr(omega_max[i] >= delta[i,k], name=f"omax_ge_delta[{i},{k}]")
        m.addConstr(omega_max[i] <= delta[i,k] + M*(1 - z_max[i,k]),
                    name=f"omax_le_delta_sel[{i},{k}]")
        m.addConstr(z_max[i,k] <= x[i,k], name=f"zmax_le_x[{i},{k}]")

        # ---------- MIN (atanmayanı dışla) ----------
        m.addConstr(omega_min[i] <= delta[i,k] ,
                    name=f"omin_le_delta_if_assigned[{i},{k}]")
        m.addConstr(omega_min[i] >= delta[i,k] - M*(1 - z_min[i,k]),
                    name=f"omin_ge_delta_sel[{i},{k}]")
        m.addConstr(z_min[i,k] <= x[i,k], name=f"zmin_le_x[{i},{k}]")

    # Seçim sayıları (faz aktifse 1, değilse 0)
    m.addConstr(gp.quicksum(z_max[i,k] for k in K) == u, name=f"zmax_sum[{i}]")
    m.addConstr(gp.quicksum(z_min[i,k] for k in K) == u, name=f"zmin_sum[{i}]")

    # Ortalama (min-max midpoint)
    m.addConstr(2*omega[i] == omega_max[i] + omega_min[i], name=f"oavg_def[{i}]")

    # Faz kapalıyken sıfıra zorlamak (opsiyonel ama temiz)
    m.addConstr(omega_max[i] <=M*u, name=f"omax_off[{i}]")
    m.addConstr(omega_min[i] <= M*u, name=f"omin_off[{i}]")
    m.addConstr(omega[i] <= M*u, name=f"oavg_off[{i}]")



# A) su talebi – u_pre/u_post’a bağla
for i in Nf:
    m.addConstr(
        gp.quicksum(mu[k]*s[i,k] for k in K) >= (omega[i]+500)- M * (1 - u_pre[i]), name=f"dem_suppress_pre[{i}]")
    m.addConstr(
        gp.quicksum(mu[k]*s[i,k] for k in K) >= (omega[i]+500)-M * (1 - u_post[i]), name=f"dem_suppress_post[{i}]")
    m.addConstrs(
        (s[i, k] <= M * (u_pre[i] + u_post[i]) for k in K),
        name=f"no_water_if_no_control[{i}]"
    )
for i in Nf:
    for k in K:
        m.addConstr(v[i, k] >= t[i, k] + d[i, k] - M * (1 - x[i, k]), name=f"v_eq_td_lo[{i},{k}]")
        m.addConstr(v[i, k] <= t[i, k] + d[i, k] + M * (1 - x[i, k]), name=f"v_eq_td_up[{i},{k}]")

EPS = 1e-4  # zaman biriminize göre (örn. saat ise 1e-3 ~ 3.6 sn gibi)


# # (i) Atanmamışsa değişkenleri sıfıra yaklaştır (upper bound)
m.addConstrs((t[i,k] <= M * x[i,k] for i in Nf for k in K), name="t_zero_if_not_assigned")
m.addConstrs((s[i,k] <= M * x[i,k] for i in Nf for k in K), name="s_zero_if_not_assigned")
m.addConstrs((v[i,k] <= M * x[i,k] for i in Nf for k in K), name="v_zero_if_not_assigned")
#
Nplus = {i: set([i]) | set(neighbors.get(i, [])) for i in N}

# yardımcı min değişken
ts_min = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0.0, name="ts_min")

# seçim değişkenleri (sadece j in Nplus[i] için)
zmin = {}
for i in N:
    for j in Nplus[i]:
        zmin[i, j] = m.addVar(vtype=GRB.BINARY, name=f"zmin[{i},{j}]")

# Big-M (mümkünse zaman ufku kadar seçin)
T_max = P.get("time_horizon", 1e5)   # elinizde varsa daha sıkı bir değer verin
M_time = T_max

# (1) ts_min[i] <= ts[j]   tüm adaylar için
for i in Nf:
    for j in Nplus[i]:
        m.addConstr(ts_min[i] <= ts[j], name=f"tsmin_le_ts[{i},{j}]")

# (2) ts_min[i] >= ts[j] - M*(1 - zmin[i,j])  seçilen j için yapıştırma
for i in Nf:
    for j in Nplus[i]:
        m.addConstr(ts_min[i] >= ts[j] - M_time*(1 - zmin[i, j]),
                    name=f"tsmin_ge_ts_sel[{i},{j}]")

# (3) her i için tek seçim
for i in Nf:
    m.addConstr(gp.quicksum(zmin[i, j] for j in Nplus[i]) == 1,
                name=f"tsmin_one_choice[{i}]")


eps_arr = 1e-3        # "0 olmasın" için küçük pozitif değer (zaman biriminize göre ayarlayın)
buffer_time = 5.0     # +5 (dakika mı saat mi? sizin model biriminize göre)

for i in Nf:
    for k in K:
        m.addConstr(v[i,k] >= eps_arr - M_time*(1 - x[i,k]),
                    name=f"v_nonzero_bigM[{i},{k}]")
for i in Nf:
    for k in K:
        m.addConstr(t[i,k] >= ts_min[i] + buffer_time - M_time*(1 - x[i,k]),
                    name=f"t_ge_tsmin_plus5_bigM[{i},{k}]")


# # (ii) Varış yangın başlangıcından önce olmasın ve tm_i'yi geçmesin
m.addConstrs((v[i,k] >= ts[i] - M * (1 - x[i,k]) for i in Nf for k in K), name="eq_16")
m.addConstrs((v[i,k] <= tm[i] - EPS + M*(1 - x[i,k]) for i in Nf for k in K), name="eq_16+1")

# (iii) Bastırma bitişi kontrol zamanını aşmasın kısıt 15???
m.addConstrs((v[i,k] + s[i,k] <= tc[i] + M * (1 - x[i,k]) for i in Nf for k in K), name="eq_15")

m.addConstrs((x[i,k]<=y[i] for i in Nf for k in K),
    name="cons 8")

m.addConstrs((u_pre[i]+u_post[i] >= x[i,k] for i in Nf for k in K),
    name="cons 9")

m.addConstrs((u_pre[i]+u_post[i] <= gp.quicksum(x[i,k] for k in K) for i in Nf ),
    name="cons 10")

m.addConstrs((gp.quicksum(x[i,k] for i in Nf) <= 1 for k in K ),name="11")


m.addConstrs((tc[i]<=te[i] for i in Nf ),
    name="cons 12")

m.addConstrs((tc[i]>=ts[i] - M*(1-u_post[i]-u_pre[i]) for i in Nf ),
    name="cons 13")

m.addConstrs((tc[i]<=M*(u_post[i]+u_pre[i]) for i in Nf ),
    name="cons 14")

m.addConstrs((tc[i]<= tm[i] +epsilon+ M *(1-u_pre[i]) for i in Nf ),
    name="cons 15")

m.addConstrs((tc[i]>=tm[i]  - M *(1-u_post[i]) for i in Nf ),
    name="cons 16")

m.addConstrs((u_pre[i]+u_post[i] <=1 for i in Nf ),
    name="cons 17")
m.addConstrs((gp.quicksum(z[i, j] for j in neighbors[i]) == len(neighbors[i]) * (y[i] - u_pre[i]) for i in Nf),
                 name="cons 18")

m.addConstrs((len(neighbors[j]) * y[j] >= gp.quicksum(z[i, j] for i in neighbors[j]) for j in Nf),
                 name="cons 19")

m.addConstrs((q[i,j]<=z[i,j] for i in Nf for j in neighbors[i]),
                 name="cons 20")

m.addConstrs((gp.quicksum(q[i, j] for i in neighbors[j]) == y[j]-e[j] for j in Nf ),
                 name="cons 21")


m.addConstrs((ts[j]==gp.quicksum(tm[i]*q[i,j] for i in neighbors[j])  for j in Nf ),name="cons 22")

m.addConstrs((ts[j]<= tm[i]+ M*(1-z[i,j]) for j in Nf for i in neighbors[j] ),
                 name="cons 23")

m.addConstr(gp.quicksum(y[i] for i in Na) == len(Na),
    name="cons 24")

m.addConstr( gp.quicksum(ts[i] for i in Na) == 0,name="cons 25")

m.addConstrs(( tm[i] == ts[i] + alpha / degrade[i] for i in Nf ),
    name="cons 26")


m.addConstrs(( te[i] == tm[i] + alpha / ameliorate[i] for i in Nf ),
    name="cons 27")

# objective function
m.setObjective(gp.quicksum(p[i] for i in Nf), GRB.MAXIMIZE)



m.setParam("MIPGap", 0.03)
# --- Optimize et ---
m.optimize()

# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Eğer kodunuzun başka yerinde RESULT_DIR tanımlıysa bunu kaldırabilirsiniz.
RESULT_DIR = "result"

# -------------------------------
# 0 SATIRLARI YAZMAMA AYARLARI
# -------------------------------
ZERO_TOL = 1e-9  # |value| <= ZERO_TOL ise "0" kabul edilir

# Bu sayfalarda 0 satırlarını yazma (sizin talep ettiğiniz liste)
NO_ZERO_SHEETS = {
    "v_minmaxavg_ik",
    "omega_minmaxavg_ik",
    "delta",
    "v",
    "q",
    "z",
    "s",
    "t",
    "x",
}


def _drop_zero_rows(df: pd.DataFrame, value_col: str, zero_tol: float = ZERO_TOL) -> pd.DataFrame:
    """
    value_col sütununa göre 0 (veya çok küçük) değerli satırları çıkarır.
    Not: NaN satırları da elenir (mask ile).
    """
    if df is None or df.empty or value_col not in df.columns:
        return df
    mask = df[value_col].notna() & (df[value_col].abs() > zero_tol)
    return df.loc[mask].copy()


def _minmaxavg_long(df_val, df_x=None, df_t=None,
                    value_col="v_ik", x_col="x_ik", t_col="t_ik",
                    out_prefix="v", assign_threshold=0.5):
    """
    df_val: columns [i,k,value_col]
    df_x  : columns [i,k,x_col] (opsiyonel)
    df_t  : columns [i,k,t_col] (opsiyonel)

    Çıktı:
      - df_long: (i,k) bazında value, x, t, max/min işaretlemeleri ve average_{i}(i,k)
      - df_sum : i bazında özet (k_min, k_max, min, max, average_midpoint, average_mean,
                ayrıca t_kmin ve t_kmax)
    Not:
      average_{i}(i,k) midpoint ortalaması = (min+max)/2
      Filtre: yalnızca x_ik >= assign_threshold olan (atanmış) çiftler üzerinden min/max/avg hesaplanır.
    """

    df = df_val.copy()

    # x bilgisi
    if df_x is not None:
        df = df.merge(df_x, on=["i", "k"], how="left")
    else:
        df[x_col] = 1.0  # x yoksa hepsi "dahil" varsayılır

    # t bilgisi (uzun tabloya ek)
    if df_t is not None:
        df = df.merge(df_t, on=["i", "k"], how="left")

    # atanmış filtre
    df_ass = df[df[x_col].fillna(0.0) >= assign_threshold].copy()

    # atanmış yoksa boş döndür
    if df_ass.empty:
        df_long = df.copy()
        df_long[f"max_{out_prefix}(i,k)"] = np.nan
        df_long[f"min_{out_prefix}(i,k)"] = np.nan
        df_long[f"average_{out_prefix}(i,k)"] = np.nan

        df_sum_cols = ["i", "k_min", "min", "k_max", "max",
                       "average_midpoint", "average_mean", "t_kmin", "t_kmax"]
        df_sum = pd.DataFrame(columns=df_sum_cols)
        return df_long, df_sum

    # i bazında min/max/mean
    g = df_ass.groupby("i")[value_col]
    df_sum = g.agg(min="min", max="max", average_mean="mean").reset_index()
    df_sum["average_midpoint"] = 0.5 * (df_sum["min"] + df_sum["max"])

    # argmin/argmax (i bazında k seçimi)
    idx_min = df_ass.groupby("i")[value_col].idxmin()
    idx_max = df_ass.groupby("i")[value_col].idxmax()

    sel_min_cols = ["i", "k", value_col] + ([t_col] if t_col in df_ass.columns else [])
    sel_max_cols = ["i", "k", value_col] + ([t_col] if t_col in df_ass.columns else [])

    sel_min = df_ass.loc[idx_min, sel_min_cols].copy()
    sel_max = df_ass.loc[idx_max, sel_max_cols].copy()

    sel_min = sel_min.rename(columns={
        value_col: f"min_{out_prefix}(i,k)",
        "k": "k_min",
        t_col: "t_kmin" if t_col in sel_min.columns else t_col
    })

    sel_max = sel_max.rename(columns={
        value_col: f"max_{out_prefix}(i,k)",
        "k": "k_max",
        t_col: "t_kmax" if t_col in sel_max.columns else t_col
    })

    # long tabloya ekle: i bazlı average
    df_long = df.merge(df_sum[["i", "average_midpoint"]], on="i", how="left")

    # min/max yalnızca ilgili satırlarda dolu olsun diye (i,k) üzerinden merge
    sel_min_for_merge = sel_min.rename(columns={"k_min": "k"}).drop(columns=["t_kmin"], errors="ignore")
    sel_max_for_merge = sel_max.rename(columns={"k_max": "k"}).drop(columns=["t_kmax"], errors="ignore")

    df_long = df_long.merge(sel_min_for_merge[["i", "k", f"min_{out_prefix}(i,k)"]], on=["i", "k"], how="left")
    df_long = df_long.merge(sel_max_for_merge[["i", "k", f"max_{out_prefix}(i,k)"]], on=["i", "k"], how="left")

    df_long[f"average_{out_prefix}(i,k)"] = df_long["average_midpoint"]

    # özet tabloya k_min/k_max ve t_kmin/t_kmax bilgisi
    df_sum = (
        df_sum.merge(sel_min[["i", "k_min"] + (["t_kmin"] if "t_kmin" in sel_min.columns else [])], on="i", how="left")
              .merge(sel_max[["i", "k_max"] + (["t_kmax"] if "t_kmax" in sel_max.columns else [])], on="i", how="left")
    )

    # sütun düzeni
    ordered = ["i", "k_min", "min", "k_max", "max", "average_midpoint", "average_mean"]
    if "t_kmin" in df_sum.columns:
        ordered += ["t_kmin"]
    if "t_kmax" in df_sum.columns:
        ordered += ["t_kmax"]
    df_sum = df_sum[ordered]

    return df_long, df_sum


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


def _safe_write(writer, sheet_name, df, empty_columns=None):
    """
    df boş olsa bile sheet oluşturur.
    empty_columns: df boşsa bu kolonlarla boş bir tablo oluşturur.
    """
    if df is None:
        df = pd.DataFrame(columns=empty_columns or [])
    elif isinstance(df, pd.DataFrame) and df.empty:
        # Kolonlar hiç yoksa, beklenen kolonları ver
        if empty_columns is not None and len(df.columns) == 0:
            df = pd.DataFrame(columns=empty_columns)
        # df boş ama kolonları varsa, yine de yazılır

    df.to_excel(writer, index=False, sheet_name=sheet_name)


def _read_params(excel_paths):
    """
    inputs Excel'inin 'parameters' sayfasından n_nodes ve n_vehicles'ı çek.
    Yoksa None döndür.
    """
    try:
        p = pd.read_excel(excel_paths, sheet_name="parameters")
        p = p.set_index("parameter")["value"].to_dict()
        n_nodes = int(p.get("n_nodes")) if "n_nodes" in p else None
        n_vehicles = int(p.get("n_vehicles")) if "n_vehicles" in p else None
        return n_nodes, n_vehicles, p
    except Exception:
        return None, None, {}


def write_results_to_excel(model: gp.Model,
                           vars_dict: dict,
                           excel_paths,
                           out_dir=RESULT_DIR,
                           file_prefix="result",
                           d_matrix: dict = None,
                           vehicle_info: list = None  # Veri yükleme fonksiyonundan gelen liste
                           ):

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        raise RuntimeError(f"Çözülebilir bir çözüm bulunamadı (Status={model.Status}).")

    excel_inputs_path = Path(excel_paths)
    n_nodes, n_vehicles, params_all = _read_params(excel_inputs_path)

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx = Path(out_dir) / f"{file_prefix}__{excel_inputs_path.stem}__{ts_str}.xlsx"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:

        # 1) Meta Sayfası
        try:
            mipgap_val = float(model.MIPGap)  # 0.0–1.0 arası (oran)
        except Exception:
            mipgap_val = None

        mipgap_pct = (mipgap_val * 100.0) if mipgap_val is not None else None

        meta_rows = [
            ("InputFile", str(excel_inputs_path)),
            ("Status", model.Status),
            ("Objective", getattr(model, "ObjVal", float("nan"))),
            ("Runtime_sec", getattr(model, "Runtime", float("nan"))),
            ("MIPGap", mipgap_val if mipgap_val is not None else "N/A"),
            ("MIPGap_%", mipgap_pct if mipgap_pct is not None else "N/A"),
            ("ZERO_TOL", ZERO_TOL),
        ]
        _safe_write(writer, "meta", pd.DataFrame(meta_rows, columns=["metric", "value"]),
                    empty_columns=["metric", "value"])

        # 2) Deployment_Details + Vehicle_Master_List
        deploy_cols = ["Vehicle_ID", "Type", "Base", "Speed_km_h", "Capacity_L", "Target_Node", "Travel_Time_Hours"]
        veh_cols = ["Vehicle_ID", "Base", "Type", "Speed", "Capacity"]

        if "x" in vars_dict and vehicle_info is not None:
            veh_specs = {v['id']: v for v in vehicle_info}
            deployment_rows = []
            x_var = vars_dict["x"]

            for (i, k), var in x_var.items():
                if var.X is not None and var.X > 0.5:
                    specs = veh_specs.get(k, {})
                    deployment_rows.append({
                        "Vehicle_ID": k,
                        "Type": specs.get("type", "N/A"),
                        "Base": specs.get("base_id", "N/A"),
                        "Speed_km_h": specs.get("speed", 0),
                        "Capacity_L": specs.get("capacity", 0),
                        "Target_Node": i,
                        "Travel_Time_Hours": d_matrix.get((i, k), float("nan")) if d_matrix else "N/A"
                    })

            if deployment_rows:
                df_deploy = pd.DataFrame(deployment_rows).sort_values(by=["Base", "Vehicle_ID"])
            else:
                df_deploy = pd.DataFrame(columns=deploy_cols)

            _safe_write(writer, "Deployment_Details", df_deploy, empty_columns=deploy_cols)

            if vehicle_info:
                df_all_vehs = pd.DataFrame(vehicle_info)[['id', 'base_id', 'type', 'speed', 'capacity']].copy()
                df_all_vehs.columns = veh_cols
            else:
                df_all_vehs = pd.DataFrame(columns=veh_cols)

            _safe_write(writer, "Vehicle_Master_List", df_all_vehs, empty_columns=veh_cols)
        else:
            # Yine de sayfalar oluşsun
            _safe_write(writer, "Deployment_Details", pd.DataFrame(columns=deploy_cols), empty_columns=deploy_cols)
            _safe_write(writer, "Vehicle_Master_List", pd.DataFrame(columns=veh_cols), empty_columns=veh_cols)

        # 3) Tek indeksli değişkenler
        for name, (sheet, idx, colname) in {
            "p": ("p", "i", "p_i"),
            "y": ("y", "i", "y_i"),
            "ts": ("ts", "i", "t_i_s"),
            "tm": ("tm", "i", "t_i_m"),
            "te": ("te", "i", "t_i_e"),
            "tc": ("tc", "i", "t_i_c"),
            "u_pre": ("u_pre", "i", "u_i_pre"),
            "u_post": ("u_post", "i", "u_i_post"),
            "omega": ("omega", "i", "omega_i")
        }.items():
            if name in vars_dict and vars_dict[name] is not None:
                df = _single_index_to_df(vars_dict[name], idx, colname)
                _safe_write(writer, sheet, df, empty_columns=[idx, colname])
            else:
                # İsterseniz tek indeksli olmayanları da boş sheet basabilirsiniz.
                pass

        # 4) Çift indeksli değişkenler (0 satırlarını ayıklıyoruz ama sheet’i yine de basıyoruz)
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
                if sheet in NO_ZERO_SHEETS:
                    df = _drop_zero_rows(df, value_col=colname, zero_tol=ZERO_TOL)
                expected_cols = list(idx_names) + [colname]
                _safe_write(writer, sheet, df, empty_columns=expected_cols)
            else:
                # Sheet'in oluşmasını isterseniz:
                expected_cols = list(idx_names) + [colname]
                _safe_write(writer, sheet, pd.DataFrame(columns=expected_cols), empty_columns=expected_cols)

        # 5) TÜRETİLMİŞ RAPORLAR

        # --- v_minmaxavg ---
        expected_vik_cols = ["i", "k", "v_ik", "x_ik", "t_ik",
                             "average_midpoint", "min_v(i,k)", "max_v(i,k)", "average_v(i,k)"]
        expected_vi_cols = ["i", "k_min", "min", "k_max", "max", "average_midpoint", "average_mean", "t_kmin", "t_kmax"]

        if "v" in vars_dict and vars_dict["v"] is not None and "x" in vars_dict and vars_dict["x"] is not None:
            df_v = _double_index_to_df(vars_dict["v"], ("i", "k"), "v_ik")
            df_x = _double_index_to_df(vars_dict["x"], ("i", "k"), "x_ik")
            df_t = None
            if "t" in vars_dict and vars_dict["t"] is not None:
                df_t = _double_index_to_df(vars_dict["t"], ("i", "k"), "t_ik")

            df_v_long, df_v_sum = _minmaxavg_long(
                df_val=df_v, df_x=df_x, df_t=df_t,
                value_col="v_ik", x_col="x_ik", t_col="t_ik",
                out_prefix="v", assign_threshold=0.5
            )

            # v_minmaxavg_ik sayfasında ana değer v_ik=0 satırlarını yazma
            df_v_long = _drop_zero_rows(df_v_long, value_col="v_ik", zero_tol=ZERO_TOL)

            _safe_write(writer, "v_minmaxavg_ik", df_v_long, empty_columns=expected_vik_cols)
            _safe_write(writer, "v_minmaxavg_i", df_v_sum, empty_columns=expected_vi_cols)
        else:
            _safe_write(writer, "v_minmaxavg_ik", pd.DataFrame(columns=expected_vik_cols), empty_columns=expected_vik_cols)
            _safe_write(writer, "v_minmaxavg_i", pd.DataFrame(columns=expected_vi_cols), empty_columns=expected_vi_cols)

        # --- omega_minmaxavg (delta üzerinden) ---
        expected_omik_cols = ["i", "k", "delta_ik", "x_ik",
                              "average_midpoint", "min_omega(i,k)", "max_omega(i,k)", "average_omega(i,k)"]
        expected_omi_cols = ["i", "k_min", "min_omega(i)", "k_max", "max_omega(i)",
                             "average_omega(i)", "omega_mean(i)", "omega_i_model"]

        if "delta" in vars_dict and vars_dict["delta"] is not None and "x" in vars_dict and vars_dict["x"] is not None:
            df_delta = _double_index_to_df(vars_dict["delta"], ("i", "k"), "delta_ik")
            df_x = _double_index_to_df(vars_dict["x"], ("i", "k"), "x_ik")

            df_d_long, df_d_sum = _minmaxavg_long(
                df_val=df_delta, df_x=df_x,
                value_col="delta_ik", x_col="x_ik",
                out_prefix="omega", assign_threshold=0.5
            )

            df_omega_stats = df_d_sum.rename(columns={
                "min": "min_omega(i)",
                "max": "max_omega(i)",
                "average_midpoint": "average_omega(i)",
                "average_mean": "omega_mean(i)"
            })

            # Modelde omega_i varsa ekle
            if "omega" in vars_dict and vars_dict["omega"] is not None:
                df_omega = _single_index_to_df(vars_dict["omega"], "i", "omega_i_model")
                df_omega_stats = df_omega_stats.merge(df_omega, on="i", how="left")
            else:
                if "omega_i_model" not in df_omega_stats.columns:
                    df_omega_stats["omega_i_model"] = np.nan

            _safe_write(writer, "omega_minmaxavg_i", df_omega_stats, empty_columns=expected_omi_cols)

            # omega_minmaxavg_ik sayfasında: ana değer delta_ik=0 satırlarını yazma
            df_d_long = _drop_zero_rows(df_d_long, value_col="delta_ik", zero_tol=ZERO_TOL)
            _safe_write(writer, "omega_minmaxavg_ik", df_d_long, empty_columns=expected_omik_cols)
        else:
            _safe_write(writer, "omega_minmaxavg_i", pd.DataFrame(columns=expected_omi_cols), empty_columns=expected_omi_cols)
            _safe_write(writer, "omega_minmaxavg_ik", pd.DataFrame(columns=expected_omik_cols), empty_columns=expected_omik_cols)

    print(f"Sonuçlar Excel'e yazıldı: {xlsx}")
    return str(xlsx)


# ============================================================
# ÖRNEK KULLANIM (Sizde mevcut değişkenler olmalı: p,y,ts,...)
# ============================================================
# vars_dict örneği (sizde hangi değişkenler varsa onları ekleyin)
vars_dict = {
    "p": p, "y": y, "ts": ts, "tm": tm, "te": te, "tc": tc,
    "u_pre": u_pre, "u_post": u_post,
    "x": x, "t": t, "s": s, "z": z, "q": q,
    "omega": omega,
    "v": v,
    "delta": delta,
}

out_file = write_results_to_excel(
    model=m,
    vars_dict=vars_dict,
    excel_paths=EXCEL_PATH,
    d_matrix=data["d_matrix"],
    vehicle_info=data["vehicle_info"]
)




