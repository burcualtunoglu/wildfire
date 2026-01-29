# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB
from load_inputs import load_wildfire_data_final
import pandas as pd
excel_path = r"C:\Users\Mert\deprem\wildfire\inputs_new_6x6.xlsx"

try:
    data = load_wildfire_data_final(excel_path)
except Exception as e:
    print(f"Veri yüklenirken hata oluştu: {e}")
    raise
data = load_wildfire_data_final(excel_path)

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


# def export_model_parameters_to_excel(data, Na, e, filename="model_kontrol.xlsx"):
#     """
#     Model parametrelerini doğrulama amacıyla Excel dosyasına aktarır.
#     """
#     with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
#         # --- 1. DÜĞÜM BAZLI VERİLER (Node-based Data) ---
#         # N, state, v0, mu, degrade, ameliorate, su, beta ve e değerlerini birleştiriyoruz
#         node_data = {
#             "Node_ID": data["N"],
#             "State (Aktif Mi?)": [data["state"][i] for i in data["N"]],
#             "e_i (Indicative)": [e[i] for i in data["N"]],
#             "v0 (Yangın Yükü)": [data["v0"][i] for i in data["N"]],
#             "su (Kaynak İhtiyacı)": [data["su"][i] for i in data["N"]],
#             "degrade (Yayılma)": [data["degrade"][i] for i in data["N"]],
#             "ameliorate (Sönümleme)": [data["ameliorate"][i] for i in data["N"]],
#             "beta (Önem)": [data["beta"][i] for i in data["N"]]
#         }
#         df_nodes = pd.DataFrame(node_data)
#         df_nodes.to_excel(writer, sheet_name='Dugum_Verileri', index=False)
#
#         # --- 2. MESAFE MATRİSİ (Distance Matrix) ---
#         # gp.tupledict yapısını (i, j) formatından tablo formatına çeviriyoruz
#         d_matrix = data["d_matrix"]
#         # Sözlüğü DataFrame'e çevirip pivot yapıyoruz (Satır: i, Sütun: j)
#         d_list = [{"i": k[0], "j": k[1], "Distance": v} for k, v in d_matrix.items()]
#         df_dist = pd.DataFrame(d_list).pivot(index='i', columns='j', values='Distance')
#         df_dist.to_excel(writer, sheet_name='Mesafe_Matrisi')
#
#         # --- 3. GENEL VE LİSTE BAZLI PARAMETRELER ---
#         # Araçlar, Aktif Düğümler (Na) ve Global parametreler
#         general_params = {
#             "Parametre Adı": ["Araç Kümesi (K)", "Aktif Düğümler (Na)", "Bölge Kenar Uzunluğu (alpha)"],
#             "Değer": [str(list(data["K"])), str(Na), data["P"].get("region_side_length", 1.0)]
#         }
#         df_general = pd.DataFrame(general_params)
#         df_general.to_excel(writer, sheet_name='Genel_Parametreler', index=False)
#
#         # --- 4. KOMŞULUK İLİŞKİLERİ ---
#         neighbors_data = [{"Düğüm": i, "Komşular": str(data["neighbors"][i])} for i in data["N"]]
#         df_neighbors = pd.DataFrame(neighbors_data)
#         df_neighbors.to_excel(writer, sheet_name='Komsu_Listesi', index=False)
#
#     print(f"Veriler başarıyla '{filename}' dosyasına kaydedildi.")
# export_model_parameters_to_excel(data, Na, e)
M = 1e6
OMEGA_MAX =36322000000
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
        m.addConstr(omega_max[i] <= delta[i,k] + OMEGA_MAX*(1 - z_max[i,k]),
                    name=f"omax_le_delta_sel[{i},{k}]")
        m.addConstr(z_max[i,k] <= x[i,k], name=f"zmax_le_x[{i},{k}]")

        # ---------- MIN (atanmayanı dışla) ----------
        m.addConstr(omega_min[i] <= delta[i,k] ,
                    name=f"omin_le_delta_if_assigned[{i},{k}]")
        m.addConstr(omega_min[i] >= delta[i,k] - OMEGA_MAX*(1 - z_min[i,k]),
                    name=f"omin_ge_delta_sel[{i},{k}]")
        m.addConstr(z_min[i,k] <= x[i,k], name=f"zmin_le_x[{i},{k}]")

    # Seçim sayıları (faz aktifse 1, değilse 0)
    m.addConstr(gp.quicksum(z_max[i,k] for k in K) == u, name=f"zmax_sum[{i}]")
    m.addConstr(gp.quicksum(z_min[i,k] for k in K) == u, name=f"zmin_sum[{i}]")

    # Ortalama (min-max midpoint)
    m.addConstr(2*omega[i] == omega_max[i] + omega_min[i], name=f"oavg_def[{i}]")

    # # Faz kapalıyken sıfıra zorlamak (opsiyonel ama temiz)
    # m.addConstr(omega_max[i] <= OMEGA_MAX*u, name=f"omax_off[{i}]")
    # m.addConstr(omega_min[i] <= OMEGA_MAX*u, name=f"omin_off[{i}]")
    # m.addConstr(omega[i] <= OMEGA_MAX*u, name=f"oavg_off[{i}]")



# A) su talebi – u_pre/u_post’a bağla
for i in Nf:
    m.addGenConstrIndicator(u_pre[i], 1,
        gp.quicksum(mu[k]*s[i,k] for k in K) >= (omega[i]+500), name=f"dem_suppress_pre[{i}]")
    m.addGenConstrIndicator(u_post[i], 1,
        gp.quicksum(mu[k]*s[i,k] for k in K) >= (omega[i]+500), name=f"dem_suppress_post[{i}]")
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
import pandas as pd

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

    sel_min = df_ass.loc[idx_min, ["i", "k", value_col] + ([t_col] if t_col in df_ass.columns else [])].copy()
    sel_max = df_ass.loc[idx_max, ["i", "k", value_col] + ([t_col] if t_col in df_ass.columns else [])].copy()

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

    # long tabloya ekle: i bazlı average ve yalnız seçili satırlara min/max
    df_long = df.merge(df_sum[["i", "average_midpoint"]], on="i", how="left")

    # min/max yalnızca ilgili satırlarda dolu olsun diye (i,k) üzerinden merge
    # sel_min/sel_max'ta k_min/k_max var; df_long ile (i,k) eşleştirmek için kolon adı uyarlayalım:
    sel_min_for_merge = sel_min.rename(columns={"k_min": "k"}).drop(columns=["t_kmin"], errors="ignore")
    sel_max_for_merge = sel_max.rename(columns={"k_max": "k"}).drop(columns=["t_kmax"], errors="ignore")

    df_long = df_long.merge(sel_min_for_merge[["i", "k", f"min_{out_prefix}(i,k)"]], on=["i", "k"], how="left")
    df_long = df_long.merge(sel_max_for_merge[["i", "k", f"max_{out_prefix}(i,k)"]], on=["i", "k"], how="left")

    df_long[f"average_{out_prefix}(i,k)"] = df_long["average_midpoint"]

    # özet tabloya k_min/k_max ve t_kmin/t_kmax bilgisi
    df_sum = df_sum.merge(sel_min[["i", "k_min"] + (["t_kmin"] if "t_kmin" in sel_min.columns else [])], on="i", how="left") \
                 .merge(sel_max[["i", "k_max"] + (["t_kmax"] if "t_kmax" in sel_max.columns else [])], on="i", how="left")

    # sütun düzeni
    ordered = ["i", "k_min", "min", "k_max", "max", "average_midpoint", "average_mean"]
    if "t_kmin" in df_sum.columns: ordered += ["t_kmin"]
    if "t_kmax" in df_sum.columns: ordered += ["t_kmax"]
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
                           excel_inputs_path,
                           out_dir=r"C:\Users\Mert\deprem\wildfire\result",
                           file_prefix="results",
                           d_matrix: dict = None,
                           vehicle_info: list = None # Veri yükleme fonksiyonundan gelen liste
                           ):
    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        raise RuntimeError(f"Çözülebilir bir çözüm bulunamadı (Status={model.Status}).")

    excel_inputs_path = Path(excel_inputs_path)
    n_nodes, n_vehicles, params_all = _read_params(excel_inputs_path)

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx = Path(out_dir) / f"{file_prefix}__{excel_inputs_path.stem}__{ts_str}.xlsx"

    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
        # 1. Meta Sayfası
        try:
            mipgap_val = float(model.MIPGap)  # 0.0–1.0 arası (oran)
        except Exception:
            mipgap_val = None

        # İsterseniz yüzde olarak da yazdırabilirsiniz:
        mipgap_pct = (mipgap_val * 100.0) if mipgap_val is not None else None

        meta_rows = [
            ("InputFile", str(excel_inputs_path)),
            ("Status", model.Status),
            ("Objective", getattr(model, "ObjVal", float("nan"))),
            ("Runtime_sec", getattr(model, "Runtime", float("nan"))),

            # GAP (oran ve yüzde)
            ("MIPGap", mipgap_val if mipgap_val is not None else "N/A"),
            ("MIPGap_%", mipgap_pct if mipgap_pct is not None else "N/A"),
        ]

        _safe_write(writer, "meta", pd.DataFrame(meta_rows, columns=["metric", "value"]))

        # --- GÜNCELLENEN KISIM: Araç Teknik Özellikleri ve Atamalar ---
        if "x" in vars_dict and vehicle_info is not None:
            # Hızlı erişim için araç özelliklerini ID'ye göre sözlüğe dönüştür
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
                _safe_write(writer, "Deployment_Details", df_deploy)

            # Tüm araçların listesini ayrı bir sayfada raporla (Opsiyonel)
            df_all_vehs = pd.DataFrame(vehicle_info)[['id', 'base_id', 'type', 'speed', 'capacity']]
            df_all_vehs.columns = ['Vehicle_ID', 'Base', 'Type', 'Speed', 'Capacity']
            _safe_write(writer, "Vehicle_Master_List", df_all_vehs)
        # --- YENİ EKLENEN KISIM: Base'den Varış Süreleri Raporu ---
        # Eğer x değişkeni, d matrisi ve araç atamaları verildiyse bu raporu oluştur.

        # -----------------------------------------------------------

        # 3. Tek İndeksli Değişkenler
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
                _safe_write(writer, sheet, df)

        # 4. Çift İndeksli Değişkenler
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
        # -----------------------------------------------------------
        # 5) TÜRETİLMİŞ RAPORLAR: v için max/min/avg ve omega için max/min/avg
        # -----------------------------------------------------------

        # --- v için (i,k) bazında max_v(i,k), min_v(i,k), average_v(i,k) ---
        if "v" in vars_dict and vars_dict["v"] is not None:
            df_v = _double_index_to_df(vars_dict["v"], ("i", "k"), "v_ik")
            df_x = _double_index_to_df(vars_dict["x"], ("i", "k"), "x_ik")  # varsa
            df_t = _double_index_to_df(vars_dict["t"], ("i", "k"), "t_ik")  # t_ik eklenecekse

            df_v_long, df_v_sum = _minmaxavg_long(
                df_val=df_v, df_x=df_x, df_t=df_t,
                value_col="v_ik", x_col="x_ik", t_col="t_ik",
                out_prefix="v", assign_threshold=0.5
            )

            _safe_write(writer, "v_minmaxavg_ik", df_v_long)
            _safe_write(writer, "v_minmaxavg_i",  df_v_sum)

        # --- omega için: delta üzerinden i bazında max_omega(i), min_omega(i), average_omega(i) ---
        # Not: Burada omega'nın "max/min/avg" kavramını delta_{ik} (atanmış ekipler) üzerinden türetiyoruz.
        if "delta" in vars_dict and vars_dict["delta"] is not None:
            df_delta = _double_index_to_df(vars_dict["delta"], ("i", "k"), "delta_ik")

            df_x = None
            if "x" in vars_dict and vars_dict["x"] is not None:
                df_x = _double_index_to_df(vars_dict["x"], ("i", "k"), "x_ik")

            df_d_long, df_d_sum = _minmaxavg_long(
                df_val=df_delta, df_x=df_x,
                value_col="delta_ik", x_col="x_ik",
                out_prefix="omega", assign_threshold=0.5
            )

            # df_d_sum şu sütunları içerir: i, k_min, min, k_max, max, average_midpoint, average_mean
            # İstenen adlara dönüştürelim:
            df_omega_stats = df_d_sum.rename(columns={
                "min": "min_omega(i)",
                "max": "max_omega(i)",
                "average_midpoint": "average_omega(i)",
                "average_mean": "omega_mean(i)"  # opsiyonel: gerçek aritmetik ortalama
            })

            # Modeldeki omega_i varsa yanına ekleyelim (tutarlılık kontrolü için faydalı)
            if "omega" in vars_dict and vars_dict["omega"] is not None:
                df_omega = _single_index_to_df(vars_dict["omega"], "i", "omega_i_model")
                df_omega_stats = df_omega_stats.merge(df_omega, on="i", how="left")

            _safe_write(writer, "omega_minmaxavg_i", df_omega_stats)

            # İsterseniz (i,k) bazında da hangi satırın min/max seçildiğini görmek için:
            _safe_write(writer, "omega_minmaxavg_ik", df_d_long)

    print(f"Sonuçlar Excel'e yazıldı: {xlsx}")
    return str(xlsx)


# vars_dict örneği (sizde hangi değişkenler varsa onları ekleyin)
# vars_dict oluşturma (Mevcut kodunuz)
vars_dict = {
    "p": p, "y": y, "ts": ts, "tm": tm, "te": te, "tc": tc,
    "u_pre": u_pre, "u_post": u_post,
    "x": x, "t": t, "s": s, "z": z, "q": q,
    "omega": omega,
    "v": v,
    "delta": delta,
}

# Fonksiyonu çağırma (GÜNCELLENMİŞ HALİ)
out_file = write_results_to_excel(
    model=m,
    vars_dict=vars_dict,
    excel_inputs_path=r"C:\Users\Mert\deprem\wildfire\inputs_new_6x6.xlsx",
    d_matrix=data["d_matrix"],       # Seyahat süreleri matrisi
    vehicle_info=data["vehicle_info"] # Araç teknik özellikleri listesi
)

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib as mpl
from pathlib import Path
from datetime import datetime
from matplotlib.colors import to_rgb, LinearSegmentedColormap
from matplotlib.lines import Line2D

# =========================================================
# 0) KONFİGÜRASYON & AKADEMİK AYARLAR
# =========================================================
BASE_DIR = Path(r"C:\Users\Mert\deprem\wildfire")
EXCEL_PATH = BASE_DIR / "inputs_new_6x6.xlsx"
RESULTS_DIR = BASE_DIR / "result"
OUT_DIR = BASE_DIR / "maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# İç orman yolu çiftleri
ROAD_PAIRS = [(2, 3), (3, 9), (4, 10), (14, 15), (20, 21), (20, 26), (18, 24), (17, 23)]

# Akademik Font ve Çözünürlük Ayarları
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "figure.dpi": 300,
    "axes.labelsize": 16,
    "axes.titlesize": 24,
    "legend.fontsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "savefig.dpi": 300,
    "axes.linewidth": 1.5
})

# Renk Paleti (Akademik - Muted Tones)
COLOR_BLACK = "#34495e"
COLOR_ORANGE = "#e67e22"
COLOR_YELLOW = "#f39c12"
COLOR_GREEN = "#27ae60"
COLOR_WATER = "#2980b9"

BASE_COLORS = {
    "A": "#3498db", "B": "#9b59b6", "C": "#e84393", "D": "#a84300", "E": "#1abc9c"
}


# =========================================================
# 1) YARDIMCI FONKSİYONLAR
# =========================================================
def as_int(x, default=0):
    try:
        return int(float(x))
    except:
        return default


def fnum(v):
    return f"{v:.2f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "-"


def load_wildfire_data_final(excel_path: Path):
    xls = pd.ExcelFile(excel_path)
    df_nodes = pd.read_excel(xls, sheet_name="inputs_df")
    df_bases = pd.read_excel(xls, sheet_name="bases")

    vehicle_records = []
    current_k = 1
    vehicle_types = ["Helicopter", "Fire Engine", "FRV"]
    for _, row in df_bases.iterrows():
        b_id = str(row.get("Base", row.get("base_id", ""))).strip()
        bx, by = float(row["x_coordinate"]), float(row["y_coordinate"])
        for v_type in vehicle_types:
            if v_type in df_bases.columns and pd.notna(row[v_type]):
                count = int(float(row[v_type]))
                for _ in range(count):
                    vehicle_records.append({
                        "id": current_k, "base_id": b_id, "type": v_type,
                        "bx": bx, "by": by
                    })
                    current_k += 1
    return {"vehicle_info": vehicle_records, "nodes_df": df_nodes, "bases_df": df_bases}


def _latest_results_file(results_dir: Path):
    files = sorted(results_dir.glob("results_*.xlsx"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def read_sheet(path: Path, sheet: str, key="i"):
    try:
        df = pd.read_excel(path, sheet_name=sheet)
        return dict(zip(df[key], df[df.columns[-1]]))
    except:
        return {}


def read_meta_info(path: Path):
    try:
        df_meta = pd.read_excel(path, sheet_name="meta")
        meta = dict(zip(df_meta.iloc[:, 0], df_meta.iloc[:, 1]))

        obj = meta.get("Objective", "N/A")
        runtime = meta.get("Runtime_sec", "N/A")
        gap = meta.get("MIPGap_%", "N/A")

        if isinstance(obj, str): obj = float(obj.replace(",", "."))
        if isinstance(runtime, str): runtime = float(runtime.replace(",", "."))
        if isinstance(gap, str): gap = float(gap.replace(",", "."))

        return obj, runtime, gap
    except:
        return "N/A", "N/A", "N/A"


def read_x_sum(path: Path):
    try:
        return pd.read_excel(path, sheet_name="x").groupby("i").iloc[:, -1].sum().to_dict()
    except:
        return {}


def read_v_max_from_stats(path: Path):
    """
    GÜNCELLEME: v_minmaxavg_i sayfasından 'max' sütununu okur.
    Bu sütun T_arr (son aracın varış zamanı) değerini verir.
    """
    try:
        df = pd.read_excel(path, sheet_name="v_minmaxavg_i")
        # Sütun isimlerindeki boşlukları temizle (örn: " max" -> "max")
        df.columns = df.columns.str.strip()

        # 'max' sütununu bul
        if "max" in df.columns and "i" in df.columns:
            return dict(zip(df["i"], df["max"]))
        else:
            print("Uyarı: v_minmaxavg_i sayfasında 'i' veya 'max' sütunu bulunamadı.")
            return {}
    except Exception as e:
        # Sayfa yoksa veya hata varsa boş dön
        # print(f"Bilgi: v_minmaxavg_i okunamadı ({e}), eski yöntem deneniyor...")
        return {}


def read_q(path: Path):
    try:
        df = pd.read_excel(path, sheet_name="q")
        val = df.columns[-1]
        return df[df[val] >= 0.5].groupby("i")["j"].apply(list).to_dict()
    except:
        return {}


# =========================================================
# 2) GRID VE HARİTA ARAÇLARI
# =========================================================
def grid_setup_from_nodes(df_nodes):
    x_coords = sorted(df_nodes["x_coordinate"].unique())
    y_coords = sorted(df_nodes["y_coordinate"].unique())
    dx = (x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0
    dy = (y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0
    grid_extent = [min(x_coords) - dx / 2, max(x_coords) + dx / 2, min(y_coords) - dy / 2, max(y_coords) + dy / 2]
    return x_coords, y_coords, dx, dy, grid_extent, {x: i for i, x in enumerate(x_coords)}, {y: i for i, y in
                                                                                             enumerate(y_coords)}


def apply_cell_boundary_grid(ax, x_coords, y_coords, dx, dy):
    x_edges = [min(x_coords) - dx / 2 + k * dx for k in range(len(x_coords) + 1)]
    y_edges = [min(y_coords) - dy / 2 + k * dy for k in range(len(y_coords) + 1)]
    ax.set_xticks(x_edges, minor=True)
    ax.set_yticks(y_edges, minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=2.0, alpha=0.6, zorder=5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def get_map_extent(df_nodes, df_bases, dx, dy, pad_cells=0.6):
    all_x = np.concatenate([df_nodes["x_coordinate"].values, df_bases["x_coordinate"].values])
    all_y = np.concatenate([df_nodes["y_coordinate"].values, df_bases["y_coordinate"].values])
    pad = pad_cells * max(dx, dy)
    return [min(all_x) - pad, max(all_x) + pad, min(all_y) - pad, max(all_y) + pad]


def add_north_arrow(ax, xy=(0.96, 0.95), arrow_length=0.08):
    ax.annotate('N', xy=xy, xytext=(xy[0], xy[1] - arrow_length),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='top', fontsize=24, fontweight='bold', xycoords='axes fraction', zorder=100)


def add_scale_bar(ax, dx, extent):
    x_start = extent[0] + dx * 0.5
    y_start = extent[2] + dx * 0.3
    ax.plot([x_start, x_start + dx], [y_start, y_start], color='black', linewidth=4, zorder=100)
    tick_h = dx * 0.1
    ax.plot([x_start, x_start], [y_start - tick_h / 2, y_start + tick_h / 2], color='black', linewidth=2, zorder=100)
    ax.plot([x_start + dx, x_start + dx], [y_start - tick_h / 2, y_start + tick_h / 2], color='black', linewidth=2,
            zorder=100)
    ax.text(x_start + dx / 2, y_start - tick_h, "1 km", ha='center', va='top', fontsize=20, fontweight='bold',
            zorder=100)


# =========================================================
# 3) ÇİZİM FONKSİYONLARI
# =========================================================
def draw_bases(ax, df_bases):
    for _, row in df_bases.iterrows():
        b_id = str(row.get("Base", row.get("base_id", ""))).strip()
        bx, by = float(row["x_coordinate"]), float(row["y_coordinate"])
        b_color = BASE_COLORS.get(b_id, "#333333")
        ax.scatter(bx, by, s=1500, marker="s", color=b_color, edgecolors="black", linewidth=2.0, zorder=40)
        ax.text(bx, by, b_id, color="white", fontweight="bold", ha="center", va="center", fontsize=20, zorder=41)


def draw_inner_forest_roads(ax, df_nodes, dx, dy, road_pairs):
    node_xy = {int(r.node_id): (float(r.x_coordinate), float(r.y_coordinate))
               for r in df_nodes.itertuples()}
    for (id1, id2) in road_pairs:
        if id1 in node_xy and id2 in node_xy:
            x1, y1 = node_xy[id1]
            x2, y2 = node_xy[id2]
            line_style = '-'
            line_width = 6
            line_color = 'black'

            if abs(abs(x1 - x2) - dx) < 1e-6 and abs(y1 - y2) < 1e-6:
                ex = (x1 + x2) / 2.0
                ax.plot([ex, ex], [y1 - dy / 2, y1 + dy / 2], color=line_color, linewidth=line_width, zorder=7,
                        linestyle=line_style)
            elif abs(abs(y1 - y2) - dy) < 1e-6 and abs(x1 - x2) < 1e-6:
                ey = (y1 + y2) / 2.0
                ax.plot([x1 - dx / 2, x1 + dx / 2], [ey, ey], color=line_color, linewidth=line_width, zorder=7,
                        linestyle=line_style)


def mark_initial_sources(ax, df_nodes, dx, dy):
    for _, row in df_nodes.iterrows():
        if as_int(row.get("state", 0)) == 1:
            cx, cy = float(row["x_coordinate"]), float(row["y_coordinate"])
            rect = mpatches.Rectangle((cx - dx / 2 + 0.05, cy - dy / 2 + 0.05), dx - 0.1, dy - 0.1,
                                      fill=False, edgecolor="#c0392b", linewidth=5, zorder=15, linestyle='-')
            ax.add_patch(rect)


# =========================================================
# 4) PLOT DETAIL MAP (YERLEŞİM DÜZENLENDİ)
# =========================================================
def plot_detail_map(data):
    df_nodes, df_bases = data["nodes_df"], data["bases_df"]
    veh_map = {v["id"]: {"type": v["type"], "base": v["base_id"]} for v in data["vehicle_info"]}

    res_path = _latest_results_file(RESULTS_DIR)
    if res_path is None: return

    obj_val, runtime, gap_val = read_meta_info(res_path)
    p_map, y_status = read_sheet(res_path, "p"), read_sheet(res_path, "y")

    tc_map = read_sheet(res_path, "tc")
    ts_map, tm_map, te_map = read_sheet(res_path, "ts"), read_sheet(res_path, "tm"), read_sheet(res_path, "te")

    sumx = read_x_sum(res_path)

    # GÜNCELLEME: İstatistik sayfasından T_arr oku
    max_v = read_v_max_from_stats(res_path)

    q_ones = read_q(res_path)
    raw_omega = read_sheet(res_path, "omega") or read_sheet(res_path, "w")
    start_vals = dict(zip(df_nodes["node_id"], df_nodes["value_at_start"]))

    node_vehicles = {}
    try:
        df_x = pd.read_excel(res_path, sheet_name="x")
        for _, r in df_x[df_x.iloc[:, -1] >= 0.5].iterrows():
            i, k = int(r["i"]), int(r["k"])
            node_vehicles.setdefault(i, []).append(veh_map.get(k, {}))
    except:
        pass

    omega_map = {nid: (val + 500 if (nid in node_vehicles and node_vehicles[nid]) else val)
                 for nid, val in raw_omega.items()}

    x_coords, y_coords, dx, dy, grid_extent, x_to_idx, y_to_idx = grid_setup_from_nodes(df_nodes)

    # Harita boyutu
    fig_w, fig_h = max(30, len(x_coords) * 7), max(30, len(y_coords) * 7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    rgb_grid = np.zeros((len(y_coords), len(x_coords), 3), dtype=float)
    for _, row in df_nodes.iterrows():
        nid, xi, yi = int(row["node_id"]), x_to_idx[row["x_coordinate"]], y_to_idx[row["y_coordinate"]]
        st = as_int(row.get("state", 0))

        if st == 2:
            color = COLOR_WATER
        else:
            p, s, yv = float(p_map.get(nid, 0)), float(start_vals.get(nid, 0)), float(y_status.get(nid, 0))
            if p <= 0.001:
                color = COLOR_BLACK
            elif p < (s - 0.001):
                color = COLOR_ORANGE
            elif yv >= 0.5:
                color = COLOR_YELLOW
            else:
                color = COLOR_GREEN
        rgb_grid[yi, xi] = to_rgb(color)

    ax.imshow(rgb_grid, origin="lower", aspect="equal", extent=grid_extent, interpolation="nearest", zorder=1)

    extent = get_map_extent(df_nodes, df_bases, dx, dy)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    apply_cell_boundary_grid(ax, x_coords, y_coords, dx, dy)
    draw_inner_forest_roads(ax, df_nodes, dx, dy, ROAD_PAIRS)
    mark_initial_sources(ax, df_nodes, dx, dy)
    draw_bases(ax, df_bases)
    add_north_arrow(ax)
    add_scale_bar(ax, dx, extent)

    NODE_ID_FS = 42
    INFO_FS = 20

    for _, row in df_nodes.iterrows():
        nid, cx, cy = int(row["node_id"]), float(row["x_coordinate"]), float(row["y_coordinate"])
        st = as_int(row.get("state", 0))

        if st == 2:
            ax.text(cx, cy, str(nid), fontsize=NODE_ID_FS, fontweight="bold", color="white",
                    zorder=10, path_effects=[pe.withStroke(linewidth=6, foreground="black")])
            continue

        p_val = float(p_map.get(nid, 0))
        s_val = float(start_vals.get(nid, 0))
        y_val = float(y_status.get(nid, 0))

        current_color = COLOR_BLACK if p_val <= 0.001 else (
            COLOR_ORANGE if p_val < s_val - 0.001 else (COLOR_YELLOW if y_val >= 0.5 else COLOR_GREEN))
        text_color = "black" if current_color == COLOR_YELLOW else "white"

        # GÜNCELLEME: Node ID biraz daha aşağıya (0.35 -> 0.30)
        ax.text(cx, cy + dy * 0.30, str(nid), fontsize=NODE_ID_FS, fontweight="bold",
                ha="center", color=text_color, zorder=10)

        lines = [f"$V_0$: {fnum(s_val)}"]
        if current_color != COLOR_GREEN:
            lines[0] += f" → $V_e$: {fnum(p_val)}"
            if y_val >= 0.5:
                # Time Windows
                win_str = f"Win: [{fnum(ts_map.get(nid))}, {fnum(tm_map.get(nid))}, {fnum(te_map.get(nid))}]"
                lines.append(win_str)

                # T_arr (Max) ve T_c
                t_arr_val = max_v.get(nid)
                t_c_val = tc_map.get(nid)
                times_str = f"T_arr: {fnum(t_arr_val)} | T_c: {fnum(t_c_val)}"
                lines.append(times_str)

                lines.append(f"Water: {fnum(omega_map.get(nid))}")

                ign_list = q_ones.get(nid, [])
                if ign_list:
                    ign_str = ",".join(map(str, ign_list))
                    lines.append(f"Ign: {ign_str}")

        # GÜNCELLEME: Bilgi kutusu tam ortaya (0.02 -> 0.0)
        ax.text(cx, cy, "\n".join(lines), fontsize=INFO_FS, ha="center", va="center",
                color=text_color, fontweight="normal", zorder=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.15, edgecolor="none"))

        assignments = node_vehicles.get(nid, [])
        if assignments:
            dot_rad = dx * 0.08
            spacing = dot_rad * 2.5
            for idx, veh in enumerate(assignments):
                r_idx, c_idx = divmod(idx, 4)
                px = (cx - spacing * 1.5) + c_idx * spacing
                # GÜNCELLEME: Araçlar en alta (-0.42 -> -0.35 civarı)
                py = (cy - dy * 0.35) + r_idx * spacing
                v_col = BASE_COLORS.get(veh["base"], "gray")

                patch_args = dict(color=v_col, ec="white", lw=1.5, zorder=20)
                if veh["type"] == "FRV":
                    ax.add_patch(mpatches.RegularPolygon((px, py), 3, radius=dot_rad * 1.2, **patch_args))
                elif veh["type"] == "Helicopter":
                    ax.add_patch(mpatches.Circle((px, py), radius=dot_rad, **patch_args))
                else:
                    ax.add_patch(
                        mpatches.Rectangle((px - dot_rad, py - dot_rad), dot_rad * 2, dot_rad * 2, **patch_args))

    obj_str = f"{obj_val:.3f}" if isinstance(obj_val, (int, float)) else str(obj_val)
    gap_str = f"{gap_val:.2f}%" if isinstance(gap_val, (int, float)) else str(gap_val)
    run_str = f"{runtime:.3f}" if isinstance(runtime, (int, float)) else str(runtime)

    ax.set_title(
        f"Detailed Wildfire Optimization Analysis\nObjective: {obj_str} | Gap: {gap_str} | Runtime: {run_str} sec",
        fontsize=36, fontweight="bold", pad=20)

    legend_elements = [
        mpatches.Patch(color=COLOR_BLACK, label="Total Loss"),
        mpatches.Patch(color=COLOR_ORANGE, label="Partial Loss"),
        mpatches.Patch(color=COLOR_YELLOW, label="No Loss"),
        mpatches.Patch(color=COLOR_GREEN, label="Intact Forest"),
        mpatches.Patch(color=COLOR_WATER, label="Water Source"),
        mpatches.Patch(facecolor="none", edgecolor="#c0392b", linewidth=3, label="Ignition"),
        Line2D([0], [0], color="black", linestyle='-', linewidth=4, label="Forest Road"),
        Line2D([0], [0], marker="s", color="w", label="Engine", markerfacecolor="gray", markersize=18),
        Line2D([0], [0], marker="^", color="w", label="FRV", markerfacecolor="gray", markersize=18),
        Line2D([0], [0], marker="o", color="w", label="Heli", markerfacecolor="gray", markersize=18),
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.05),
              ncol=5, frameon=True, framealpha=0.9, fancybox=True, shadow=True, fontsize=20)

    plt.tight_layout()
    file_name = f"Detail_Map_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(OUT_DIR / file_name, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# =========================================================
# 5) AKADEMİK BİRLEŞİK GRAFİK (GROUPED BAR CHART)
# =========================================================
def draw_combined_academic_plots(v_df, t_df, x_df, save_dir: Path, ref_name: str):
    # Veri hazırlığı
    # Sheetlerden gelen kolon adları bazen değişken olabilir, son sütunu value varsayıyoruz
    v_df = v_df.rename(columns={v_df.columns[-1]: "val"})
    t_df = t_df.rename(columns={t_df.columns[-1]: "val"})
    x_df = x_df.rename(columns={x_df.columns[-1]: "x_ik"})

    # Atanmış (x=1) araçları bul
    assigned = x_df.loc[x_df["x_ik"] > 0.5, ["i", "k"]].drop_duplicates()

    # Merge işlemleri
    v_merged = v_df.merge(assigned, on=["i", "k"], how="inner")
    t_merged = t_df.merge(assigned, on=["i", "k"], how="inner")

    if v_merged.empty:
        print("Grafik çizilecek veri bulunamadı.")
        return

    # Sadece atama yapılan (active) node'ları al
    active_nodes = sorted(v_merged["i"].unique())

    # Grafik Ayarları
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, dpi=300)

    # Renkler ve Desenler
    cmap = plt.cm.viridis
    colors = [cmap(i) for i in np.linspace(0.2, 0.8, 10)]  # Yedek renk havuzu
    hatches = ['', '///', '...', '\\\\\\', 'xx', 'OO', '--', '++']

    # Maksimum araç sayısına göre bar genişliği ayarla
    # Her node'a en fazla kaç araç gitmiş?
    max_k_per_node = v_merged.groupby("i")["k"].count().max()
    if pd.isna(max_k_per_node) or max_k_per_node == 0: max_k_per_node = 1

    # Bir grup için toplam genişlik 0.8 olsun.
    total_width = 0.8
    bar_width = total_width / max_k_per_node

    # --- PLOTTING ---
    for i_idx, node in enumerate(active_nodes):
        # Bu node'a ait veriler
        node_v = v_merged[v_merged["i"] == node].sort_values("k")
        node_t = t_merged[t_merged["i"] == node].sort_values("k")

        # Bu node'a kaç araç gitmiş?
        num_vehicles = len(node_v)

        # Çubukları ortalamak için başlangıç ofseti
        # Örn: 2 araç varsa, merkezden -width/2 ve +width/2
        # Formül: (j - (num-1)/2) * width

        for j in range(num_vehicles):
            offset = (j - (num_vehicles - 1) / 2) * bar_width

            # Verileri çek
            val_v = node_v.iloc[j]["val"]
            # t verisini k'ya göre eşleştirerek bul
            k_val = node_v.iloc[j]["k"]
            val_t_row = node_t[node_t["k"] == k_val]
            val_t = val_t_row.iloc[0]["val"] if not val_t_row.empty else 0

            # Görsel Stil
            col = colors[j % len(colors)]
            hatch = hatches[j % len(hatches)]

            # Üst Grafik (Arrival Time)
            ax1.bar(i_idx + offset, val_v, width=bar_width * 0.9,
                    color=col, edgecolor="black", linewidth=0.8, hatch=hatch, alpha=0.9,
                    label=f"Veh {k_val}" if i_idx == 0 else "")  # Lejant için sadece ilk döngüde label

            # Alt Grafik (Dispatch Time)
            ax2.bar(i_idx + offset, val_t, width=bar_width * 0.9,
                    color=col, edgecolor="black", linewidth=0.8, hatch=hatch, alpha=0.9)

    # Eksen Etiketleri ve Format
    ax1.set_ylabel(r"Arrival Time ($v_{ik}$)", fontsize=16, fontweight='bold')
    ax2.set_ylabel(r"Dispatch Time ($t_{ik}$)", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Node Index ($i$)", fontsize=16, fontweight='bold')

    ax2.set_xticks(range(len(active_nodes)))
    ax2.set_xticklabels(active_nodes, fontsize=12)

    for ax in [ax1, ax2]:
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)

    fig.suptitle(f"Temporal Distribution of Resources (Grouped by Node)\nSource: {ref_name}", fontsize=20, y=0.96)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    plt.savefig(save_dir / f"Arrival-dispatch graphGrouped_{ts}.png", bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 6) MAIN
# =========================================================
if __name__ == "__main__":
    wildfire_data = load_wildfire_data_final(EXCEL_PATH)
    plot_detail_map(wildfire_data)

    res_path = _latest_results_file(RESULTS_DIR)
    if res_path:
        try:
            v_df = pd.read_excel(res_path, sheet_name="v")
            x_df = pd.read_excel(res_path, sheet_name="x")

            # t veya ts sayfası kontrolü
            xl = pd.ExcelFile(res_path)
            t_sheet = "t" if "t" in xl.sheet_names else "ts"
            t_df = pd.read_excel(res_path, sheet_name=t_sheet)

            draw_combined_academic_plots(v_df, t_df, x_df, OUT_DIR, res_path.name)
        except Exception as e:
            print(f"Hata: {e}")
            import traceback

            traceback.print_exc()