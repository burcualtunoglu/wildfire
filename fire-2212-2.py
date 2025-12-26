# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB
from yeni import load_wildfire_data_final
import pandas as pd
excel_path = r"C:\Users\Mert\deprem\wildfire\inputs_to_load-5x5_yyo.xlsx"

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

m.addConstrs(( p[i] <= v0[i] -beta[i]*(tc[i]-ts[i]) for i in N),
    name="cons 2")

m.addConstrs(( p[i] <= v0[i]*(u_pre[i]+u_post[i]+1-y[i]) for i in N),
    name="cons 3")


delta = m.addVars(N, K, vtype=GRB.CONTINUOUS, lb=0.0, name="delta")



for i in N:
    for k in K:
               m.addConstr(
            delta[i, k] >= su[i] * (v[i, k] - ts[i]) - M * (1 - x[i, k]),
            name=f"delta_lo[{i},{k}]")

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

m.addConstrs((omega[i] <= OMEGA_MAX * (u_pre[i] + u_post[i]) for i in N), name="omega_off_if_no_control")


# A) su talebi – u_pre/u_post’a bağla
for i in N:
    m.addGenConstrIndicator(u_pre[i], 1,
        gp.quicksum(mu[k]*s[i,k] for k in K) >= (omega[i]+500), name=f"dem_suppress_pre[{i}]")
    m.addGenConstrIndicator(u_post[i], 1,
        gp.quicksum(mu[k]*s[i,k] for k in K) >= (omega[i]+500), name=f"dem_suppress_post[{i}]")
    m.addConstrs(
        (s[i, k] <= M * (u_pre[i] + u_post[i]) for k in K),
        name=f"no_water_if_no_control[{i}]"
    )
for i in N:
    for k in K:
        m.addConstr(v[i, k] >= t[i, k] + d[i, k] - M * (1 - x[i, k]), name=f"v_eq_td_lo[{i},{k}]")
        m.addConstr(v[i, k] <= t[i, k] + d[i, k] + M * (1 - x[i, k]), name=f"v_eq_td_up[{i},{k}]")



# # (i) Atanmamışsa değişkenleri sıfıra yaklaştır (upper bound)
m.addConstrs((t[i,k] <= M * x[i,k] for i in N for k in K), name="t_zero_if_not_assigned")
m.addConstrs((s[i,k] <= M * x[i,k] for i in N for k in K), name="s_zero_if_not_assigned")
m.addConstrs((v[i,k] <= M * x[i,k] for i in N for k in K), name="v_zero_if_not_assigned")
#
# # (ii) Varış yangın başlangıcından önce olmasın
m.addConstrs((v[i,k] >= ts[i] - M * (1 - x[i,k]) for i in N for k in K), name="eq_16")

# (iii) Bastırma bitişi kontrol zamanını aşmasın kısıt 15???
m.addConstrs((v[i,k] + s[i,k] <= tc[i] + M * (1 - x[i,k]) for i in N for k in K), name="eq_15")

m.addConstrs((x[i,k]<=y[i] for i in N for k in K),
    name="cons 8")

m.addConstrs((u_pre[i]+u_post[i] >= x[i,k] for i in N for k in K),
    name="cons 9")

m.addConstrs((u_pre[i]+u_post[i] <= gp.quicksum(x[i,k] for k in K) for i in N ),
    name="cons 10")

m.addConstrs((gp.quicksum(x[i,k] for i in N) <= 1 for k in K ),name="11")


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


m.addConstrs((ts[j]==gp.quicksum(tm[i]*q[i,j] for i in neighbors[j])  for j in N ),name="cons 22")

m.addConstrs((ts[j]<= tm[i]+ M*(1-z[i,j]) for j in N for i in neighbors[j] ),
                 name="cons 23")

m.addConstr(gp.quicksum(y[i] for i in Na) == len(Na),
    name="cons 24")

m.addConstr( gp.quicksum(ts[i] for i in Na) == 0,name="cons 25")

m.addConstrs(( tm[i] == ts[i] + alpha / degrade[i] for i in N ),
    name="cons 26")


m.addConstrs(( te[i] == tm[i] + alpha / ameliorate[i] for i in N ),
    name="cons 27")

# objective function
m.setObjective(gp.quicksum(p[i] for i in N), GRB.MAXIMIZE)



m.setParam("MIPGap", 0.02)
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
        # 1. Meta Sayfası
        meta_rows = [("InputFile", str(excel_inputs_path)), ("Status", model.Status),
                     ("Objective", getattr(model, "ObjVal", float("nan"))),
                     ("Runtime_sec", getattr(model, "Runtime", float("nan")))]
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
    excel_inputs_path=r"C:\Users\Mert\deprem\wildfire\inputs_to_load-5x5_yyo.xlsx",
    d_matrix=data["d_matrix"],       # Seyahat süreleri matrisi
    vehicle_info=data["vehicle_info"] # Araç teknik özellikleri listesi
)

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ast
from pathlib import Path
from datetime import datetime
from matplotlib.colors import to_rgb, Normalize
from matplotlib.lines import Line2D

# === AKADEMİK RENK PALETİ VE AYARLAR ===
COLOR_BLACK = "#2c3e50"  # Never Controlled
COLOR_ORANGE = "#d35400"  # Controlled (Partial Loss)
COLOR_YELLOW = "#f1c40f"  # Controlled (No Loss)
COLOR_GREEN = "#27ae60"  # No Fire (Protected Area)

BASE_COLORS = {
    "A": "#1f77b4", "B": "#9467bd", "C": "#e377c2", "D": "#8c564b", "E": "#17becf"
}

EXCEL_PATH = r"C:\Users\Mert\deprem\wildfire\inputs_to_load-5x5_yyo.xlsx"
RESULTS_DIR = Path(r"C:\Users\Mert\deprem\wildfire\result")
OUT_DIR = Path(r"C:\Users\Mert\deprem\wildfire\maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. VERİ YÜKLEME VE YARDIMCI FONKSİYONLAR
# ==========================================

def load_wildfire_data_final(excel_path: str):
    xls = pd.ExcelFile(excel_path)
    df_nodes = pd.read_excel(xls, sheet_name='inputs_df')
    df_bases = pd.read_excel(xls, sheet_name='bases')

    vehicle_records = []
    current_k = 1
    vehicle_types = ['Helicopter', 'Fire Engine', 'FRV']

    for _, row in df_bases.iterrows():
        b_id = str(row['Base']) if 'Base' in df_bases.columns else str(row['base_id'])
        bx, by = float(row['x_coordinate']), float(row['y_coordinate'])

        for v_type in vehicle_types:
            if v_type in df_bases.columns and pd.notna(row[v_type]):
                count = int(row[v_type])
                for _ in range(count):
                    vehicle_records.append({
                        'id': current_k, 'base_id': b_id, 'type': v_type,
                        'bx': bx, 'by': by
                    })
                    current_k += 1
    return {"vehicle_info": vehicle_records, "nodes_df": df_nodes, "bases_df": df_bases}


def read_sheet(path, sheet, key="i"):
    try:
        df = pd.read_excel(path, sheet_name=sheet)
        return dict(zip(df[key], df[df.columns[-1]]))
    except:
        return {}


def read_meta_info(path):
    try:
        df_meta = pd.read_excel(path, sheet_name="meta")
        meta_dict = dict(zip(df_meta.iloc[:, 0], df_meta.iloc[:, 1]))
        obj = meta_dict.get("Objective", "N/A")
        runtime = meta_dict.get("Runtime_sec", "N/A")
        if isinstance(obj, str): obj = float(obj.replace(',', '.'))
        if isinstance(runtime, str): runtime = float(runtime.replace(',', '.'))
        return obj, runtime
    except:
        return "N/A", "N/A"


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


def fnum(v):
    return f"{v:.2f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "-"


def get_map_extent(df_nodes, df_bases):
    all_x = np.concatenate([df_nodes["x_coordinate"].values, df_bases["x_coordinate"].values])
    all_y = np.concatenate([df_nodes["y_coordinate"].values, df_bases["y_coordinate"].values])
    x_coords = sorted(df_nodes["x_coordinate"].unique())
    y_coords = sorted(df_nodes["y_coordinate"].unique())
    dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 5.0
    dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 5.0
    margin = 5.0
    extent = [min(all_x) - margin, max(all_x) + margin, min(all_y) - margin, max(all_y) + margin]
    return extent, dx, dy


def draw_bases(ax, df_bases):
    for _, row in df_bases.iterrows():
        b_id, bx, by = str(row['Base']), float(row['x_coordinate']), float(row['y_coordinate'])
        b_color = BASE_COLORS.get(b_id, '#333333')
        ax.scatter(bx, by, s=600, marker='s', color=b_color, edgecolors='white', linewidth=2, zorder=40)
        ax.text(bx, by, b_id, color='white', fontweight='bold', ha='center', va='center', fontsize=12, zorder=41)


# ==========================================
# 2. HARİTA 1: INITIAL MAP
# ==========================================

def plot_initial_map(data):
    df_nodes, df_bases = data["nodes_df"], data["bases_df"]
    extent, dx, dy = get_map_extent(df_nodes, df_bases)
    x_coords = sorted(df_nodes["x_coordinate"].unique())
    y_coords = sorted(df_nodes["y_coordinate"].unique())
    cmap, norm = plt.cm.Greens, Normalize(vmin=df_nodes["value_at_start"].min(), vmax=df_nodes["value_at_start"].max())
    fig, ax = plt.subplots(figsize=(12, 12))
    rgb_grid = np.zeros((len(y_coords), len(x_coords), 3))
    grid_extent = [min(x_coords) - dx / 2, max(x_coords) + dx / 2, min(y_coords) - dy / 2, max(y_coords) + dy / 2]
    for _, row in df_nodes.iterrows():
        xi, yi = x_coords.index(row["x_coordinate"]), y_coords.index(row["y_coordinate"])
        rgb_grid[yi, xi] = cmap(norm(row["value_at_start"]))[:3]
    ax.imshow(rgb_grid, origin="lower", aspect='equal', extent=grid_extent, interpolation='nearest', zorder=1)
    ax.set_xlim(extent[0], extent[1]);
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal');
    ax.grid(which='major', color='gray', linestyle='--', linewidth=0.5, zorder=0)

    for _, row in df_nodes.iterrows():
        nid, cx, cy = int(row["node_id"]), row["x_coordinate"], row["y_coordinate"]
        ax.text(cx, cy + dy * 0.1, str(nid), fontsize=12, fontweight='bold', ha='center', zorder=10)
        ax.text(cx, cy - dy * 0.1, f"Val: {row['value_at_start']:.1f}", fontsize=9, ha='center', zorder=10)

        # Başlangıç yangını ise etrafını kırmızı ile çiz
        if row.get("state") == 1:
            rect = mpatches.Rectangle((cx - dx / 2 + 0.05, cy - dy / 2 + 0.05), dx - 0.1, dy - 0.1,
                                      fill=False, edgecolor='red', linewidth=3, zorder=15)
            ax.add_patch(rect)

    draw_bases(ax, df_bases)
    ax.set_title("Initial Situation: Fire Sources and Node Values", fontsize=16, fontweight='bold')
    plt.savefig(OUT_DIR / f"initial_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300, bbox_inches='tight')


# ==========================================
# 3. HARİTA 2: DETAIL MAP
# ==========================================

def plot_detail_map(data):
    veh_map = {v['id']: {'type': v['type'], 'base': v['base_id']} for v in data["vehicle_info"]}
    df_nodes, df_bases = data["nodes_df"], data["bases_df"]
    results = sorted(RESULTS_DIR.glob("results_*.xlsx"))
    if not results: return
    res_path = max(results, key=lambda p: p.stat().st_mtime)

    # Verileri Oku
    obj_val, runtime = read_meta_info(res_path)
    p_map, y_status = read_sheet(res_path, "p"), read_sheet(res_path, "y")
    ts_map, tm_map, te_map, tc_map = read_sheet(res_path, "ts"), read_sheet(res_path, "tm"), \
        read_sheet(res_path, "te"), read_sheet(res_path, "tc")
    sumx, max_v, q_ones = read_x_sum(res_path), read_v_max(res_path), read_q(res_path)
    raw_omega = read_sheet(res_path, "omega") or read_sheet(res_path, "w")
    start_vals = dict(zip(df_nodes["node_id"], df_nodes["value_at_start"]))

    # Araç atamalarını ve atanmış düğümleri belirle
    try:
        df_x = pd.read_excel(res_path, sheet_name="x")
        node_vehicles = {}
        for _, row in df_x[df_x.iloc[:, -1] >= 0.5].iterrows():
            i, k = int(row['i']), int(row['k'])
            if i not in node_vehicles: node_vehicles[i] = []
            if k in veh_map: node_vehicles[i].append(veh_map[k])
    except:
        node_vehicles = {}

    # === SU MİKTARI GÜNCELLEMESİ ===
    # Eğer araç atanmışsa (node_vehicles içinde kayıt varsa) +500 ekle
    omega_map = {}
    for nid, val in raw_omega.items():
        if nid in node_vehicles and len(node_vehicles[nid]) > 0:
            omega_map[nid] = val + 500
        else:
            omega_map[nid] = val

    extent, dx, dy = get_map_extent(df_nodes, df_bases)
    x_coords = sorted(df_nodes["x_coordinate"].unique())
    y_coords = sorted(df_nodes["y_coordinate"].unique())
    fig, ax = plt.subplots(figsize=(16, 16))
    rgb_grid = np.zeros((len(y_coords), len(x_coords), 3))
    grid_extent = [min(x_coords) - dx / 2, max(x_coords) + dx / 2, min(y_coords) - dy / 2, max(y_coords) + dy / 2]

    # Boyama döngüsü
    for _, row in df_nodes.iterrows():
        nid = int(row["node_id"])
        xi, yi = x_coords.index(row["x_coordinate"]), y_coords.index(row["y_coordinate"])
        p, s, y = p_map.get(nid, 0.0), start_vals.get(nid, 0.0), y_status.get(nid, 0.0)

        if p <= 0.001:
            color = COLOR_BLACK
        elif p < (s - 0.001):
            color = COLOR_ORANGE
        elif y >= 0.5:
            color = COLOR_YELLOW
        else:
            color = COLOR_GREEN
        rgb_grid[yi, xi] = to_rgb(color)

    ax.imshow(rgb_grid, origin="lower", aspect='equal', extent=grid_extent, interpolation='nearest', zorder=1)
    ax.set_xlim(extent[0], extent[1]);
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal');
    ax.grid(which='major', color='white', linestyle='-', linewidth=2, zorder=2)

    for _, row in df_nodes.iterrows():
        nid, cx, cy = int(row["node_id"]), row["x_coordinate"], row["y_coordinate"]
        p_val, s_val, y_val = p_map.get(nid, 0.0), start_vals.get(nid, 0.0), y_status.get(nid, 0.0)

        # Hücre rengini tekrar belirleyelim
        if p_val <= 0.001:
            current_color = COLOR_BLACK
        elif p_val < (s_val - 0.001):
            current_color = COLOR_ORANGE
        elif y_val >= 0.5:
            current_color = COLOR_YELLOW
        else:
            current_color = COLOR_GREEN

        text_color = "black" if current_color == COLOR_YELLOW else "white"
        ax.text(cx, cy + dy * 0.3, str(nid), fontsize=16, fontweight='bold', ha='center', color=text_color, zorder=10)

        # Başlangıç yangını ise etrafını kırmızı ile çiz
        if row.get("state") == 1:
            rect = mpatches.Rectangle((cx - dx / 2 + 0.05, cy - dy / 2 + 0.05), dx - 0.1, dy - 0.1,
                                      fill=False, edgecolor='red', linewidth=3, zorder=15)
            ax.add_patch(rect)

        # Metin Yazımı
        if current_color == COLOR_GREEN:
            lines = [f"Val: {fnum(s_val)}"]
        else:
            lines = [f"Val: {fnum(s_val)}→{fnum(p_val)}"]
            if y_val >= 0.5:
                lines.append(
                    f"T: ({fnum(ts_map.get(nid))},{fnum(tm_map.get(nid))},{fnum(te_map.get(nid))})→{fnum(tc_map.get(nid))}")
                lines.append(f"Water: {fnum(omega_map.get(nid))}")
                lines.append(f"Team: Σx={sumx.get(nid, 0.0):.1f} | max(v)={fnum(max_v.get(nid, 0.0))}")
                lines.append(f"Ignite: {','.join(map(str, q_ones.get(nid, []))) if q_ones.get(nid) else '-'}")
            elif current_color == COLOR_BLACK:
                lines.append("Status: Not Controlled")

        ax.text(cx, cy - dy * 0.05, "\n".join(lines), fontsize=7.5, ha='center', va='center', color=text_color,
                fontweight='bold', zorder=10)

        assignments = node_vehicles.get(nid, [])
        if assignments:
            dot_rad = dx * 0.035;
            spacing = dot_rad * 3.2
            for idx, veh in enumerate(assignments):
                r_idx, c_idx = divmod(idx, 5)
                px = (cx - (min(len(assignments), 5) - 1) * spacing / 2) + c_idx * spacing
                py = (cy - dy / 2) + dot_rad * 1.8 + r_idx * spacing
                v_col = BASE_COLORS.get(veh['base'], 'gray')
                if veh['type'] == 'FRV':
                    ax.add_patch(
                        mpatches.RegularPolygon((px, py), 3, radius=dot_rad * 1.2, color=v_col, ec='white', lw=0.5,
                                                zorder=20))
                elif veh['type'] == 'Helicopter':
                    ax.add_patch(
                        mpatches.Circle((px, py), radius=dot_rad * 1.2, color=v_col, ec='white', lw=0.5, zorder=20))
                else:
                    ax.add_patch(mpatches.Rectangle((px - dot_rad, py - dot_rad), dot_rad * 2, dot_rad * 2, color=v_col,
                                                    ec='white', lw=0.5, zorder=20))

    draw_bases(ax, df_bases)
    obj_str = f"{obj_val:.3f}" if isinstance(obj_val, (int, float)) else obj_val
    run_str = f"{runtime:.3f}" if isinstance(runtime, (int, float)) else runtime
    title_str = f"Detailed Wildfire Optimization Analysis\nObjective: {obj_str} | Runtime: {run_str} sec"
    ax.set_title(title_str, fontsize=16, fontweight='bold', pad=20)

    legend_elements = [
        mpatches.Patch(color=COLOR_BLACK, label=r'Never Controlled'),
        mpatches.Patch(color=COLOR_ORANGE, label=r'Controlled (Partial Loss)'),
        mpatches.Patch(color=COLOR_YELLOW, label='Controlled (No Loss)'),
        mpatches.Patch(color=COLOR_GREEN, label='No Fire'),
        mpatches.Patch(facecolor='none', edgecolor='red', linewidth=2, label='Initial Fire Source'),
        Line2D([0], [0], marker='s', color='w', label='Fire Engine (Square)', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='FRV (Triangle)', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Helicopter (Circle)', markerfacecolor='gray', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)
    plt.savefig(OUT_DIR / f"detail_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    wildfire_data = load_wildfire_data_final(EXCEL_PATH)
    plot_initial_map(wildfire_data)
    plot_detail_map(wildfire_data)
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from datetime import datetime

# === 1. YOL VE DEĞİŞKEN TANIMLAMALARI (Hata Alınan Kısım) ===
# Bu bölümü kendi dosya sisteminize göre doğrulayınız.
BASE_DIR = Path(r"C:\Users\Mert\deprem\wildfire")
RESULTS_DIR = BASE_DIR / "result"
OUT_DIR = BASE_DIR / "maps"

# Çıktı dizini yoksa oluştur
OUT_DIR.mkdir(parents=True, exist_ok=True)

# En güncel sonuç dosyasını bulma (res_path tanımı)
result_files = sorted(RESULTS_DIR.glob("results_*.xlsx"))
if not result_files:
    raise FileNotFoundError(f"{RESULTS_DIR} dizininde 'results_*.xlsx' formatında dosya bulunamadı.")

res_path = max(result_files, key=lambda p: p.stat().st_mtime)

# --- 2. AKADEMİK GÖRSEL AYARLAR ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "figure.dpi": 300
})


def draw_combined_academic_plots(v_df, t_df, x_df, save_dir, ref_name):
    """v_ik ve t_ik değerlerini alt alta iki panelde çizer ve kaydeder."""

    # Sütun isimlerini normalize et
    v_df = v_df.rename(columns={"v": "val", "value": "val", "v_ik": "val"})
    t_df = t_df.rename(columns={"t": "val", "value": "val", "t_ik": "val"})
    x_df = x_df.rename(columns={"x": "x_ik", "value": "x_ik", "x_ik": "x_ik"})

    # Atanmış (x_ik=1) olanları filtrele
    assigned = x_df.loc[x_df["x_ik"] > 0.5, ["i", "k"]].drop_duplicates()

    # Verileri birleştir (Merge)
    v_merged = v_df.merge(assigned, on=["i", "k"], how="inner").sort_values(["i", "val"])
    t_merged = t_df.merge(assigned, on=["i", "k"], how="inner").sort_values(["i", "val"])

    if v_merged.empty or t_merged.empty:
        print("Uyarı: Atanmış veri (v veya t için) bulunamadı. Lütfen sonuç dosyasını kontrol edin.")
        return

    nodes = sorted(v_merged["i"].unique())

    # Alt alta iki grafik
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    cmap = mpl.colormaps.get_cmap("viridis")
    patterns = ["", "///", "\\\\\\", "xxx", "---"]
    w, gap = 0.18, 1.2

    for ax, data, label in zip([ax1, ax2], [v_merged, t_merged],
                               ["Arrival Time ($v_{ik}$)", "Dispatch Time ($t_{ik}$)"]):
        curr_x = 0.0
        ticks_pos = []

        for idx, node in enumerate(nodes):
            node_data = data[data["i"] == node]
            vals = node_data["val"].values
            m = len(vals)
            start = curr_x - ((m - 1) * w / 2 if m > 1 else 0)

            for j, val in enumerate(vals):
                ax.bar(start + j * w, val, width=w * 0.85,
                       color=cmap(idx / len(nodes)), edgecolor="black",
                       linewidth=0.6, hatch=patterns[j % len(patterns)])

            ticks_pos.append(curr_x)
            curr_x += gap

        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        if ax == ax2:
            ax.set_xlabel("Node Index ($i$)", fontsize=11, fontweight='bold')
            ax.set_xticks(ticks_pos)
            ax.set_xticklabels(nodes)

    fig.suptitle(f"Optimization Results: Temporal Distributions\nFile: {ref_name}",
                 fontsize=13, fontweight='bold', y=0.98)

    # Benzersiz İsimle Kaydet
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"combined_temporal_plot_{ts}.png"
    save_path = Path(save_dir) / filename

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"Birleşik grafik başarıyla oluşturuldu: {filename}")


# --- 3. ÇALIŞTIRMA MANTIĞI ---
if __name__ == "__main__":
    try:
        print(f"Kullanılan sonuç dosyası: {res_path.name}")

        # Dosyaları oku
        v_df = pd.read_excel(res_path, sheet_name="v")
        x_df = pd.read_excel(res_path, sheet_name="x")

        # Eğer t sayfası yoksa ts, tm gibi alternatifleri kontrol et
        try:
            t_df = pd.read_excel(res_path, sheet_name="t")
        except:
            print("Not: 't' sayfası bulunamadı, alternatif 'ts' sayfası deneniyor.")
            t_df = pd.read_excel(res_path, sheet_name="ts")

        draw_combined_academic_plots(v_df, t_df, x_df, OUT_DIR, res_path.name)

    except Exception as e:
        print(f"Hata oluştu: {e}")