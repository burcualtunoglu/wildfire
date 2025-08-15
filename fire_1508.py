import math, ast
import pandas as pd
import gurobipy as gp
from pathlib import Path
from gurobipy import Model, GRB
from datetime import datetime

BASE_DIR = Path(r"C:\Users\Mert\deprem\wildfire")
FILE_NAME = "inputs_fire.xlsx"
XLS_PATH  = BASE_DIR / FILE_NAME
SHEET     = "inputs_df"

if not XLS_PATH.exists():
    raise FileNotFoundError(f"Excel bulunamadı: {XLS_PATH}")

df = pd.read_excel(XLS_PATH, sheet_name=SHEET)

required_cols = {
    "node_id", "x_coordinate", "y_coordinate",
    "value_at_start", "value_degradation_rate", "neighborhood_list"
}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Eksik sütun(lar): {sorted(missing)}")

# N: düğümler
N = df["node_id"].astype(int).tolist()

# i+ : komşuluk kümeleri
def _parse_neighbors(x):
    if pd.isna(x): return []
    if isinstance(x, (list, tuple)): return list(x)
    try:
        parsed = ast.literal_eval(str(x))
        return list(parsed) if isinstance(parsed, (list, tuple)) else [int(parsed)]
    except Exception:
        return [int(s.strip()) for s in str(x).strip("[]").split(",") if s.strip()!=""]

neighbors = {
    int(r.node_id): set(int(j) for j in _parse_neighbors(r.neighborhood_list))
    for r in df.itertuples(index=False)
}

# Parametreler ---------------------------------------------------------
# τ_i ve β_i
tau  = {int(r.node_id): float(r.value_at_start) for r in df.itertuples(index=False)}
beta = {int(r.node_id): float(r.value_degradation_rate) for r in df.itertuples(index=False)}

# Koordinatlar
coords = {int(r.node_id): (float(r.x_coordinate), float(r.y_coordinate))
          for r in df.itertuples(index=False)}
K = [f"hk{idx:02d}" for idx in range(1, 21)]
# d_ij: Öklid mesafesi (koordinatlar km cinsinden bunu helikopter hızı ile dakikaya çevireceğiz)
# Ka-32 helikopteri için 230 km/h=3.83 km/dk
# 2.1) Depo düğümü ve genişletilmiş düğüm kümesi
depot = -1   # yeni bir ID; Excel'deki N ile çakışmasın
coords[depot] = (0.0, 0.0)
N0 = [depot] + N

# Hız (km/dk gibi): her k için sabit alıyorum; isterseniz vehicles sayfasından okuyun
V_DEFAULT = 3.83  # örnek: 230 km/saat -> 3.83 km/dk
v = {k: V_DEFAULT for k in K}

# Mesafe ve seyahat süreleri
d_ext = {}
tau2 = {}  # travel time
for i in N0:
    xi, yi = coords[i]
    for j in N0:
        if i == j:
            continue
        xj, yj = coords[j]
        d_ext[(i, j)] = math.hypot(xi - xj, yi - yj)
        tau2[(i, j)]   = d_ext[(i, j)] / v[K[0]]  # tüm k için aynı hızsa
# eğer k'ya özgü hız olacaksa: tau2[(i,j,k)] = d_ext[(i,j)] / v[k]


# W_i: toplam iş yükü
W = {i: 250000 for i in N}  #25 km2 alan için 250000000 litre su gerekli

# e_i: exogenous fire param
exogenous_nodes = {8, 11, 20}
e = {i: (1 if i in exogenous_nodes else 0) for i in N}

# t_i^m ve t_i^e: yayılma ve sönme zamanları (yer tutucu)
t_m = {i: 8 for i in N}   # örnek: 8 zaman birimi sonra yayılır
t_e = {i: 24 for i in N}  # örnek: 24 zaman birimi sonra tamamen söner




OMEGA_DEFAULT = 1667  # birim: litre/dakika
omega = {k: OMEGA_DEFAULT for k in K}

M = 1e6  # Big-M

model = Model("Wildfire_Suppression")

# --- Karar değişkenleri ------------------------------------------------------
a = model.addVars(N0, K, vtype=GRB.BINARY, name="a")   # a_{ik} ∈ {0,1} : araç k, düğüm i'ye atanırsa 1
t_arr = model.addVars(N0, K, lb=0.0, ub=M, vtype=GRB.CONTINUOUS, name="t") # t_{ik} ≥ 0 : araç k'nın düğüm i'ye varış zamanı
delta = model.addVars(N0, K, lb=0.0, ub=M, vtype=GRB.CONTINUOUS, name="delta") # δ_{ik} ≥ 0 : araç k'nın düğüm i'de işlem (müdahale) süresi
y = model.addVars(N0, vtype=GRB.BINARY, name="y")  # y_i ∈ {0,1} : düğüm i'de yangın gerçekleşirse 1
u_a = model.addVars(N0, vtype=GRB.BINARY, name="u_a")  # u_i^a ∈ {0,1} : i düğümündeki yangın planlama ufku içinde herhangi bir anda kontrol altına alınırsa 1
u_b = model.addVars(N0, vtype=GRB.BINARY, name="u_b") # u_i^b ∈ {0,1} : i düğümündeki yangın, t_i^m (yayılmanın orta noktası) öncesinde kontrol altına alınırsa 1
t_c = model.addVars(N0, lb=0.0, ub=M, vtype=GRB.CONTINUOUS, name="t_c") # t_i^c ≥ 0 : i düğümünün kontrol altına alındığı zaman
t_start = model.addVars(N0, lb=0.0, ub=M, vtype=GRB.CONTINUOUS, name="t_start") # t_i^s ≥ 0 : i düğümünde yangının başlangıç zamanı
Arc = [(i, j) for i in N for j in neighbors[i]]  # j ∈ i^+  #Yayılım ve neden-sonuç ikilileri için kenar seti
z = model.addVars(Arc, vtype=GRB.BINARY, name="z") # z_{ij} ∈ {0,1} : yangın i'den j'ye yayılırsa 1
q = model.addVars(Arc, vtype=GRB.BINARY, name="q") # q_{ij} ∈ {0,1} : j düğümündeki yangının nedeni i ise 1
x = model.addVars(N0, N0, K, vtype=GRB.BINARY, name="x")  # x[i,j,k], i!=j kullanacağız
#
rev_neighbors = {j: set() for j in N}              # j^+ kümeleri
for i, js in neighbors.items():
    for j in js:
        rev_neighbors[j].add(i)


# ================================
# AŞAMA 1: Başlangıç yangınlarına rotalama (yalın çekirdek)
# ================================

# 0) Depo için değişkenleri sabitle / temizle
model.addConstrs((a[depot, k] == 0 for k in K),          name="depot_no_assignment")
model.addConstrs((delta[depot, k] == 0.0 for k in K),    name="depot_zero_service")
model.addConstrs((t_arr[depot, k] == 0.0 for k in K),    name="depot_time_zero")
model.addConstr( y[depot] == 0,                          name="no_fire_at_depot")


Omega_sum = sum(omega[k] for k in K)            # toplam debi [L/dk]
M_DELTA_i = {i: math.ceil(W[i]/max(Omega_sum,1)) + 5 for i in N}  # dk, + güvenlik payı
TAU_MAX = max(tau2.values()) if tau2 else 0.0
M_TIME  = TAU_MAX + max(M_DELTA_i.values(), default=0.0) + 10.0  # güvenlik payı
# 8) Zaman tutarlılığı (time propagation)
model.addConstrs((
    t_arr[j, k] >= t_arr[i, k] + tau2[(i, j)] - M_TIME*(1 - x[i, j, k])
    for k in K for i in N0 for j in N if i != j
), name="time_propagation_no_service")

#Araç Kullanımı----------------------------------------------
# Helikopter kullanım göstergesi - Yeni Karar değişkeni
u = model.addVars(K, vtype=GRB.BINARY, name="used")
# Depodan çıkış = kullanıldı (0/1)
model.addConstrs((
    gp.quicksum(x[depot, j, k] for j in N) == u[k]
    for k in K
), name="departure_iff_used")
# Kullanılmadıysa atama olamaz; kullanıldıysa çoklu atama serbest (üst sınırı N)
model.addConstrs((
    gp.quicksum(a[i, k] for i in N) <= len(N) * u[k]
    for k in K
), name="assign_only_if_used")

# *) Depodan çıkış/dönüş ve ilk atama
model.addConstrs((gp.quicksum(x[depot, j, k] for j in N) <= 1 for k in K),
                 name="one_departure_max")
model.addConstrs((gp.quicksum(x[i, depot, k] for i in N) ==
                  gp.quicksum(x[depot, j, k] for j in N) for k in K),
                 name="return_balance")
model.addConstrs((x[depot, j, k] <= y[j] for j in N for k in K),
                 name="first_leg_to_initial_fire")

# EK KISIT
model.addConstrs((x[i, i, k] == 0 for i in N0 for k in K), name="no_self_loop")

# 1) Workload Satisfaction if Fire Controlled
model.addConstrs((
    gp.quicksum(omega[k] * delta[i, k] for k in K) >= W[i] * y[i]
    for i in N
), name="workload_water_cover")

# i düğümünden çıkan kenarlar için uygun M: en kötü (travel + i’deki max servis) + pay
M_TIME_i = {i: TAU_MAX + M_DELTA_i[i] + 10.0 for i in N}
M_TIME_0 = TAU_MAX + 10.0  # depo satırı için (delta=0)

# Varış süresi
model.addConstrs((t_arr[i,k] <= (M_TIME_i[i] if i!=depot else M_TIME_0) * a[i,k]
                  for i in N for k in K), name="arrival_active_if_assigned")

# 2a) Processing Duration Linked to Assignment
model.addConstrs((delta[i,k] <= M * a[i,k] for i in N for k in K),
                 name="service_active_if_assigned")

#2b) Processing Ends at Suppression Time
model.addConstrs((t_arr[i,k]+delta[i,k]<=t_c[i]+M*(1-a[i,k]) for i in N for k in K),name="processing ends")

# 2c) Only Assign Vehicles to Burning Nodes
model.addConstrs((a[i,k] <= y[i] for i in N for k in K),
                 name="assign_only_if_fire")
# a[i,k] = 1  ==>  u_a[i] = 1
model.addConstrs((u_a[i] >= a[i,k] for i in N for k in K),
                 name="ua_if_assigned")
# u_a = OR_k a[i,k]
model.addConstrs((u_a[i] <= gp.quicksum(a[i,k] for k in K) for i in N),
                 name="ua_only_if_assigned")
#  Atama sadece yangın olan düğümlere
model.addConstrs((a[i,k] <= y[i] for i in N for k in K),
                 name="assign_only_if_fire")

# 3a) Suppression Must Finish Before Burnout
# model.addConstrs((t_c[i]<=t_e[i]*u_a[i] for i in N),name="before burnout")

# 3b) Early Control Timing (Before Spread)
model.addConstrs((t_c[i]<=t_m[i]+M*(1-u_b[i]) for i in N),name="before spread")
model.addConstrs((t_c[i]>=t_m[i]-M*u_b[i] for i in N),name="before spread_2")
# 3c) Early Control Implies Suppression
model.addConstrs((u_b[i]<=u_a[i] for i in N),name="early control")

# 4)Fire Spreads if Not Controlled Before t_m[i]
model.addConstrs((gp.quicksum(z[i, j] for j in neighbors[i]) == len(neighbors[i]) * (y[i] - u_b[i]) for i in N),name="spread_logic")
# 5) Fire Ignition Attribution
model.addConstrs((q[i, j] <= z[i, j] for (i, j) in Arc),name="cause_implies_spread")
# exogenous olanlarda neden atanamaz
model.addConstrs((q[i, j] <= 1 - e[j] for (i, j) in Arc),name="no_cause_if_exogenous")
# # j’deki yangının nedeni
model.addConstrs((gp.quicksum(q[i, j] for i in rev_neighbors[j]) == y[j] - e[j] for j in N),name="cause_balance")

model.addConstrs((t_c[i]-t_start[i]>=0 for i in N),name="must positive")




# 6) Fire Start Time Definition 
# model.addConstrs((
#     t_start[j] == gp.quicksum(t_m[i] * q[i, j] for i in rev_neighbors[j])
#     for j in N
# ), name="tstart_from_cause_tm")

# t_s_j = 0  if e_j = 1
# for j in N:
#     if e[j] == 1:
#         model.addConstr(t_start[j] == 0.0, name=f"tstart_zero_if_exogenous_{j}")



# 9) Geçici amaç: toplam seyahat + servis süresi minimizasyonu
# model.setObjective(
#     gp.quicksum(tau2[(i, j)] * x[i, j, k] for k in K for i in N0 for j in N if i != j) + \
#     gp.quicksum(delta[i, k] for i in N for k in K),
#     GRB.MINIMIZE
# )
model.setObjective(
    gp.quicksum(tau[i] - beta[i]* (t_c[i] - t_start[i]) for i in N),
    GRB.MAXIMIZE
)


model.optimize()
if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
    print(f"Optimal değer: {model.objVal}")
    for i in N:
        print(f"Node {i}: y={y[i].X}, u_a={u_a[i].X}, t_s={t_start[i].X:.2f}, t_c={t_c[i].X:.2f}")



timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
import os
base_dir = Path(r"C:\Users\Mert\deprem\wildfire")
results_dir = base_dir / "results"
os.makedirs(results_dir, exist_ok=True)

out_path = results_dir / f"wildfire_solution_nonzero_{timestamp}.xlsx"

ok_status = {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED}
if model.status not in ok_status:
    raise RuntimeError(f"Model feasible bir çözüm üretmedi (status={model.status}).")


def vec_to_df(var_dict, index_name, var_name):
    rows = []
    for i in N:
        val = var_dict[i].X
        if abs(val) > 1e-9:
            rows.append({index_name: i, var_name: val})
    return pd.DataFrame(rows)

def mat_to_df(var_dict, index_names, var_name, i_list, j_list):
    rows = []
    for i in i_list:
        for j in j_list(i) if callable(j_list) else j_list:
            val = var_dict[i, j].X
            if abs(val) > 1e-9:
                rows.append({index_names[0]: i, index_names[1]: j, var_name: val})
    return pd.DataFrame(rows)

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    # Meta bilgiler
    meta = {
        "model_name": [model.ModelName],
        "status":     [int(model.status)],
        "obj_value":  [getattr(model, "objVal", float("nan"))],
        "runtime_s":  [getattr(model, "Runtime", float("nan"))],
        "gap":        [getattr(model, "MIPGap", float("nan")) if hasattr(model, "MIPGap") else float("nan")]
    }
    pd.DataFrame(meta).to_excel(writer, sheet_name="meta", index=False)

    # Tek indeksli
    vec_to_df(y,        "i", "y").to_excel(writer, sheet_name="y_i", index=False)
    vec_to_df(u_a,      "i", "u_any").to_excel(writer, sheet_name="u_a_i", index=False)
    vec_to_df(u_b,      "i", "u_before").to_excel(writer, sheet_name="u_b_i", index=False)
    vec_to_df(t_c,      "i", "t_ctrl").to_excel(writer, sheet_name="t_c_i", index=False)
    vec_to_df(t_start,  "i", "t_start").to_excel(writer, sheet_name="t_start_i", index=False)

    # Çift indeksli (i,k)
    mat_to_df(a,     ("i","k"), "a",     i_list=N, j_list=K).to_excel(writer, sheet_name="a_ik", index=False)
    mat_to_df(t_arr, ("i","k"), "t",     i_list=N, j_list=K).to_excel(writer, sheet_name="t_ik", index=False)
    mat_to_df(delta, ("i","k"), "delta", i_list=N, j_list=K).to_excel(writer, sheet_name="delta_ik", index=False)

    # Çift indeksli (i,j)
    mat_to_df(z, ("i","j"), "z", i_list=N, j_list=lambda i: neighbors[i]).to_excel(writer, sheet_name="z_ij", index=False)
    mat_to_df(q, ("i","j"), "q", i_list=N, j_list=lambda i: neighbors[i]).to_excel(writer, sheet_name="q_ij", index=False)

    # Üç indeksli (i,j,k) için x[i,j,k]
    rows_x = []
    for i in N0:
        for j in N0:
            if i == j:
                continue
            for k in K:
                val = x[i, j, k].X
                if abs(val) > 1e-9:
                    rows_x.append({"i": i, "j": j, "k": k, "x": val})
    pd.DataFrame(rows_x).to_excel(writer, sheet_name="x_ijk", index=False)

