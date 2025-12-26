# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB
from loader_wildfire_inputs import load_inputs, distribute_vehicles_priority,generate_vehicle_params,compute_d_param_variable



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
# 2. Araçları dağıtın (Base A'ya 5 tane, kalanı diğerlerine) assignments = distribute_vehicles_priority(K, base_locations) # 3. Araç özelliklerini (mu ve hız) türetin mu_dict, speed_dict = generate_vehicle_params(assignments) # 4. Süre matrisini (d_ik) hesaplayın # Dikkat: speed_dict veriyoruz, sabit sayı değil. d_matrix = compute_d_param_variable(nodes, speed_dict, K, base_locations, assignments) # Artık 'mu_dict' ve 'd_matrix' optimizasyon modeline girmeye hazır.

# 1. Üs Koordinatları (Mevcut yapınız)
BASES = {
    "A": (0.0, 0.0),  # Helikopter Üssü (Öklid, 120 km/s, 10000 lt)
    "B": (20.0, 10.0),  # Kara Üssü (Manhattan, 50 km/s, 2500 lt)
    "C": (5.0, 15.0),
    "D": (20.0, 20.0),
    "E": (10.0, 25.0)
}


# --- YARDIMCI FONKSİYON: Parametre Belirleyici ---
def get_vehicle_specs(assignments):
    """
    Araç atamalarına bakarak Hız ve Kapasite sözlüklerini üretir.
    "A" üssü -> Helikopter
    Diğerleri -> Kara Aracı
    """
    mu_dict = {}
    speed_dict = {}

    for k_id, base_name in assignments.items():
        if base_name == "A":
            # HELİKOPTER SPECS
            speed_dict[k_id] = 120.0  # km/saat
            mu_dict[k_id] = 10000.0  # litre
        else:
            # KARA ARACI SPECS
            speed_dict[k_id] = 50.0  # km/saat
            mu_dict[k_id] = 2500.0  # litre

    return mu_dict, speed_dict


# 2. Araçları Üslere Öncelikli Dağıt
# (Base A'ya 5 araç, kalanı diğerlerine)
# UYARI: distribute_vehicles_priority fonksiyonunda 'target_base="A"' olduğundan emin olun
# veya fonksiyonu çağırırken parametre olarak geçin.
try:
    # Eğer fonksiyonunuz parametre destekliyorsa:
    # vehicle_home = distribute_vehicles_priority(K, BASES, target_base="A", target_count=5)
    # Desteklemiyorsa ve kod içine gömüldüyse direkt çağırın:
    vehicle_home = distribute_vehicles_priority(K, BASES)
except NameError:
    print("HATA: 'distribute_vehicles_priority' fonksiyonu tanımlanmamış. Lütfen önceki cevaptaki fonksiyonu ekleyin.")
    raise

# (Opsiyonel) Kontrol Çıktısı
print("--- Araç Dağılım ve Özellik Özeti ---")
from collections import Counter

summary = Counter(vehicle_home.values())
for base, count in summary.items():
    v_type = "Helikopter" if base == "A" else "Kara Aracı"
    print(f"Üs {base}: {count} adet araç ({v_type})")
print("-------------------------------------")

# 3. Hız ve Kapasite (mu) Parametrelerini Oluştur
mu, speed_dict = get_vehicle_specs(vehicle_home)

# 4. Mesafeleri ve Süreleri Hesapla (d matrisi)
# Not: compute_d_param_variable fonksiyonunu kullanıyoruz (Variable Speed)
d = compute_d_param_variable(
    coords=coords,
    speed_dict=speed_dict,  # Artık sabit sayı değil, sözlük gidiyor
    K=K,
    base_locations=BASES,
    vehicle_assignments=vehicle_home
)

# Artık 'mu' sözlüğü ve 'd' matrisi modele girmeye hazır.
# 'mu_default' değişkenine artık ihtiyacınız yok çünkü her aracın mu'su 'mu' sözlüğünde.

# d'yi (i,k) anahtarlı tupledict yap (tek tip indeksleme)
if not isinstance(d, gp.tupledict):
    d = gp.tupledict({(i,k): d[(i,k)] for i in N for k in K})
# Bundan sonra HER YERDE: d[i,k] değil, d[i,k] yerine d[i,k] KULLANMAYIN; d[i,k] tupledict'te d[i,k] çalışır mı karışır.
# En güvenlisi: d[i,k] yerine d[i,k] değil d[i,k] yazmak yerine d[i,k] -> d[i,k] yine tupledict'te (i,k) ile çağrılır.
# Gurobi tupledict: d[i,k] ifadesi aslında d[i,k] şeklinde çalışır (iki argümanlı indexing). Bu nedenle aşağıda d[i,k] kullanıyorum.

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

m.addConstrs((omega[i] <= OMEGA_MAX * (u_pre[i] + u_post[i]) for i in N), name="omega_off_if_no_control")


w = m.addVars(N, K, vtype=GRB.CONTINUOUS, lb=0.0, name="w")  # delivered water (litre)

# Kapasite bağlantısı: atanırsa teslim edebilir; teslim üst sınırı mu[k]
m.addConstrs((w[i,k] <= mu[k] * x[i,k] for i in N for k in K), name="water_cap")

# Kontrol yoksa su teslimi olmasın (opsiyonel ama önerilir)
m.addConstrs((w[i,k] <= mu[k] * (u_pre[i] + u_post[i]) for i in N for k in K), name="no_water_without_control")

# Talep karşılama: u_pre veya u_post aktifse omega karşılanmalı
for i in N:
    m.addGenConstrIndicator(u_pre[i], 1,
        gp.quicksum(w[i,k] for k in K) >= omega[i], name=f"dem_suppress_pre[{i}]")
    m.addGenConstrIndicator(u_post[i], 1,
        gp.quicksum(w[i,k] for k in K) >= omega[i], name=f"dem_suppress_post[{i}]")


for i in N:
    for k in K:
        m.addConstr(v[i, k] >= t[i, k] + d[i, k] - M * (1 - x[i, k]), name=f"v_eq_td_lo[{i},{k}]")
        m.addConstr(v[i, k] <= t[i, k] + d[i, k] + M * (1 - x[i, k]), name=f"v_eq_td_up[{i},{k}]")



# (i) Atanmamışsa değişkenleri sıfıra yaklaştır (upper bound)
m.addConstrs((t[i,k] <= M * x[i,k] for i in N for k in K), name="t_zero_if_not_assigned")
m.addConstrs((s[i,k] <= M * x[i,k] for i in N for k in K), name="s_zero_if_not_assigned")
m.addConstrs((v[i,k] <= M * x[i,k] for i in N for k in K), name="v_zero_if_not_assigned")

# (ii) Varış yangın başlangıcından önce olmasın
m.addConstrs((v[i,k] >= ts[i] - M * (1 - x[i,k]) for i in N for k in K), name="arrive_after_start")

# (iii) Bastırma bitişi kontrol zamanını aşmasın
m.addConstrs((v[i,k] + s[i,k] <= tc[i] + M * (1 - x[i,k]) for i in N for k in K), name="finish_before_control")



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
                           excel_inputs_path,
                           out_dir=r"C:\Users\Mert\deprem\wildfire\result",
                           file_prefix="results",
                           # --- YENİ EKLENEN PARAMETRELER ---
                           d_matrix: dict = None,  # d[(i,k)] süreleri
                           vehicle_assignments: dict = None  # k -> Base Name eşleşmesi
                           ):
    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        raise RuntimeError(f"Çözülebilir bir çözüm bulunamadı (Status={model.Status}).")

    # Girdi parametrelerini oku
    excel_inputs_path = Path(excel_inputs_path)
    n_nodes, n_vehicles, params_all = _read_params(excel_inputs_path)

    # Dosya adı oluşturma
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    in_stem = excel_inputs_path.stem
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    xlsx = out_path / f"{file_prefix}__{in_stem}__{ts_str}.xlsx"

    # Gurobi meta verileri
    status = model.Status
    objval = getattr(model, "ObjVal", float("nan"))
    mipgap = getattr(model, "MIPGap", float("nan"))
    runtime = getattr(model, "Runtime", float("nan"))

    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
        # 1. Meta Sayfası
        meta_rows = [
            ("InputFile", str(excel_inputs_path)),
            ("Status", status),
            ("Objective", objval),
            ("MIPGap", mipgap),
            ("Runtime_sec", runtime),
            ("ExportTime", ts_str),
        ]
        meta = pd.DataFrame(meta_rows, columns=["metric", "value"])
        _safe_write(writer, "meta", meta)

        # 2. Parametreler (Opsiyonel)
        if params_all:
            params_df = pd.DataFrame([{"parameter": k, "value": v} for k, v in params_all.items()])
            _safe_write(writer, "parameters_inherited", params_df)

        # --- YENİ EKLENEN KISIM: Base'den Varış Süreleri Raporu ---
        # Eğer x değişkeni, d matrisi ve araç atamaları verildiyse bu raporu oluştur.
        if "x" in vars_dict and d_matrix is not None and vehicle_assignments is not None:
            deployment_rows = []
            x_var = vars_dict["x"]

            # x_ik değişkenlerini tara
            for (i, k), var in x_var.items():
                # Eğer araç k, düğüm i'ye gittiyse (x_ik = 1)
                # (Floating point hataları için > 0.5 kontrolü güvenlidir)
                if var.X is not None and var.X > 0.5:
                    base_name = vehicle_assignments.get(k, "Unknown")
                    travel_time = d_matrix.get((i, k), float("nan"))

                    deployment_rows.append({
                        "Vehicle_ID": k,
                        "Assigned_Base": base_name,
                        "Target_Node": i,
                        "Travel_Time_Hours": travel_time
                    })

            if deployment_rows:
                df_deploy = pd.DataFrame(deployment_rows)
                # Okunabilirlik için sıralayalım
                df_deploy = df_deploy.sort_values(by=["Assigned_Base", "Vehicle_ID"])
                _safe_write(writer, "Deployment_Details", df_deploy)
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
    out_dir=r"C:\Users\Mert\deprem\wildfire\result",
    file_prefix="results",

    # --- BURAYI EKLİYORUZ ---
    d_matrix=d,  # Yukarıda hesapladığınız d matrisi
    vehicle_assignments=vehicle_home  # Yukarıda hesapladığınız araç atamaları
)

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from matplotlib.colors import to_rgb, Normalize

# === AYARLAR: AKADEMİK RENK PALETİ (DURUM RENKLERİ) ===
COLOR_BLACK = "#2c3e50"  # Never Controlled
COLOR_ORANGE = "#d35400"  # Controlled (Partial Loss)
COLOR_YELLOW = "#f1c40f"  # Controlled (No Loss)
COLOR_GREEN = "#27ae60"  # No Fire

# === TANIYICILAR VE RENKLERİ (BASES) ===
BASES = {
    "A": (0.0, 0.0),
    "B": (20.0, 10.0),
    "C": (5.0, 15.0),
    "D": (20.0, 20.0),
    "E": (10.0, 25.0)
}

# === DÜZELTME: BASE RENKLERİ GÜNCELLENDİ ===
# Kırmızı, Yeşil ve Turuncu ÇIKARILDI.
# Yerine karışıklık yaratmayacak net renkler eklendi.
BASE_COLORS = {
    "A": "#1f77b4",  # Mavi (Blue)
    "B": "#9467bd",  # Mor (Purple)
    "C": "#e377c2",  # Pembe (Pink)
    "D": "#8c564b",  # Kahverengi (Brown)
    "E": "#17becf"  # Turkuaz (Cyan)
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


# === ALGORİTMA: ARAÇLARI ÜSLERE DAĞIT ===
def distribute_vehicles_to_bases_logic(all_vehicles, base_dict):
    K = sorted(list(all_vehicles))
    base_names = list(base_dict.keys())

    num_bases = len(base_names)
    num_vehicles = len(K)

    if num_bases == 0: return {}

    assignments = {}
    q, r = divmod(num_vehicles, num_bases)

    start_idx = 0
    for i, base_name in enumerate(base_names):
        count = q + 1 if i < r else q
        assigned_vehicles = K[start_idx: start_idx + count]
        for vehicle_id in assigned_vehicles:
            assignments[vehicle_id] = base_name
        start_idx += count

    return assignments


def get_node_incoming_bases(result_path):
    try:
        df_x = pd.read_excel(result_path, sheet_name="x")
        all_vehicle_ids = df_x['k'].unique()
        veh_to_base_map = distribute_vehicles_to_bases_logic(all_vehicle_ids, BASES)

        df_active = df_x[df_x.iloc[:, -1] >= 0.5]

        node_colors = {}
        for _, row in df_active.iterrows():
            i = int(row['i'])
            k = int(row['k'])
            base_name = veh_to_base_map.get(k)
            if base_name:
                if i not in node_colors:
                    node_colors[i] = []
                node_colors[i].append(base_name)
        return node_colors
    except Exception as e:
        print(f"Araç haritalamada hata: {e}")
        return {}


# === VERİLERİ YÜKLEME ===
idf = pd.read_excel(excel_inputs, sheet_name="inputs_df")
if "value_at_start" not in idf.columns: idf["value_at_start"] = 0
start_val_map = dict(zip(idf["node_id"], idf["value_at_start"]))

if "state" in idf.columns:
    Na_set = set(idf[idf["state"] == 1]["node_id"].astype(int))
else:
    Na_set = {3, 6, 9, 12, 15}

# Koordinat Sistemi
x_vals = sorted(idf["x_coordinate"].unique())
y_vals = sorted(idf["y_coordinate"].unique())

dx = x_vals[1] - x_vals[0] if len(x_vals) > 1 else 5.0
dy = y_vals[1] - y_vals[0] if len(y_vals) > 1 else 5.0

extent = [
    min(x_vals) - dx / 2,
    max(x_vals) + dx / 2,
    min(y_vals) - dy / 2,
    max(y_vals) + dy / 2
]

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

# === ARAÇ HARİTALAMASINI HAZIRLA ===
node_incoming_bases = get_node_incoming_bases(res_path)


def fnum(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "-"
    return f"{v:.1f}"


file_ts = datetime.now().strftime('%Y%m%d_%H%M%S')


# === BASES ÇİZİM YARDIMCISI ===
def draw_bases(ax):
    for label, (bx, by) in BASES.items():
        b_color = BASE_COLORS.get(label, "#333333")
        # Base Station Kareleri (Beyaz çerçeveli)
        ax.scatter(bx, by, s=350, marker='s', color=b_color,
                   edgecolors='white', linewidth=1.5, zorder=30, clip_on=False)
        ax.text(bx, by, label, color='white', fontweight='bold',
                fontsize=10, ha='center', va='center', zorder=31, clip_on=False)


# ==========================================
# HARİTA 1: INITIAL SITUATION
# ==========================================
def plot_initial_map():
    vals = [v for v in start_val_map.values() if not np.isnan(v)]
    vmin, vmax = (min(vals), max(vals)) if vals else (0, 100)

    # Yeşil tonları (Greens)
    cmap = plt.cm.Greens
    norm = Normalize(vmin=vmin, vmax=vmax)

    rgb_grid = np.zeros((ny, nx, 3))
    for _, row in idf.iterrows():
        nid = int(row["node_id"])
        xi = coord_x_to_idx[row["x_coordinate"]]
        yi = coord_y_to_idx[row["y_coordinate"]]
        s_val = start_val_map.get(nid, 0.0)
        rgb_grid[yi, xi] = cmap(norm(s_val))[:3]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_grid, origin="lower", aspect='equal', interpolation='nearest', extent=extent)

    ax.set_xticks(np.arange(extent[0], extent[1] + 0.1, dx))
    ax.set_yticks(np.arange(extent[2], extent[3] + 0.1, dy))
    ax.grid(which='major', color='white', linestyle='-', linewidth=2)

    ax.set_xlabel("X Coordinate (km)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Y Coordinate (km)", fontsize=11, fontweight='bold')
    ax.set_title(f"Initial Situation (Value Gradient)\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 fontsize=16, fontweight='bold', pad=15)

    draw_bases(ax)

    for _, row in idf.iterrows():
        nid = int(row["node_id"])
        cx, cy = row["x_coordinate"], row["y_coordinate"]
        s_val = start_val_map.get(nid, 0.0)

        # Yazı rengi kontrastı
        text_color = "white" if s_val > (vmax + vmin) / 2 else "black"

        if nid in Na_set:
            rect = mpatches.Rectangle((cx - dx / 2 + 0.1, cy - dy / 2 + 0.1), dx - 0.2, dy - 0.2,
                                      fill=False, edgecolor='red', linewidth=3, zorder=20)
            ax.add_patch(rect)

        ax.text(cx, cy + dy * 0.15, str(nid), ha='center', va='center', fontsize=12, color=text_color,
                fontweight='bold', zorder=15)
        ax.text(cx, cy - dy * 0.15, f"Val: {fnum(s_val)}", ha='center', va='center', fontsize=9, color=text_color,
                zorder=15)

    patches = [
        mpatches.Patch(facecolor='none', edgecolor='red', linewidth=2, label='Initial Fire Source'),
        mpatches.Patch(color=BASE_COLORS["A"], label='Base A'),
        mpatches.Patch(color=BASE_COLORS["B"], label='Base B'),
        mpatches.Patch(color=BASE_COLORS["C"], label='Base C'),
        mpatches.Patch(color=BASE_COLORS["D"], label='Base D'),
        mpatches.Patch(color=BASE_COLORS["E"], label='Base E')
    ]
    ax.legend(handles=patches, loc='upper left', fontsize=10, frameon=True)

    plt.tight_layout()
    save_path = out_dir / f"initial_situation_{file_ts}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Initial Map kaydedildi: {save_path}")


# ==========================================
# HARİTA 2: DETAIL MAP
# ==========================================
def plot_detail_map():
    rgb_grid = np.zeros((ny, nx, 3))

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
    ax.imshow(rgb_grid, origin="lower", aspect='equal', interpolation='nearest', extent=extent)

    ax.set_xticks(np.arange(extent[0], extent[1] + 0.1, dx))
    ax.set_yticks(np.arange(extent[2], extent[3] + 0.1, dy))
    ax.grid(which='major', color='white', linestyle='-', linewidth=2)

    ax.set_xlabel("X Coordinate (km)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Y Coordinate (km)", fontsize=11, fontweight='bold')

    z_str = f"{z_val:,.2f}" if z_val is not None else "N/A"
    title_str = f"Optimization Result\nObjective Value (Z): {z_str}\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ax.set_title(title_str, fontsize=16, fontweight='bold', pad=15)

    draw_bases(ax)

    # --- DÜĞÜMLERİ ÇİZ ---
    for _, row in idf.iterrows():
        nid = int(row["node_id"])
        cx, cy = row["x_coordinate"], row["y_coordinate"]

        p_val = p_map.get(nid, 0.0)
        s_val = start_val_map.get(nid, 0.0)
        y_val = fire_status_map.get(nid, 0.0)

        if p_val <= 0.001:
            text_color = "white"
        elif p_val < (s_val - 0.001):
            text_color = "white"
        elif y_val >= 0.5:
            text_color = "black"
        else:
            text_color = "white"

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

        # --- YUVARLAK (DOT) ÇİZİMİ (ÇOK SATIRLI) ---
        bases_here = node_incoming_bases.get(nid, [])
        if bases_here:
            bases_here.sort()
            num_dots = len(bases_here)

            # Ayarlar
            dot_radius = dx * 0.035
            dot_spacing = dot_radius * 2.5
            max_dots_per_row = 8

            # Başlangıç Y konumu
            start_y = (cy - dy / 2) + dot_radius * 1.5

            for i, b_name in enumerate(bases_here):
                row_idx = i // max_dots_per_row
                col_idx = i % max_dots_per_row

                dots_in_this_row = min(max_dots_per_row, num_dots - row_idx * max_dots_per_row)
                row_width = (dots_in_this_row - 1) * dot_spacing
                row_start_x = cx - row_width / 2

                dot_x = row_start_x + col_idx * dot_spacing
                dot_y = start_y + row_idx * dot_spacing

                dot_color = BASE_COLORS.get(b_name, 'gray')
                circle = mpatches.Circle((dot_x, dot_y), radius=dot_radius,
                                         color=dot_color, ec='white', lw=0.5, zorder=25)
                ax.add_patch(circle)

    # Lejant
    patches = [
        mpatches.Patch(color=COLOR_BLACK, label='Never Controlled'),
        mpatches.Patch(color=COLOR_ORANGE, label='Controlled (Partial Loss)'),
        mpatches.Patch(color=COLOR_YELLOW, label='Controlled (No Loss)'),
        mpatches.Patch(color=COLOR_GREEN, label='No Fire'),
        mpatches.Patch(facecolor='none', edgecolor='red', linewidth=2, label='Initial Fire Source'),
        mpatches.Patch(color=BASE_COLORS["A"], label='Base A (Vehicle)'),
        mpatches.Patch(color=BASE_COLORS["B"], label='Base B (Vehicle)'),
        mpatches.Patch(color=BASE_COLORS["C"], label='Base C (Vehicle)'),
        mpatches.Patch(color=BASE_COLORS["D"], label='Base D (Vehicle)'),
        mpatches.Patch(color=BASE_COLORS["E"], label='Base E (Vehicle)')
    ]
    ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.08),
              fancybox=True, shadow=False, ncol=5, fontsize=10, frameon=False)

    plt.tight_layout()
    save_path = out_dir / f"detail_map_{file_ts}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Detail Map kaydedildi: {save_path}")


if __name__ == "__main__":
    plot_initial_map()
    plot_detail_map()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from datetime import datetime

# --- 1. AYARLAR VE FONKSİYON TANIMI ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "figure.dpi": 300
})


def draw_final_academic_plot(v_df, x_df, save_dir, ref_name):
    """Veriyi işler ve benzersiz isimle kaydeder."""
    # Sütun isimlerini normalize et
    v_df = v_df.rename(columns={"v": "v_ik", "value": "v_ik", "val": "v_ik"})
    x_df = x_df.rename(columns={"x": "x_ik", "value": "x_ik", "val": "x_ik"})

    # x_ik = 1 olanları filtrele
    assigned = x_df.loc[x_df["x_ik"] > 0.5, ["i", "k"]].drop_duplicates()
    merged = v_df.merge(assigned, on=["i", "k"], how="inner").sort_values(["i", "v_ik"])

    if merged.empty:
        print("Hata: Atanmış veri (x_ik=1) bulunamadı.")
        return

    nodes = sorted(merged["i"].unique())
    fig, ax = plt.subplots(figsize=(10, 5))

    # Stil: Renk ve Doku
    cmap = mpl.colormaps.get_cmap("viridis")
    patterns = ["", "///", "\\\\\\", "xxx", "---"]

    curr_x, gap, w = 0.0, 1.2, 0.2
    ticks_pos = []

    for idx, node in enumerate(nodes):
        vals = merged[merged["i"] == node]["v_ik"].values
        m = len(vals)
        start = curr_x - ((m - 1) * w / 2 if m > 1 else 0)

        for j, val in enumerate(vals):
            ax.bar(start + j * w, val, width=w * 0.85, color=cmap(idx / len(nodes)),
                   edgecolor="black", linewidth=0.6, hatch=patterns[j % len(patterns)])

        ticks_pos.append(curr_x)
        curr_x += gap

    # Eksen Ayarları
    ax.set_ylabel(r"Arrival Time ($v_{ik}$)")
    ax.set_xlabel(r"Node Index ($i$)")
    ax.set_xticks(ticks_pos)
    ax.set_xticklabels(nodes)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Benzersiz İsimle Kaydet
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"academic_plot_{Path(ref_name).stem}_{ts}.png"
    final_path = Path(save_dir) / unique_filename

    plt.tight_layout()
    plt.savefig(final_path, bbox_inches="tight")
    plt.show()
    print(f"Grafik başarıyla oluşturuldu: {unique_filename}")


# --- 2. VERİ OKUMA VE ÇALIŞTIRMA (Hata Burada Çözülüyor) ---
# Not: res_path ve out_dir'in daha önce kodunuzda tanımlanmış olması gerekir.
try:
    # ÖNCE VERİYİ OKUYORUZ (v_df ve x_df burada doğuyor)
    print(f"Veri okunuyor: {res_path.name}")
    v_df = pd.read_excel(res_path, sheet_name="v")
    x_df = pd.read_excel(res_path, sheet_name="x")

    # SONRA FONKSİYONU ÇAĞIRIYORUZ
    draw_final_academic_plot(v_df, x_df, out_dir, res_path.name)

except NameError as e:
    print(f"Hata: Değişken tanımlı değil. Lütfen 'res_path' ve 'out_dir' yollarını kontrol edin. -> {e}")
except Exception as e:
    print(f"Beklenmeyen bir hata oluştu: {e}")