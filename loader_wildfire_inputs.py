from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
import pandas as pd
import ast


@dataclass(frozen=True)
class Node:
    id: int
    x: float
    y: float
    v0: float
    degrade: float
    ameliorate: float
    state: int
    neighbors: Tuple[int, ...]
    beta: float = 0.0  # <-- YENİ: düğüm başına beta


def _parse_neighbors(obj) -> Tuple[int, ...]:
    """Parse neighbors cell which may be a string like '[1, 2]' or already a list.
    Returns a sorted, deduplicated tuple of ints (hashable & stable for dataclasses).
    """
    if isinstance(obj, (list, tuple)):
        vals = list(obj)
    elif pd.isna(obj):
        vals = []
    else:
        try:
            parsed = ast.literal_eval(str(obj))
            if isinstance(parsed, (list, tuple, set)):
                vals = list(parsed)
            else:
                vals = [int(parsed)]
        except Exception:
            vals = [int(x) for x in str(obj).replace('[', '').replace(']', '').split(',') if str(x).strip()]
    clean = sorted({int(v) for v in vals})
    return tuple(clean)


def load_inputs(
        excel_path: str | Path,
        sheet_nodes: str = "inputs_df",
        sheet_params: str = "parameters",
        require_beta: bool = False,  # <-- YENİ: beta zorunlu mu?
        beta_column_candidates: Tuple[str, ...] = ("beta", "beta_i", "beta_value"),  # <-- YENİ
) -> Dict[str, Any]:
    """Load wildfire input data for a 5x5=25-node instance (or any size) from Excel.

    Returns a dictionary with Gurobi-ready sets and parameter dicts.
    Keys:
        N: List[int]  # node ids
        coords: Dict[int, Tuple[float, float]]
        v0: Dict[int, float]  # value_at_start
        degrade: Dict[int, float]  # fire_degradation_rate
        ameliorate: Dict[int, float]  # fire_amelioration_rate
        state: Dict[int, int]
        neighbors: Dict[int, Tuple[int, ...]]
        A: Dict[Tuple[int,int], int]  # adjacency (i -> j) 1 if j in neighbors[i]
        P: Dict[str, float | int]  # global scalar parameters from 'parameters' sheet
        K: List[int]  # vehicle index set [1..P['n_vehicles']]
        beta: Dict[int, float]  # <-- YENİ: düğüm başına beta
    """
    p = Path(excel_path)
    if not p.exists():
        raise FileNotFoundError(f"Excel file not found: {p}")

    df = pd.read_excel(p, sheet_name=sheet_nodes)
    params_df = pd.read_excel(p, sheet_name=sheet_params)

    required_cols = [
        "node_id",
        "x_coordinate",
        "y_coordinate",
        "value_at_start",
        "fire_degradation_rate",
        "fire_amelioration_rate",
        "state",
        "neighborhood_list",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in sheet '{sheet_nodes}': {missing}")

    # --- P sözlüğünü oku (önce, çünkü beta_default buradan gelebilir)
    if "parameter" in params_df.columns and "value" in params_df.columns:
        P: Dict[str, Any] = {}
        for _, r in params_df.iterrows():
            key = str(r["parameter"])
            val = r["value"]
            try:
                fval = float(val)
                ival = int(fval)
                P[key] = ival if fval.is_integer() else fval
            except Exception:
                P[key] = val
    else:
        P = {str(r.iloc[0]): r.iloc[1] for _, r in params_df.iterrows()}

    # --- beta için kolon eşleme
    beta_col: Optional[str] = None
    for cand in beta_column_candidates:
        if cand in df.columns:
            beta_col = cand
            break

    # --- beta bulunamazsa strateji
    global_beta_default = float(P.get("beta_default", 0.0))
    if beta_col is None:
        if require_beta and "beta_default" not in P:
            raise KeyError(
                f"'beta' sütunu ({beta_column_candidates}) bulunamadı ve 'parameters' sayfasında "
                f"'beta_default' yok. Ya inputs_df'e bir beta kolonu ekleyin ya da parameters'a beta_default koyun."
            )
        # Aksi halde düğümler için varsayılanı kullanacağız (beta_default varsa onu, yoksa 0.0)

    # --- Düğümleri inşa et
    nodes: Dict[int, Node] = {}
    for _, row in df.iterrows():
        i = int(row["node_id"])
        nbh = _parse_neighbors(row["neighborhood_list"])
        # beta değeri
        if beta_col is not None:
            try:
                beta_i = float(row[beta_col])
            except Exception:
                beta_i = global_beta_default
        else:
            beta_i = global_beta_default

        nodes[i] = Node(
            id=i,
            x=float(row["x_coordinate"]),
            y=float(row["y_coordinate"]),
            v0=float(row["value_at_start"]),
            degrade=float(row["fire_degradation_rate"]),
            ameliorate=float(row["fire_amelioration_rate"]),
            state=int(row["state"]),
            neighbors=nbh,
            beta=beta_i,  # <-- YENİ
        )

    N = sorted(nodes.keys())
    coords: Dict[int, Tuple[float, float]] = {i: (nodes[i].x, nodes[i].y) for i in N}
    v0: Dict[int, float] = {i: nodes[i].v0 for i in N}
    degrade: Dict[int, float] = {i: nodes[i].degrade for i in N}
    ameliorate: Dict[int, float] = {i: nodes[i].ameliorate for i in N}
    state: Dict[int, int] = {i: nodes[i].state for i in N}
    neighbors: Dict[int, Tuple[int, ...]] = {i: nodes[i].neighbors for i in N}
    beta: Dict[int, float] = {i: nodes[i].beta for i in N}  # <-- YENİ

    # Adjacency
    A: Dict[tuple[int, int], int] = {}
    for i in N:
        for j in N:
            A[(i, j)] = 1 if j in neighbors[i] else 0

    # Araç kümesi
    n_vehicles = int(P.get("n_vehicles", 0)) if "n_vehicles" in P else 0
    K = list(range(1, n_vehicles + 1)) if n_vehicles > 0 else []
    # İsteğe bağlı: hız okunuyor ama burada kullanılmıyor; P içinde zaten mevcut
    # speed_km_per_min = P.get("vehicle_flight_speed", None)

    return {
        "N": N,
        "coords": coords,
        "v0": v0,
        "degrade": degrade,
        "ameliorate": ameliorate,
        "state": state,
        "neighbors": neighbors,
        "A": A,
        "P": P,
        "K": K,
        "beta": beta,  # <-- YENİ
    }


def demo_print_summary(data: Dict[str, Any]) -> str:
    """
    Return a short human-readable summary string (for quick sanity checks).
    """
    N = data["N"]
    P = data["P"]
    lines = []
    lines.append(f"Nodes: {len(N)} -> min id {min(N)}, max id {max(N)}")
    if "n_vehicles" in P:
        lines.append(f"Vehicles (n_vehicles): {P['n_vehicles']}")
    # Show first 3 nodes
    for i in N[:3]:
        x, y = data["coords"][i]
        lines.append(
            f"i={i}: (x,y)=({x},{y}) v0={data['v0'][i]} degrade={data['degrade'][i]} "
            f"ameliorate={data['ameliorate'][i]} state={data['state'][i]} "
            f"beta={data['beta'][i]} neigh={list(data['neighbors'][i])}"
        )
    return "\n".join(lines)


def compute_d_param(coords: dict[int, tuple[float, float]],
                    speed_km_per_min: float,
                    K: list[int]) -> dict[tuple[int, int], float]:
    """
    d_[i,k] = üsten (0,0) düğüm i'ye ulaşım süresi (dakika).
    Hız tüm araçlar için eşit varsayılmıştır.
    """
    import math
    d = {}
    for i, (x, y) in coords.items():
        tmin = math.hypot(x, y) / speed_km_per_min  # sqrt(x^2+y^2)/v
        for k in K:
            d[(i, k)] = tmin
    return d

# # ----------------------------------------------------------------------
# # EK: Yüklenen verileri Excel'e/CSV'ye dökme yardımcıları
# # ----------------------------------------------------------------------
# import pandas as pd
# from pathlib import Path
# from typing import Dict, Any, Tuple, List, Optional
#
# def export_loaded_data_to_excel(
#     data: Dict[str, Any],
#     out_path: str | Path,
#     include_d: bool = False,
#     d: Dict[Tuple[int, int], float] | None = None,
#     neighbors_as_string: bool = True,
# ) -> Path:
#     """
#     load_inputs(...) tarafından döndürülen sözlüğü, denetim amaçlı olarak Excel'e yazar.
#
#     Üreteceği sayfalar:
#       - 'parameters'     : P sözlüğü (parameter, value)
#       - 'nodes'          : düğüm düzeyi tablo (x, y, v0, degrade, ameliorate, state, neighbors, beta)
#       - 'adjacency_A'    : NxN komşuluk matrisi (A[(i,j)])
#       - 'vehicles_K'     : araç kümesi (K)
#       - 'summary'        : kısa metinsel özet (demo_print_summary kullanılarak)
#       - 'd_param'        : (opsiyonel) d[(i,k)] seyahat süreleri (dakika)
#
#     Parametreler
#     ------------
#     data : Dict[str, Any]
#         load_inputs(...) çıktısı.
#     out_path : str | Path
#         Yazılacak Excel dosyası yolu (örn. 'loaded_check.xlsx').
#     include_d : bool
#         True ise d parametresi yazılır. d=None ise ValueError fırlatır.
#     d : Dict[(int,int), float] | None
#         d[(i,k)] sözlüğü. include_d=True ise zorunludur.
#     neighbors_as_string : bool
#         True ise neighbors sütunu '[1, 2, 3]' biçimli string; False ise liste olarak yazılır.
#
#     Dönüş
#     -----
#     Path
#         Yazılan dosyanın yolu.
#     """
#     # --- Anahtar kontrolü
#     required_keys = ["N", "coords", "v0", "degrade", "ameliorate", "state", "neighbors", "A", "P", "K", "beta"]
#     missing = [k for k in required_keys if k not in data]
#     if missing:
#         raise KeyError(f"data sözlüğünde eksik anahtar(lar) var: {missing}")
#
#     if include_d and d is None:
#         raise ValueError("include_d=True iken 'd' sözlüğü verilmelidir.")
#
#     N: List[int] = list(data["N"])
#     P: Dict[str, Any] = dict(data["P"])
#     coords: Dict[int, Tuple[float, float]] = dict(data["coords"])
#     v0: Dict[int, float] = dict(data["v0"])
#     degrade: Dict[int, float] = dict(data["degrade"])
#     ameliorate: Dict[int, float] = dict(data["ameliorate"])
#     state: Dict[int, int] = dict(data["state"])
#     neighbors: Dict[int, Tuple[int, ...]] = dict(data["neighbors"])
#     A: Dict[Tuple[int, int], int] = dict(data["A"])
#     K: List[int] = list(data["K"])
#     beta: Dict[int, float] = dict(data["beta"])
#
#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#
#     # --- 1) parameters sayfası
#     params_df = pd.DataFrame({"parameter": list(P.keys()), "value": list(P.values())}) \
#                   .sort_values("parameter", ignore_index=True)
#
#     # --- 2) nodes sayfası
#     node_rows = []
#     for i in N:
#         neigh = list(neighbors[i])
#         node_rows.append({
#             "node_id": i,
#             "x_coordinate": coords[i][0],
#             "y_coordinate": coords[i][1],
#             "value_at_start": v0[i],
#             "fire_degradation_rate": degrade[i],
#             "fire_amelioration_rate": ameliorate[i],
#             "state": state[i],
#             "neighborhood_list": str(neigh) if neighbors_as_string else neigh,
#             "beta": beta[i],
#         })
#     nodes_df = pd.DataFrame(node_rows).sort_values("node_id", ignore_index=True)
#
#     # --- 3) adjacency_A sayfası (NxN matris)
#     adj_df = pd.DataFrame(index=N, columns=N, dtype=int)
#     for i in N:
#         for j in N:
#             adj_df.at[i, j] = int(A.get((i, j), 0))
#     adj_df.index.name = "i\\j"
#
#     # --- 4) vehicles_K sayfası
#     vehicles_df = pd.DataFrame({"vehicle_id": K})
#
#     # --- 5) summary sayfası
#     try:
#         summary_text = demo_print_summary(data)  # modülde tanımlı
#     except Exception:
#         summary_text = f"N={len(N)} düğüm; K={len(K)} araç; P_keys={len(P)}."
#     summary_df = pd.DataFrame({"summary": [summary_text]})
#
#     # --- 6) (opsiyonel) d_param sayfası
#     if include_d:
#         di, dk, dv = [], [], []
#         for (i, k), val in d.items():
#             di.append(i); dk.append(k); dv.append(float(val))
#         d_df = pd.DataFrame({"i": di, "k": dk, "d_value_min": dv}).sort_values(["i", "k"], ignore_index=True)
#
#     # --- Yazım
#     with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
#         params_df.to_excel(writer, sheet_name="parameters", index=False)
#         nodes_df.to_excel(writer, sheet_name="nodes", index=False)
#         adj_df.to_excel(writer, sheet_name="adjacency_A")
#         vehicles_df.to_excel(writer, sheet_name="vehicles_K", index=False)
#         summary_df.to_excel(writer, sheet_name="summary", index=False)
#         if include_d:
#             d_df.to_excel(writer, sheet_name="d_param", index=False)
#
#         # Basit kolon genişletme
#         for sheet in writer.sheets:
#             ws = writer.sheets[sheet]
#             for col in range(0, 12):
#                 ws.set_column(col, col, 18)
#
#     return out_path
#
#
# def export_loaded_data_to_csvs(
#     data: Dict[str, Any],
#     out_dir: str | Path,
#     include_d: bool = False,
#     d: Dict[Tuple[int, int], float] | None = None,
# ) -> Dict[str, Path]:
#     """
#     Excel yerine/yanı sıra düz CSV çıktı istersek kullanışlı minik yardımcı.
#     Dönüş: yazılan dosyaların yollarını içeren sözlük.
#     """
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     N: List[int] = list(data["N"])
#     P: Dict[str, Any] = dict(data["P"])
#     coords: Dict[int, Tuple[float, float]] = dict(data["coords"])
#     v0: Dict[int, float] = dict(data["v0"])
#     degrade: Dict[int, float] = dict(data["degrade"])
#     ameliorate: Dict[int, float] = dict(data["ameliorate"])
#     state: Dict[int, int] = dict(data["state"])
#     neighbors: Dict[int, Tuple[int, ...]] = dict(data["neighbors"])
#     A: Dict[Tuple[int, int], int] = dict(data["A"])
#     K: List[int] = list(data["K"])
#     beta: Dict[int, float] = dict(data["beta"])
#
#     # parameters.csv
#     params_df = pd.DataFrame({"parameter": list(P.keys()), "value": list(P.values())}) \
#                   .sort_values("parameter", ignore_index=True)
#     p_params = out_dir / "parameters.csv"
#     params_df.to_csv(p_params, index=False)
#
#     # nodes.csv
#     rows = []
#     for i in N:
#         rows.append({
#             "node_id": i,
#             "x_coordinate": coords[i][0],
#             "y_coordinate": coords[i][1],
#             "value_at_start": v0[i],
#             "fire_degradation_rate": degrade[i],
#             "fire_amelioration_rate": ameliorate[i],
#             "state": state[i],
#             "neighborhood_list": str(list(neighbors[i])),
#             "beta": beta[i],
#         })
#     p_nodes = out_dir / "nodes.csv"
#     pd.DataFrame(rows).sort_values("node_id", ignore_index=True).to_csv(p_nodes, index=False)
#
#     # adjacency_A.csv (uzun form: i,j,Aij)
#     a_rows = []
#     for (i, j), val in A.items():
#         a_rows.append({"i": i, "j": j, "Aij": int(val)})
#     p_adj = out_dir / "adjacency_A.csv"
#     pd.DataFrame(a_rows).sort_values(["i", "j"], ignore_index=True).to_csv(p_adj, index=False)
#
#     # vehicles_K.csv
#     p_k = out_dir / "vehicles_K.csv"
#     pd.DataFrame({"vehicle_id": K}).to_csv(p_k, index=False)
#
#     # (opsiyonel) d_param.csv
#     p_d: Optional[Path] = None
#     if include_d:
#         if d is None:
#             raise ValueError("include_d=True iken 'd' sözlüğü verilmelidir.")
#         di, dk, dv = zip(*[(i, k, float(val)) for (i, k), val in d.items()]) if d else ([], [], [])
#         p_d = out_dir / "d_param.csv"
#         pd.DataFrame({"i": di, "k": dk, "d_value_min": dv}).to_csv(p_d, index=False)
#
#     return {
#         "parameters": p_params,
#         "nodes": p_nodes,
#         "adjacency_A": p_adj,
#         "vehicles_K": p_k,
#         "d_param": p_d if include_d else None
#     }
#
#
# # ----------------------------------------------------------------------
# # ÖRNEK ÇALIŞTIRMA BLOĞU (isteğe bağlı)
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     # 1) Excel'den veriyi yükleyin
#     excel_path = r"C:\Users\Mert\deprem\wildfire\inputs_to_load-5x5_burcu.xlsx"
#     data = load_inputs(excel_path)
#
#     # 2) (opsiyonel) d parametresini hesaplayın
#     #    Hız (km/dk): örn. 230 km/saat = 230/60 ≈ 3.8333 km/dk
#     speed_km_per_min = 3.83
#     d = compute_d_param(data["coords"], speed_km_per_min=speed_km_per_min, K=data["K"])
#
#     # 3) Excel çıktı (d dahil)
#     out_xlsx = Path(r"C:\Users\Mert\deprem\wildfire\loaded_check.xlsx")
#     export_loaded_data_to_excel(data, out_xlsx, include_d=True, d=d)
#
#     # 4) CSV çıktılar (opsiyonel)
#     out_dir = Path(r"C:\Users\Mert\deprem\wildfire\loaded_check_csvs")
#     export_loaded_data_to_csvs(data, out_dir, include_d=True, d=d)
#
#     print(f"Excel yazıldı: {out_xlsx}")
#     print(f"CSV klasörü:  {out_dir}")
