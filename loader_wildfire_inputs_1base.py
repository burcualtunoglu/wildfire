from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional, Union
import pandas as pd
import ast
import math


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
    beta: float = 0.0  # düğüm başına beta


def _parse_neighbors(obj) -> Tuple[int, ...]:
    """
    'neighbors' hücresini parse eder. Örn: "[1, 2]" ya da hali hazırda list/tuple.
    Dönüş: sıralı ve tekrarsız int tuple (hashable & stabil).
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
    excel_path: Union[str, Path],
    sheet_nodes: str = "inputs_df",
    sheet_params: str = "parameters",
    require_beta: bool = False,
    beta_column_candidates: Tuple[str, ...] = ("beta", "beta_i", "beta_value"),
    # su/omega kolon adları için adaylar:
    su_column_candidates: Tuple[str, ...] = ("su", "omega", "water_required", "water", "omega_i"),
    require_su: bool = False,
) -> Dict[str, Any]:
    """
    Excel'den düğüm ve parametreleri yükler.

    Dönen sözlük:
        N: List[int]
        coords: Dict[int, Tuple[float, float]]
        v0, degrade, ameliorate: Dict[int, float]
        state: Dict[int, int]
        neighbors: Dict[int, Tuple[int, ...]]
        A: Dict[Tuple[int,int], int]  # adjacency (i->j)
        P: Dict[str, float|int|str]
        K: List[int]
        beta: Dict[int, float]
        su: Dict[int, float]          # (varsa) su/omega kolonundan
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

    # --- Parametre sözlüğü P ---
    if "parameter" in params_df.columns and "value" in params_df.columns:
        P: Dict[str, Any] = {}
        for _, r in params_df.iterrows():
            key = str(r["parameter"])
            val = r["value"]
            try:
                fval = float(val)
                ival = int(fval)
                P[key] = ival if float(ival) == fval else fval
            except Exception:
                P[key] = val
    else:
        # Serbest iki kolonlu biçimleri desteklemek için basit fallback
        P = {str(r.iloc[0]): r.iloc[1] for _, r in params_df.iterrows()}

    # --- beta ve su/omega kolon eşlemeleri ---
    beta_col: Optional[str] = None
    for cand in beta_column_candidates:
        if cand in df.columns:
            beta_col = cand
            break

    su_col: Optional[str] = None
    for cand in su_column_candidates:
        if cand in df.columns:
            su_col = cand
            break

    global_beta_default = float(P.get("beta_default", 0.0))
    if require_beta and (beta_col is None) and ("beta_default" not in P):
        raise KeyError(
            f"'beta' sütunu ({beta_column_candidates}) yok ve 'parameters' sayfasında 'beta_default' yok. "
            f"Ya inputs_df'e bir beta kolonu ekleyin ya da parameters'a beta_default koyun."
        )
    if require_su and (su_col is None):
        raise KeyError(
            f"'su' (omega) sütunu ({su_column_candidates}) bulunamadı. "
            f"inputs_df'e bir su/omega kolonu ekleyin ya da require_su=False kullanın."
        )

    # --- Düğümleri inşa et ---
    nodes: Dict[int, Node] = {}
    su_dict: Dict[int, float] = {}
    for _, row in df.iterrows():
        i = int(row["node_id"])
        nbh = _parse_neighbors(row["neighborhood_list"])

        # beta_i: varsa hücreden, yoksa global_default
        if beta_col is not None:
            try:
                beta_i = float(row[beta_col])
            except Exception:
                beta_i = global_beta_default
        else:
            beta_i = global_beta_default

        # su/omega: varsa oku
        if su_col is not None:
            try:
                su_i = float(row[su_col])
                su_dict[i] = su_i
            except Exception:
                # hücre boş/uygunsuzsa atla
                pass

        nodes[i] = Node(
            id=i,
            x=float(row["x_coordinate"]),
            y=float(row["y_coordinate"]),
            v0=float(row["value_at_start"]),
            degrade=float(row["fire_degradation_rate"]),
            ameliorate=float(row["fire_amelioration_rate"]),
            state=int(row["state"]),
            neighbors=nbh,
            beta=beta_i,
        )

    N = sorted(nodes.keys())
    coords: Dict[int, Tuple[float, float]] = {i: (nodes[i].x, nodes[i].y) for i in N}
    v0: Dict[int, float] = {i: nodes[i].v0 for i in N}
    degrade: Dict[int, float] = {i: nodes[i].degrade for i in N}
    ameliorate: Dict[int, float] = {i: nodes[i].ameliorate for i in N}
    state: Dict[int, int] = {i: nodes[i].state for i in N}
    neighbors: Dict[int, Tuple[int, ...]] = {i: nodes[i].neighbors for i in N}
    beta: Dict[int, float] = {i: nodes[i].beta for i in N}

    # Adjacency matrisi (i->j)
    A: Dict[Tuple[int, int], int] = {}
    for i in N:
        for j in N:
            A[(i, j)] = 1 if j in neighbors[i] else 0

    # Araç kümesi
    n_vehicles = int(P.get("n_vehicles", 0)) if "n_vehicles" in P else 0
    K = list(range(1, n_vehicles + 1)) if n_vehicles > 0 else []

    out: Dict[str, Any] = {
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
        "beta": beta,
    }
    if su_dict:
        out["su"] = su_dict  # varsa ekle

    return out



# # Öklid uzaklığına bağlı seyahat süresi
# def compute_d_param(
#     coords: Dict[int, Tuple[float, float]],
#     speed_km_per_hour: float,
#     K: List[int]
# ) -> Dict[Tuple[int, int], float]:
#     """
#     d[i,k] = (0,0) referansından düğüm i'ye tek yön uçuş süresi (dakika).
#     Hız tüm araçlar için eşit varsayılmıştır.
#     """
#     d: Dict[Tuple[int, int], float] = {}
#     for i, (x, y) in coords.items():
#         thour = math.hypot(x, y) / speed_km_per_hour  # sqrt(x^2+y^2)/v
#         for k in K:
#             d[(i, k)] = thour
#     return d

def compute_d_param(
        coords: Dict[int, Tuple[float, float]],
        speed_km_per_hour: float,
        K: List[int]
) -> Dict[Tuple[int, int], float]:
    """
    d[i,k] = (0,0) referansından düğüm i'ye Manhattan metriği ile hesaplanan
    tek yön seyahat süresi (SAAT).
    Matematiksel Model:
    Mesafe (Manhattan) = |x| + |y|
    Süre (Saat) = Mesafe / Hız
    """
    d: Dict[Tuple[int, int], float] = {}

    for i, (x, y) in coords.items():
        # Manhattan Uzaklığı (L1 Norm): |x| + |y|
        # Referans noktası (0,0) olduğu için (x-0) ve (y-0) işlemine gerek yoktur.
        manhattan_dist_km = abs(x) + abs(y)

        # Sürenin hesaplanması (Saat cinsinden)
        # t = x / v
        t_hour = manhattan_dist_km / speed_km_per_hour

        for k in K:
            d[(i, k)] = t_hour

    return d