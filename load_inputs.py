import pandas as pd
import ast
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


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
    beta: float = 0.0
    su: float = 0.0


def _parse_neighbors(obj) -> Tuple[int, ...]:
    if isinstance(obj, (list, tuple)):
        vals = list(obj)
    elif pd.isna(obj):
        vals = []
    else:
        try:
            parsed = ast.literal_eval(str(obj))
            vals = list(parsed) if isinstance(parsed, (list, tuple, set)) else [int(parsed)]
        except:
            vals = [int(x) for x in str(obj).replace('[', '').replace(']', '').split(',') if str(x).strip()]
    return tuple(sorted({int(v) for v in vals}))


def load_wildfire_data_final(excel_path: str) -> Dict[str, Any]:
    # 1. Excel Sayfalarını Yükle
    xls = pd.ExcelFile(excel_path)
    df_nodes = pd.read_excel(xls, sheet_name='inputs_df')
    df_bases = pd.read_excel(xls, sheet_name='bases')
    df_params = pd.read_excel(xls, sheet_name='parameters')

    # 2. Parametreleri Temizleyerek Oku
    # .strip() ile hücre içindeki gizli boşluklar temizlenir
    P = {str(row['parameter']).strip(): row['value'] for _, row in df_params.iterrows()}

    # 3. Araç Tipi Eşleştirme (Bases -> Parameters)
    vehicle_types_map = {
        'Helicopter': 'helicopter',
        'Fire Engine': 'fire_engine',
        'FRV': 'FRV'
    }

    vehicle_records = []
    current_k = 1

    for _, row in df_bases.iterrows():
        b_id = str(row['Base'])
        bx = float(row['x_coordinate'])
        by = float(row['y_coordinate'])

        for col_name, p_prefix in vehicle_types_map.items():
            # Eğer o basede araç varsa (>0)
            if col_name in df_bases.columns and pd.notna(row[col_name]) and row[col_name] > 0:
                count = int(row[col_name])

                # Hız ve Kapasite anahtarlarını P içinden bul
                # 'else' yazılmamıştır, veri eksikse kod burada duracaktır.
                v_speed = float(P[f"{p_prefix}_speed"])
                v_capacity = float(P[f"{p_prefix}_capacity"])

                for _ in range(count):
                    vehicle_records.append({
                        'id': current_k,
                        'base_id': b_id,
                        'type': col_name,
                        'speed': v_speed,
                        'capacity': v_capacity,
                        'bx': bx, 'by': by
                    })
                    current_k += 1

    if not vehicle_records:
        raise ValueError("HATA: Hiçbir araç oluşturulamadı. Lütfen 'bases' ve 'parameters' sayfalarını kontrol edin.")

    # 4. Küme ve Parametreler
    K = [v['id'] for v in vehicle_records]
    mu = {v['id']: v['capacity'] for v in vehicle_records}
    N = sorted(df_nodes['node_id'].unique().tolist())
    coords = {row['node_id']: (row['x_coordinate'], row['y_coordinate']) for _, row in df_nodes.iterrows()}
    state = {row['node_id']: row['state'] for _, row in df_nodes.iterrows()}
    v0 = {row['node_id']: row['value_at_start'] for _, row in df_nodes.iterrows()}
    degrade = {row['node_id']: row['fire_degradation_rate'] for _, row in df_nodes.iterrows()}
    ameliorate = dict(zip(df_nodes['node_id'], df_nodes['fire_amelioration_rate']))
    neighbors = {row['node_id']: _parse_neighbors(row['neighborhood_list']) for _, row in df_nodes.iterrows()}
    su = dict(zip(df_nodes['node_id'], df_nodes.get('su', 0.0)))
    beta = dict(zip(df_nodes['node_id'], df_nodes.get('beta', 0.0)))
    # 5. Seyahat Süresi Matrisi d(i, k) Hesaplama
    # Formül: $d_{ik} = \frac{distance}{velocity}$
    d_matrix = {}
    for v in vehicle_records:
        k = v['id']
        vk = v['speed']
        bx, by = v['bx'], v['by']

        for i in N:
            ix, iy = coords[i]

            # Mesafe Metriği: Helicopter -> Euclidean, Diğerleri -> Manhattan
            if v['type'] == 'Helicopter':
                dist = math.hypot(ix - bx, iy - by)
            else:
                dist = abs(ix - bx) + abs(iy - by)

            d_matrix[(i, k)] = dist / vk

    return {
        "N": N,
        "K": [v['id'] for v in vehicle_records],
        "d_matrix": d_matrix,
        "mu": mu,  # KeyError: 'mu' çözümü
        "v0": v0,
        "state": state,  # KeyError: 'state' çözümü
        "degrade": degrade,
        "ameliorate": ameliorate,  # KeyError: 'ameliorate' çözümü
        "su": su,
        "beta": beta,
        "P": P,  # Global parametreler
        "nodes_df": df_nodes,
        "vehicle_info": vehicle_records,
        "neighbors": neighbors
    }


# --- TEST VE KONTROL ---
if __name__ == "__main__":
    FILE_PATH = r"C:\Users\Mert\deprem\wildfire\inputs_to_load-5x5_yyo.xlsx"
    try:
        data = load_wildfire_data_final(FILE_PATH)

        # d_matrix boş değilse DataFrame'e çevir
        if data['d_matrix']:
            d_series = pd.Series(data['d_matrix'])
            d_series.index.names = ['node_i', 'vehicle_k']
            d_df = d_series.unstack()

            print(f"--- VERİ YÜKLEME BAŞARILI ---")
            print(f"Araç Sayısı: {len(data['K'])} | Matris Boyutu: {d_df.shape}")
            print("\nSeyahat Süresi Matrisi (İlk 5 Satır):")
            print(d_df.head())
        else:
            print("Uyarı: Matris hesaplanamadı.")

    except Exception as e:
        print(f"Hata detayı: {e}")