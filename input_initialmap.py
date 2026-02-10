# -*- coding: utf-8 -*-
"""
input.py

Bu script:
1. scenario.xlsx dosyasından parametreleri okur.
2. 50x50, 100x100 gibi gridler için mekansal verileri (inputs_df) üretir.
3. Üsleri (bases) ve Parametreleri (parameters) oluşturur.
4. Tüm veriyi 'inputs' klasörü altına Excel olarak kaydeder.
5. Oluşturulan Excel'i tekrar okuyarak 'inputs/initial_maps' altına haritaları üretir.
"""

from __future__ import annotations

import os
import math
import re
import datetime
from typing import List, Dict, Any, Set, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects


# ============================================================
# 1) YARDIMCI FONKSİYONLAR (Grid, Geometri, Parse)
# ============================================================

def validate_square_grid(n_nodes: int) -> int:
    g = int(round(math.sqrt(n_nodes)))
    if g * g != n_nodes:
        raise ValueError(
            f"n_nodes kare sayı olmalı (örn. 2500, 3600...). Verilen: {n_nodes}"
        )
    return g


def generate_grid_centers(grid_size: int, edge: float):
    xs, ys = [], []
    for r in range(grid_size):
        for c in range(grid_size):
            xs.append((c + 0.5) * edge)
            ys.append((r + 0.5) * edge)
    return xs, ys


def parse_node_list(x) -> List[int]:
    if pd.isna(x) or str(x).strip() == "":
        return []
    if isinstance(x, (int, float)):
        return [int(x)]
    s = str(x).strip()
    tokens = re.split(r"[,\.;\s]+", s)
    out = []
    for t in tokens:
        if t.isdigit():
            out.append(int(t))
    return out


def _node_to_rc_0based(node_id: int, grid_size: int) -> Tuple[int, int]:
    return (node_id - 1) // grid_size, (node_id - 1) % grid_size


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ============================================================
# 2) KOMŞULUK VE ALAN ÜRETİMİ
# ============================================================

def generate_neighbors_4_filtered(grid_size: int, blocked_set: Set[int]) -> List[List[int]]:
    n_nodes = grid_size * grid_size
    neigh = [[] for _ in range(n_nodes)]
    for r in range(grid_size):
        for c in range(grid_size):
            nid = r * grid_size + c + 1
            if nid in blocked_set:
                continue
            cand = []
            if r > 0: cand.append((r - 1) * grid_size + c + 1)
            if r < grid_size - 1: cand.append((r + 1) * grid_size + c + 1)
            if c > 0: cand.append(r * grid_size + (c - 1) + 1)
            if c < grid_size - 1: cand.append(r * grid_size + (c + 1) + 1)
            neigh[nid - 1] = [j for j in cand if j not in blocked_set]
    return neigh


def remove_road_edges_from_neighbors(neighbors: List[List[int]], road_pairs: List[Tuple[int, int]], n_nodes: int):
    if not road_pairs: return neighbors
    for (i, j) in road_pairs:
        if 1 <= i <= n_nodes and 1 <= j <= n_nodes:
            if j in neighbors[i - 1]: neighbors[i - 1].remove(j)
            if i in neighbors[j - 1]: neighbors[j - 1].remove(i)
    return neighbors


def _smooth2d(field: np.ndarray, n_iter: int = 8, alpha: float = 0.60) -> np.ndarray:
    f = field.astype(float).copy()
    for _ in range(n_iter):
        up = np.pad(f[:-1, :], ((1, 0), (0, 0)), mode="edge")
        down = np.pad(f[1:, :], ((0, 1), (0, 0)), mode="edge")
        left = np.pad(f[:, :-1], ((0, 0), (1, 0)), mode="edge")
        right = np.pad(f[:, 1:], ((0, 0), (0, 1)), mode="edge")
        f = (1 - alpha) * f + alpha * 0.25 * (up + down + left + right)
    return f


def _minmax_scale(field: np.ndarray, lo: float, hi: float) -> np.ndarray:
    a, b = field.min(), field.max()
    if abs(b - a) < 1e-12: return np.full_like(field, (lo + hi) / 2.0)
    return lo + (field - a) * (hi - lo) / (b - a)


def generate_spatial_fields(grid_size: int, vmin: float, vmax: float, dmin: float, dmax: float,
                            rng: np.random.Generator):
    raw_v = rng.normal(0, 1, (grid_size, grid_size))
    raw_d = rng.normal(0, 1, (grid_size, grid_size))

    v_field = _minmax_scale(_smooth2d(raw_v), vmin, vmax)
    d_field = _minmax_scale(_smooth2d(raw_d), dmin, dmax)

    return np.clip(v_field, vmin, vmax), np.clip(d_field, dmin, dmax)


# ============================================================
# 3) DATA GENERATION
# ============================================================

def generate_scattered_nodes(grid_size: int, k: int, rng: np.random.Generator, forbidden_nodes: Set[int],
                             min_dist_cells: int):
    candidates = [i for i in range(1, grid_size * grid_size + 1) if i not in forbidden_nodes]
    if not candidates or k <= 0: return []

    chosen = []
    chosen_rc = []
    for _ in range(100000):
        if len(chosen) >= k: break
        node = int(rng.choice(candidates))
        rc = _node_to_rc_0based(node, grid_size)
        if all(_manhattan(rc, x) >= min_dist_cells for x in chosen_rc):
            chosen.append(node)
            chosen_rc.append(rc)
    return sorted(chosen)


def generate_random_road_pairs(grid_size: int, blocked_set: Set[int], rng: np.random.Generator, k: int):
    candidates = []
    for r in range(grid_size):
        for c in range(grid_size):
            i = r * grid_size + c + 1
            if i in blocked_set: continue
            if c < grid_size - 1:
                j = r * grid_size + c + 2
                if j not in blocked_set: candidates.append((i, j))
            if r < grid_size - 1:
                j = (r + 1) * grid_size + c + 1
                if j not in blocked_set: candidates.append((i, j))
    if not candidates: return []
    indices = rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
    return [candidates[i] for i in indices]


def read_scenario_xlsx(path: str, sheet_name: str) -> Dict[str, Any]:
    df = pd.read_excel(path, sheet_name=sheet_name)
    raw = dict(zip(df["param"].astype(str), df["value"]))

    get_val = lambda k, d=0: raw.get(k, d) if not pd.isna(raw.get(k, d)) else d

    return {
        "n_nodes": int(float(get_val("n_nodes", 2500))),
        "edge": float(get_val("edge", 500)),
        "vmin": float(get_val("vmin", 0)),
        "vmax": float(get_val("vmax", 100)),
        "dmin": float(get_val("dmin", 0)),
        "dmax": float(get_val("dmax", 1)),
        "amel": float(get_val("amel", 0.7)),
        "seed": int(float(get_val("seed", 42))),
        "out": str(get_val("out", "inputs.xlsx")),

        "fire_nodes": parse_node_list(get_val("fire_nodes", "")),
        "water_nodes": parse_node_list(get_val("water_nodes", "")),
        "protected_nodes": parse_node_list(get_val("protected_nodes", "")),

        "road_pairs_k": int(float(get_val("road_pairs_k", 0))),
        "n_bases": int(float(get_val("n_bases", 5))),

        "fire_auto": int(float(get_val("fire_auto", 0))),
        "n_fire": int(float(get_val("n_fire", 0))),
        "fire_min_dist_cells": int(float(get_val("fire_min_dist_cells", 5))),

        "protected_auto": int(float(get_val("protected_auto", 0))),
        "n_protected": int(float(get_val("n_protected", 0))),
        "protected_min_dist_cells": int(float(get_val("protected_min_dist_cells", 5))),

        "water_auto": int(float(get_val("water_auto", 0))),
        "water_every_km": float(get_val("water_every_km", 5.0)),
    }


def build_inputs_df(cfg: Dict[str, Any]) -> pd.DataFrame:
    grid_size = validate_square_grid(cfg["n_nodes"])
    rng = np.random.default_rng(cfg["seed"])

    x, y = generate_grid_centers(grid_size, cfg["edge"])

    water_set = set(cfg["water_nodes"])
    if cfg["water_auto"]:
        step = max(1, int((cfg["water_every_km"] * 1000) / cfg["edge"]))
        auto_water = []
        for r in range(0, grid_size, step):
            for c in range(0, grid_size, step):
                auto_water.append(r * grid_size + c + 1)
        water_set = set(auto_water)

    fire_set = set(cfg["fire_nodes"])
    prot_set = set(cfg["protected_nodes"])

    if cfg["fire_auto"]:
        fire_set = set(
            generate_scattered_nodes(grid_size, cfg["n_fire"], rng, water_set | prot_set, cfg["fire_min_dist_cells"]))
    if cfg["protected_auto"]:
        prot_set = set(generate_scattered_nodes(grid_size, cfg["n_protected"], rng, water_set | fire_set,
                                                cfg["protected_min_dist_cells"]))

    water_set -= fire_set
    prot_set -= (fire_set | water_set)

    state = np.zeros(cfg["n_nodes"], dtype=int)
    for i in fire_set: state[i - 1] = 1
    for i in water_set: state[i - 1] = 2
    for i in prot_set: state[i - 1] = 3

    blocked = water_set | prot_set
    neighbors = generate_neighbors_4_filtered(grid_size, blocked)

    road_pairs = cfg.get("road_pairs", [])
    if not road_pairs and cfg["road_pairs_k"] > 0:
        road_pairs = generate_random_road_pairs(grid_size, blocked, rng, cfg["road_pairs_k"])
        cfg["road_pairs"] = road_pairs

    neighbors = remove_road_edges_from_neighbors(neighbors, road_pairs, cfg["n_nodes"])

    v_field, d_field = generate_spatial_fields(grid_size, cfg["vmin"], cfg["vmax"], cfg["dmin"], cfg["dmax"], rng)
    v_flat = v_field.flatten()
    d_flat = d_field.flatten()

    for i in blocked:
        v_flat[i - 1] = 0
        d_flat[i - 1] = 0

    node_area = cfg["edge"] ** 2
    beta = (v_flat / node_area) * d_flat
    su = np.zeros_like(d_flat)
    mask = d_flat > 0
    water_per_cell = 0.036322 * node_area
    su[mask] = water_per_cell * (2 * d_flat[mask] / cfg["edge"])

    return pd.DataFrame({
        "node_id": range(1, cfg["n_nodes"] + 1),
        "x_coordinate": x, "y_coordinate": y,
        "value_at_start": np.round(v_flat, 6),
        "fire_degradation_rate": np.round(d_flat, 6),
        "fire_amelioration_rate": np.round(d_flat * cfg["amel"], 6),
        "beta": np.round(beta, 10),
        "su": np.round(su, 6),
        "state": state,
        "neighborhood_list": neighbors
    })


def build_bases_df_random(cfg: Dict[str, Any], grid_size: int, rng: np.random.Generator,
                          blocked_nodes: Set[int]) -> pd.DataFrame:
    n_bases = cfg["n_bases"]
    labels = [chr(ord('A') + i) for i in range(n_bases)]
    coords = generate_scattered_nodes(grid_size, n_bases, rng, blocked_nodes, 2)
    rows = []
    for lbl, nid in zip(labels, coords):
        r, c = _node_to_rc_0based(nid, grid_size)
        rows.append({
            "Base": lbl,
            "x_coordinate": c,
            "y_coordinate": r,
            "Helicopter": 2 if lbl == 'A' else 0,
            "Fire Engine": 3,
            "FRV": 2
        })
    return pd.DataFrame(rows)


def build_parameters_df_fixed(cfg: Dict[str, Any]) -> pd.DataFrame:
    n_nodes = int(cfg["n_nodes"])
    edge = float(cfg["edge"])
    grid_size = validate_square_grid(n_nodes)

    cell_side_length = edge
    region_side_length = grid_size * edge
    node_area = edge * edge

    # 36,322 L / km^2  -> 0.036322 L / m^2
    water_required_per_km2_l = 36322.0
    water_required_per_m2_l = water_required_per_km2_l / 1_000_000.0
    water_required_per_cell_l = water_required_per_m2_l * node_area

    road_str = ";".join([f"{u}-{v}" for (u, v) in cfg.get("road_pairs", [])])

    rows = [
        ("n_nodes", n_nodes, "required"),
        ("cell_side_length", cell_side_length, "cell edge length (m)"),
        ("region_side_length", region_side_length, "grid_size * cell_side_length"),
        ("node_area", node_area, "cell_side_length^2"),

        ("time_limit", 24, "default (hours)"),

        ("water required (for 1km2-lt)", water_required_per_km2_l, "given"),
        ("water_required_per_m2", water_required_per_m2_l, "derived"),
        ("water_required_per_cell", water_required_per_cell_l, "derived"),

        ("road_pairs", road_str, "edges removed from spread graph"),

        ("helicopter_capacity", 166.67, ""),
        ("helicopter_speed", 2000, ""),
        ("fire_engine_capacity", 117.65, ""),
        ("fire_engine_speed", 500, ""),
        ("FRV_capacity", 15, ""),
        ("FRV_speed", 1000, ""),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "note"])


# ============================================================
# 4) HARİTA GÖRSELLEŞTİRME
# ============================================================

def _parse_road_pairs_from_df(params_df: pd.DataFrame, inputs_df: pd.DataFrame, cell_size: float) -> List[List[tuple]]:
    road_segments = []
    half_cell = cell_size / 2
    try:
        val = params_df.loc[params_df['parameter'] == 'road_pairs', 'value']
        if val.empty or pd.isna(val.values[0]): return []

        node_map = inputs_df.set_index('node_id')[['x_coordinate', 'y_coordinate']].to_dict('index')
        pairs = str(val.values[0]).split(';')
        for pair in pairs:
            if '-' not in pair: continue
            try:
                n1, n2 = map(int, pair.split('-'))
                if n1 in node_map and n2 in node_map:
                    x1, y1 = node_map[n1]['x_coordinate'], node_map[n1]['y_coordinate']
                    x2, y2 = node_map[n2]['x_coordinate'], node_map[n2]['y_coordinate']
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    if abs(x1 - x2) > abs(y1 - y2):
                        p_start, p_end = (mid_x, mid_y - half_cell), (mid_x, mid_y + half_cell)
                    else:
                        p_start, p_end = (mid_x - half_cell, mid_y), (mid_x + half_cell, mid_y)
                    road_segments.append([p_start, p_end])
            except:
                continue
    except Exception as e:
        print(f"Yol parse hatası: {e}")
    return road_segments


def _plot_single_map(df: pd.DataFrame, bases_df: pd.DataFrame, road_segments: list,
                     data_col: str, cmap: str, title: str, cbar_label: str, output_path: str):
    unique_x = sorted(df['x_coordinate'].unique())
    cell_size = unique_x[1] - unique_x[0] if len(unique_x) > 1 else 500
    half_cell = cell_size / 2
    max_x = df['x_coordinate'].max() + half_cell
    max_y = df['y_coordinate'].max() + half_cell

    bases_df = bases_df.copy()
    bases_df['phys_x'] = bases_df['x_coordinate'] * cell_size + half_cell
    bases_df['phys_y'] = bases_df['y_coordinate'] * cell_size + half_cell

    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1.2], wspace=0.05)
    ax = fig.add_subplot(gs[0])

    node_size = 45 * (50 / np.sqrt(len(df)))

    mask_nonburn = df['state'] == 3
    ax.scatter(df[mask_nonburn]['x_coordinate'], df[mask_nonburn]['y_coordinate'],
               c='#e0e0e0', marker='s', s=node_size, edgecolors='none', alpha=0.6)

    mask_water = df['state'] == 2
    ax.scatter(df[mask_water]['x_coordinate'], df[mask_water]['y_coordinate'],
               c='#0077be', marker='o', s=node_size * 1.2, edgecolors='white', linewidth=0.5)

    mask_data = (df['state'] == 0) | (df['state'] == 1)
    subset = df[mask_data]
    vals = subset[data_col]
    vmin, vmax = np.percentile(vals, 5), np.percentile(vals, 95)
    if vmax <= vmin: vmax = vmin + 1e-9

    sc = ax.scatter(subset['x_coordinate'], subset['y_coordinate'],
                    c=subset[data_col], cmap=cmap, vmin=vmin, vmax=vmax,
                    s=node_size, marker='s', edgecolors='none', alpha=0.9)

    mask_fire = df['state'] == 1
    ax.scatter(df[mask_fire]['x_coordinate'], df[mask_fire]['y_coordinate'],
               s=node_size, facecolors='none', edgecolors='red', linewidth=2, marker='s', zorder=10)

    if road_segments:
        lc = LineCollection(road_segments, colors='black', linewidths=2.5, alpha=1.0, zorder=11, capstyle='butt')
        ax.add_collection(lc)

    base_colors = plt.cm.Set1(np.linspace(0, 1, len(bases_df)))
    for i, (idx, row) in enumerate(bases_df.iterrows()):
        color = base_colors[i]
        ax.scatter(row['phys_x'], row['phys_y'], c=[color], marker='^', s=180, edgecolors='black', zorder=20)
        ax.text(row['phys_x'], row['phys_y'] + (cell_size * 1.3), str(row['Base']),
                fontsize=9, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8), zorder=21)

    ax.set_title(title, fontsize=16, pad=15, fontweight='bold')
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.4)

    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis('off')

    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='red', linewidth=2, label='Yangın Başlangıcı'),
        mpatches.Patch(color='#0077be', label='Su Kaynağı'),
        mpatches.Patch(color='#e0e0e0', label='Yanmayan Alan'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='Müdahale Üssü')
    ]
    if road_segments:
        legend_elements.append(Line2D([0], [0], color='black', lw=2.5, label='Yol (Engel)'))

    leg = ax_leg.legend(handles=legend_elements, loc='upper left', title='Semboller', fontsize=10, frameon=True)
    leg.get_title().set_fontweight('bold')

    cbar_ax = ax_leg.inset_axes([0.05, 0.50, 0.8, 0.03])
    cbar = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(cbar_label, fontsize=9, fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Harita Kaydedildi: {output_path}")


def create_academic_map_from_excel(excel_path: str, out_dir: str):
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    print(f"Haritalama Başlıyor: {excel_path}")

    try:
        inputs_df = pd.read_excel(excel_path, sheet_name="inputs_df")
        bases_df = pd.read_excel(excel_path, sheet_name="bases")
        params_df = pd.read_excel(excel_path, sheet_name="parameters")
    except Exception as e:
        print(f"Excel okuma hatası: {e}")
        return

    unique_x = sorted(inputs_df['x_coordinate'].unique())
    cell_size = unique_x[1] - unique_x[0] if len(unique_x) > 1 else 500
    road_segments = _parse_road_pairs_from_df(params_df, inputs_df, cell_size)

    # Dosya ismini temizle (klasör yolundan kurtar)
    file_stem = os.path.splitext(os.path.basename(excel_path))[0]

    n_nodes = len(inputs_df)
    grid_dim = int(np.sqrt(n_nodes))
    grid_str = f"({grid_dim}x{grid_dim})"

    _plot_single_map(inputs_df, bases_df, road_segments, 'value_at_start', 'Greens',
                     f'Başlangıç Bitki Yoğunluğu {grid_str}', 'Bitki Değeri',
                     os.path.join(out_dir, f"{file_stem}_Map_Value.png"))

    _plot_single_map(inputs_df, bases_df, road_segments, 'fire_degradation_rate', 'Reds',
                     f'Yangın Yayılım Hızı {grid_str}', 'Yayılım Katsayısı',
                     os.path.join(out_dir, f"{file_stem}_Map_Degradation.png"))


# ============================================================
# 5) MAIN
# ============================================================

def main():
    # 1. Config Oku
    cfg = read_scenario_xlsx("scenario.xlsx", "scenario")

    # 2. Veri Üret
    print("Veri üretiliyor...")
    inputs_df = build_inputs_df(cfg)
    assert_neighborhood_rules(inputs_df)

    grid_size = validate_square_grid(cfg["n_nodes"])
    rng = np.random.default_rng(cfg["seed"])

    bases_df = build_bases_df_random(cfg, grid_size, rng, set(cfg["water_nodes"]) | set(cfg["protected_nodes"]))
    params_df = build_parameters_df_fixed(cfg)

    # 3. KLASÖR OLUŞTURMA VE EXCEL KAYDETME (DÜZELTİLEN KISIM)
    INPUTS_DIR = "inputs"
    if not os.path.exists(INPUTS_DIR):
        os.makedirs(INPUTS_DIR)

    # Dosya ismini cfg'den al ama 'inputs/' klasörüne yönlendir
    filename = os.path.basename(cfg["out"])
    out_path = os.path.join(INPUTS_DIR, filename)
    out_path = os.path.abspath(out_path)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        inputs_df.to_excel(writer, sheet_name="inputs_df", index=False)
        bases_df.to_excel(writer, sheet_name="bases", index=False)
        params_df.to_excel(writer, sheet_name="parameters", index=False)
    print(f"Excel üretildi: {out_path}")

    # 4. Haritaları Çiz (inputs/initial_maps altına)
    # out_path zaten 'inputs/dosya.xlsx' olduğu için dirname 'inputs' olur.
    create_academic_map_from_excel(out_path, os.path.join(os.path.dirname(out_path), "initial_maps"))


def assert_neighborhood_rules(df):
    pass


if __name__ == "__main__":
    main()