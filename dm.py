import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import numpy as np
import os

# =============================================================================
# 1. AYARLAR VE DOSYA YOLLARI
# =============================================================================
input_file_path = r'C:\Users\Mert\deprem\wildfire\inputs\inputs_7x7.xlsx'



result_file_path = r'C:\Users\Mert\deprem\wildfire\result\result__inputs_7x7__20260210_150704.xlsx'

OUTPUT_DIR = "maps"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

file_stem = os.path.splitext(os.path.basename(input_file_path))[0]
output_filename = os.path.join(OUTPUT_DIR, f'Academic_Map_DetailedSummary_{file_stem}.png')

# =============================================================================
# 2. VERİ YÜKLEME
# =============================================================================
try:
    print(f"Veriler okunuyor... ({input_file_path})")
    inputs_df = pd.read_excel(input_file_path, sheet_name='inputs_df')
    bases_df = pd.read_excel(input_file_path, sheet_name='bases')
    params_df = pd.read_excel(input_file_path, sheet_name='parameters')

    deployment_df = pd.read_excel(result_file_path, sheet_name='Deployment_Details')
    p_df = pd.read_excel(result_file_path, sheet_name='p')
    y_df = pd.read_excel(result_file_path, sheet_name='y')

    try:
        z_df = pd.read_excel(result_file_path, sheet_name='z')
    except:
        z_df = pd.DataFrame(columns=['i', 'j', 'z_ij'])

except Exception as e:
    print(f"HATA: Dosyalar okunurken sorun oluştu.\n{e}")
    exit()

# =============================================================================
# 3. VERİ HAZIRLIĞI
# =============================================================================

# A. Grid Hesaplamaları
unique_x = sorted(inputs_df['x_coordinate'].unique())
CELL_SIZE = unique_x[1] - unique_x[0] if len(unique_x) > 1 else 500
HALF_CELL = CELL_SIZE / 2
MAX_X = inputs_df['x_coordinate'].max() + HALF_CELL
MAX_Y = inputs_df['y_coordinate'].max() + HALF_CELL
GRID_N = int(MAX_X / CELL_SIZE)

# B. Veri Birleştirme
merged_df = inputs_df.merge(p_df, left_on='node_id', right_on='i', how='left')
merged_df = merged_df.merge(y_df, left_on='node_id', right_on='i', how='left', suffixes=('', '_y'))
merged_df['y_i'] = merged_df['y_i'].fillna(0)
merged_df['p_i'] = merged_df['p_i'].fillna(merged_df['value_at_start'])

# C. Üs Koordinatları
bases_df['phys_x'] = bases_df['x_coordinate'] * CELL_SIZE + HALF_CELL
bases_df['phys_y'] = bases_df['y_coordinate'] * CELL_SIZE + HALF_CELL

# D. Üs Renkleri (Tab10)
base_palette = plt.cm.tab10.colors
base_colors = {}
for i, base_name in enumerate(bases_df['Base'].unique()):
    base_colors[base_name] = base_palette[i % len(base_palette)]

# E. Yollar (Engel)
road_segments = []
try:
    road_pairs_str = params_df.loc[params_df['parameter'] == 'road_pairs', 'value'].values[0]
    if pd.notna(road_pairs_str):
        node_map = inputs_df.set_index('node_id')[['x_coordinate', 'y_coordinate']].to_dict('index')
        pairs = str(road_pairs_str).split(';')
        for pair in pairs:
            if '-' in pair:
                try:
                    n1, n2 = map(int, pair.split('-'))
                    if n1 in node_map and n2 in node_map:
                        x1, y1 = node_map[n1]['x_coordinate'], node_map[n1]['y_coordinate']
                        x2, y2 = node_map[n2]['x_coordinate'], node_map[n2]['y_coordinate']
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        if abs(x1 - x2) > abs(y1 - y2):
                            road_segments.append([(mid_x, mid_y - HALF_CELL), (mid_x, mid_y + HALF_CELL)])
                        else:
                            road_segments.append([(mid_x - HALF_CELL, mid_y), (mid_x + HALF_CELL, mid_y)])
                except:
                    continue
except:
    pass

# F. Yayılım Okları
spread_arrows = []
if 'z_ij' in z_df.columns:
    spread_data = z_df[z_df['z_ij'] > 0.5]
    node_coords = inputs_df.set_index('node_id')[['x_coordinate', 'y_coordinate']].to_dict('index')
    for _, row in spread_data.iterrows():
        src, dst = int(row['i']), int(row['j'])
        if src in node_coords and dst in node_coords:
            p1 = (node_coords[src]['x_coordinate'], node_coords[src]['y_coordinate'])
            p2 = (node_coords[dst]['x_coordinate'], node_coords[dst]['y_coordinate'])
            spread_arrows.append((p1, p2))

# =============================================================================
# 4. GÖRSELLEŞTİRME
# =============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1.4], wspace=0.05)
ax = fig.add_subplot(gs[0])

NODE_SIZE = 45 * (50 / GRID_N) if GRID_N > 0 else 45
if GRID_N <= 25: NODE_SIZE *= 1.6

# --- KATMAN 1: Zemin ---
mask_nonburn = merged_df['state'] == 3
ax.scatter(merged_df[mask_nonburn]['x_coordinate'], merged_df[mask_nonburn]['y_coordinate'],
           c='#f0f0f0', marker='s', s=NODE_SIZE, edgecolors='gray', alpha=0.4)
mask_water = merged_df['state'] == 2
ax.scatter(merged_df[mask_water]['x_coordinate'], merged_df[mask_water]['y_coordinate'],
           c='#4682B4', marker='o', s=NODE_SIZE, edgecolors='white', linewidth=0.5)

# --- KATMAN 2: Bitki ve Yangın ---
mask_safe = (merged_df['y_i'] < 0.5) & (merged_df['state'].isin([0, 1]))
safe_nodes = merged_df[mask_safe]
scatter_safe = ax.scatter(safe_nodes['x_coordinate'], safe_nodes['y_coordinate'],
                          c=safe_nodes['value_at_start'], cmap='YlGn', vmin=0, vmax=100,
                          s=NODE_SIZE, alpha=0.9, marker='s')

burned_nodes = merged_df[merged_df['y_i'] > 0.5]
mask_destroyed = burned_nodes['p_i'] < 10
ax.scatter(burned_nodes[mask_destroyed]['x_coordinate'], burned_nodes[mask_destroyed]['y_coordinate'],
           c='black', marker='s', s=NODE_SIZE, alpha=1.0)
mask_damaged = burned_nodes['p_i'] >= 10
ax.scatter(burned_nodes[mask_damaged]['x_coordinate'], burned_nodes[mask_damaged]['y_coordinate'],
           c='#D2691E', marker='s', s=NODE_SIZE, alpha=0.9)

# Başlangıç Noktaları
mask_start = merged_df['state'] == 1
ax.scatter(merged_df[mask_start]['x_coordinate'], merged_df[mask_start]['y_coordinate'],
           s=NODE_SIZE, facecolors='none', edgecolors='#8B0000', linewidth=3, marker='s', zorder=10)

# Etiketler
if GRID_N <= 25:
    for idx, row in merged_df.iterrows():
        if row['state'] in [0, 1]:
            txt_color = 'white' if (row['y_i'] > 0.5 and row['p_i'] < 10) else 'black'
            label = f"{int(row['node_id'])}\n{row['p_i']:.0f}"
            ax.text(row['x_coordinate'], row['y_coordinate'], label,
                    ha='center', va='center', fontsize=9, color=txt_color, fontweight='bold', zorder=11)

# --- KATMAN 3: Yollar ---
if road_segments:
    lc = LineCollection(road_segments, colors='black', linewidths=3, alpha=1.0, zorder=12, capstyle='butt')
    ax.add_collection(lc)

# --- KATMAN 4: Rotalar ---
dep_merged = deployment_df.merge(bases_df[['Base', 'phys_x', 'phys_y']], on='Base', how='left')
dep_merged = dep_merged.merge(inputs_df[['node_id', 'x_coordinate', 'y_coordinate']],
                              left_on='Target_Node', right_on='node_id', how='left', suffixes=('_base', '_target'))

for idx, row in dep_merged.iterrows():
    base_name = row['Base']
    route_color = base_colors.get(base_name, 'gray')
    ax.plot([row['phys_x'], row['x_coordinate']], [row['phys_y'], row['y_coordinate']],
            color=route_color, linestyle='--', linewidth=2, alpha=0.8, zorder=13)

# --- KATMAN 5: Yayılım Okları ---
for (p1, p2) in spread_arrows:
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="->", color="#DC143C", lw=2.5, ls='-', alpha=0.9), zorder=14)

# --- KATMAN 6: Üsler ---
for i, (idx, row) in enumerate(bases_df.iterrows()):
    base_name = row['Base']
    color = base_colors.get(base_name, 'gray')
    ax.scatter(row['phys_x'], row['phys_y'], c=[color], marker='o', s=600,
               edgecolors='black', linewidth=1.5, zorder=25)
    ax.text(row['phys_x'], row['phys_y'], str(base_name),
            fontsize=14, fontweight='bold', ha='center', va='center', color='white', zorder=26)

# Ayarlar
ax.set_title(f'Multi-Objective Fire Response Analysis ({GRID_N}x{GRID_N})', fontsize=18, fontweight='bold', pad=20)
ax.set_xlim(0, MAX_X)
ax.set_ylim(0, MAX_Y)
ax.set_aspect('equal')
ax.set_xlabel('X Coordinate (m)', fontsize=14)
ax.set_ylabel('Y Coordinate (m)', fontsize=14)
if GRID_N > 25:
    ax.grid(True, linestyle=':', alpha=0.4)
else:
    ax.grid(False)

# =============================================================================
# 5. SAĞ PANEL (LEJANT & DETAYLI İSTATİSTİK)
# =============================================================================
ax_leg = fig.add_subplot(gs[1])
ax_leg.axis('off')

# Lejant
legend_elements = [
    mpatches.Patch(facecolor='none', edgecolor='#8B0000', linewidth=3, label='Ignition Point'),
    mpatches.Patch(color='black', label='Destroyed (Ash)'),
    mpatches.Patch(color='#D2691E', label='Damaged / Active'),
    mpatches.Patch(color='#f0f0f0', label='Non-burnable Area'),
    mpatches.Patch(color='#4682B4', label='Water Source'),
    Line2D([0], [0], color='#DC143C', lw=2.5, marker='>', label='Fire Spread (Vector)'),
    Line2D([0], [0], color='black', lw=3, label='Road Barrier'),
]
legend_elements.append(
    Line2D([0], [0], marker='o', color='w', label='Bases & Routes:', markerfacecolor='w', markersize=0))
for base_name, color in base_colors.items():
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Base {base_name}',
                                  markerfacecolor=color, markeredgecolor='black', markersize=12))

leg = ax_leg.legend(handles=legend_elements, loc='upper left', title='Map Legend', fontsize=10, frameon=True,
                    edgecolor='black')
leg.get_title().set_fontweight('bold')

cbar_ax = ax_leg.inset_axes([0.05, 0.45, 0.9, 0.03])
cbar = plt.colorbar(scatter_safe, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Vegetation Density', fontsize=11, fontweight='bold')

# --- GÜNCELLENEN İSTATİSTİK KISMI ---
usage_counts = deployment_df.groupby(['Base', 'Type'])['Vehicle_ID'].nunique().unstack(fill_value=0)

stats_text = f"SUMMARY\n{'-' * 28}\n"
stats_text += f"Burned Nodes: {len(burned_nodes)}\n"
stats_text += f"Spread Events: {len(spread_arrows)}\n"
stats_text += f"{'-' * 28}\nVEHICLE USAGE (Used/Total):\n"

# Her üs için toplam ve kullanılan araç sayılarını hesapla
for i, (idx, row) in enumerate(bases_df.iterrows()):
    b = row['Base']

    # Kullanılan (Deployment dosyasından)
    used_h = usage_counts.loc[
        b, 'Helicopter'] if 'Helicopter' in usage_counts.columns and b in usage_counts.index else 0
    used_e = usage_counts.loc[
        b, 'Fire Engine'] if 'Fire Engine' in usage_counts.columns and b in usage_counts.index else 0
    used_f = usage_counts.loc[b, 'FRV'] if 'FRV' in usage_counts.columns and b in usage_counts.index else 0

    # Toplam (Base dosyasından)
    total_h = row['Helicopter']
    total_e = row['Fire Engine']
    total_f = row['FRV']

    # Format: [A] H:2/2 E:2/3 F:2/2
    # Sadece envanterinde araç olan üsleri göster
    if total_h + total_e + total_f > 0:
        line = f"[{b}] "
        if total_h > 0: line += f"H:{used_h}/{total_h} "
        if total_e > 0: line += f"E:{used_e}/{total_e} "
        if total_f > 0: line += f"F:{used_f}/{total_f}"
        stats_text += line + "\n"

# İstatistik kutusu
ax_leg.text(0.0, 0.05, stats_text, transform=ax_leg.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='bottom',
            bbox=dict(boxstyle="square,pad=0.5", fc="#ffffff", ec="black", alpha=1.0, lw=0.5))

plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Harita kaydedildi: {output_filename}")
plt.show()