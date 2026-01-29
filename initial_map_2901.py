import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.patheffects as path_effects
import os
import datetime
import numpy as np

# --- Konfigürasyon ---
input_file_path = r"C:\Users\Mert\deprem\wildfire\inputs_new_6x6.xlsx"
output_folder = r"C:\Users\Mert\deprem\wildfire\initial map"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Veriyi Yükle
try:
    df = pd.read_excel(input_file_path, sheet_name='inputs_df')
    bases_df = pd.read_excel(input_file_path, sheet_name='bases')
except Exception as e:
    print(f"Hata: {e}")
    raise

# Matrisleri Hazırla
value_matrix = df.pivot(index='y_coordinate', columns='x_coordinate', values='value_at_start').sort_index(
    ascending=True)
degradation_matrix = df.pivot(index='y_coordinate', columns='x_coordinate', values='fire_degradation_rate').sort_index(
    ascending=True)
state_matrix = df.pivot(index='y_coordinate', columns='x_coordinate', values='state').sort_index(ascending=True)
id_matrix = df.pivot(index='y_coordinate', columns='x_coordinate', values='node_id').sort_index(ascending=True)

rows, cols = value_matrix.shape
base_colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange', 'purple', 'pink', 'brown']

# --- Yol Çiftleri (Değişmedi) ---
road_pairs = [
    (2, 3), (3, 9), (4, 10), (14, 15),
    (20, 21), (20, 26), (18, 24), (17, 23)
]


def get_shared_edge(id1, id2):
    """İki hücre komşu ise aralarındaki kenar koordinatlarını döndürür."""
    row1 = df[df['node_id'] == id1].iloc[0]
    row2 = df[df['node_id'] == id2].iloc[0]

    x1, y1 = row1['x_coordinate'], row1['y_coordinate']
    x2, y2 = row2['x_coordinate'], row2['y_coordinate']

    dx = abs(x1 - x2)
    dy = abs(y1 - y2)

    if abs(dx - 1.0) < 1e-5 and abs(dy) < 1e-5:  # Dikey Kenar
        ex = (x1 + x2) / 2
        ey_start = y1 - 0.5
        ey_end = y1 + 0.5
        return [(ex, ey_start), (ex, ey_end)]

    elif abs(dx) < 1e-5 and abs(dy - 1.0) < 1e-5:  # Yatay Kenar
        ey = (y1 + y2) / 2
        ex_start = x1 - 0.5
        ex_end = x1 + 0.5
        return [(ex_start, ey), (ex_end, ey)]

    return None


def create_final_academic_map(data_matrix, map_type, cmap, filename_prefix, save_path):
    # Sınırların Hesaplanması
    bases_min_x = bases_df['x_coordinate'].min()
    bases_max_x = bases_df['x_coordinate'].max()
    bases_min_y = bases_df['y_coordinate'].min()
    bases_max_y = bases_df['y_coordinate'].max()

    min_x = min(0, bases_min_x)
    max_x = max(cols, bases_max_x)
    min_y = min(0, bases_min_y)
    max_y = max(rows, bases_max_y)

    pad = 2
    xlims = (min_x - pad, max_x + pad)
    ylims = (min_y - pad, max_y + pad)

    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(10, 10))
    title = "Value at Start Map" if map_type == 'value' else "Fire Degradation Rate Map"

    # Isı Haritası
    im = ax.imshow(data_matrix, cmap=cmap, origin='lower', extent=[0, cols, 0, rows])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(title, fontsize=12)

    # 6x6 Alan Çerçevesi
    rect_6x6 = patches.Rectangle((0, 0), cols, rows, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect_6x6)

    # İç Izgaralar
    for x in range(1, cols):
        ax.plot([x, x], [0, rows], color='black', linewidth=0.5, alpha=0.5)
    for y in range(1, rows):
        ax.plot([0, cols], [y, y], color='black', linewidth=0.5, alpha=0.5)

    # --- Yolları Çiz (Siyah ve Daha İnce) ---
    for (id1, id2) in road_pairs:
        edge = get_shared_edge(id1, id2)
        if edge:
            (ex1, ey1), (ex2, ey2) = edge
            # Siyah renk, kalınlık 4
            ax.plot([ex1, ex2], [ey1, ey2], color='black', linewidth=4, zorder=5)

    # Mesafe Etiketleri
    label_style = dict(fontsize=9, color='black', ha='center', va='center', fontweight='bold')
    for x in range(cols + 1):
        ax.text(x, -0.6, f"{x}", **label_style)
    for y in range(rows + 1):
        ax.text(-0.6, y, f"{y}", **label_style)
    ax.text(-0.6, -0.6, "km", fontsize=9, ha='right', va='top', fontweight='bold')

    # Üsler (Bases)
    base_handles = []
    for idx, row in bases_df.iterrows():
        base_name = str(row['Base']).strip()
        bx = row['x_coordinate']
        by = row['y_coordinate']
        color = base_colors[idx % len(base_colors)]

        ax.scatter(bx, by, s=300, marker='s', color=color, edgecolors='black', linewidth=1.5, zorder=10)

        txt = ax.text(bx, by, base_name, ha='center', va='center',
                      color='black', fontsize=10, fontweight='bold', zorder=11)
        txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

        base_handles.append(mlines.Line2D([], [], color=color, marker='s', linestyle='None',
                                          markersize=10, markeredgewidth=1.5, markeredgecolor='black',
                                          label=f"Base {base_name}"))

    # Hücre İçerikleri
    for i in range(rows):
        for j in range(cols):
            st = state_matrix.iloc[i, j]
            val = data_matrix.iloc[i, j]
            node_id = int(id_matrix.iloc[i, j])
            cx, cy = j + 0.5, i + 0.5

            if st == 2:
                rect = patches.Rectangle((j, i), 1, 1, linewidth=0, facecolor='blue')
                ax.add_patch(rect)
            elif st == 3:
                rect = patches.Rectangle((j, i), 1, 1, linewidth=0, facecolor='gray')
                ax.add_patch(rect)

            if map_type == 'value' and st == 1:
                ax.text(cx, cy, 'X', color='red', ha='center', va='center', fontsize=24, fontweight='bold')

            text_color = 'white' if st in [2, 3] else 'black'
            ax.text(j + 0.1, i + 0.85, str(node_id), ha='left', va='center', color=text_color, fontsize=8,
                    fontweight='bold')

            if st not in [2, 3]:
                val_str = f"{val:.1f}" if map_type == 'value' else f"{val:.2f}"
                ax.text(cx, cy, val_str, ha='center', va='center', color='black', fontsize=10)

    # Lejant
    legend_elements = []
    if map_type == 'value':
        legend_elements.append(mlines.Line2D([], [], color='red', marker='x', linestyle='None',
                                             markersize=10, markeredgewidth=2, label='Initial Fire'))
    legend_elements.append(patches.Patch(facecolor='blue', label='Water Source'))
    legend_elements.append(mlines.Line2D([], [], color='black', linewidth=4, label='Inner Forest Road'))

    full_legend = legend_elements + base_handles
    ax.legend(handles=full_legend, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=False, frameon=True, ncol=4, fontsize=10)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_axis_off()

    # Kaydet
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Initial Map_{map_type}_{timestamp}.png"
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Kaydedildi: {full_path}")


# --- İşlemi Başlat ---
f1 = create_final_academic_map(value_matrix, 'value', 'Greens', 'final_academic_value', output_folder)
f2 = create_final_academic_map(degradation_matrix, 'degradation', 'Reds', 'final_academic_degradation', output_folder)
print("İşlem tamamlandı.")