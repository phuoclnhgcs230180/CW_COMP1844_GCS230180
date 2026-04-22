import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# --- 1. Data Structure (Spaced out layout) ---
line_colors = {
    'East West Line': '#009530', 'North South Line': '#dc241f', 'North East Line': '#9016b2', 
    'Circle Line': '#ff9a00', 'Downtown Line': '#0054b4', 'Thomson-East Coast Line': '#9d5b25'
}

data = {
'East West Line': ['Dover','Buona Vista','Commonwealth','Queenstown','Redhill','Tiong Bahru','Outram Park','Tanjong Pagar'],
    'North South Line': ['Yishun','Khatib','Yio Chu Kang','Ang Mo Kio','Bishan','Braddell'],
    'North East Line': ['HarbourFront','Outram Park','Chinatown','Clarke Quay','Dhoby Ghaut'],
    'Circle Line': ['Buona Vista','Holland Village','Farrer Road','Botanic Gardens','Caldecott','Marymount','Bishan','Lorong Chuan'],
    'Downtown Line': ['Tan Kah Kee','Botanic Gardens','Stevens','Newton','Little India'],
    'Thomson-East Coast Line': ['Bright Hill','Upper Thomson','Caldecott','Stevens','Napier'],
}
# --- 2.Nodes ---
pos = {
    'Ang Mo Kio': (8.47, 4.67),
    'Bishan': (8.47, 7.07),
    'Botanic Gardens': (3.73, 2.33),
    'Braddell': (8.47, 8.27),
    'Bright Hill': (6.49, 1.49),
    'Buona Vista': (1.40, 0.00),
    'Caldecott': (6.49, 5.09),
    'Chinatown': (8.29, 0.49),
    'Clarke Quay': (8.72, 0.92),
    'Commonwealth': (2.50, 0.00),
    'Dhoby Ghaut': (9.71, 1.91),
    'Dover': (0.00, 0.00),
    'Farrer Road': (3.03, 1.63),
    'HarbourFront': (5.96, -1.84),
    'Holland Village': (2.04, 0.64),
    'Khatib': (8.47, -1.73),
    'Little India': (7.83, 2.33),
    'Lorong Chuan': (9.67, 8.27),
    'Marymount': (7.34, 5.94),
    'Napier': (4.83, 4.03),
    'Newton': (6.43, 2.33),
    'Outram Park': (7.80, 0.00),
    'Queenstown': (3.70, 0.00),
    'Redhill': (5.10, 0.00),
    'Stevens': (4.83, 2.33),
    'Tan Kah Kee': (2.63, 2.33),
    'Tanjong Pagar': (8.80, 0.00),
    'Tiong Bahru': (6.30, 0.00),
    'Upper Thomson': (6.49, 2.89),
    'Yio Chu Kang': (8.47, 3.17),
    'Yishun': (8.47, -3.13),
}

# --- 3. Custom Label Coordinates (To prevent overlap) ---
label_pos = {
    'Ang Mo Kio': (8.8, 4.67),
    'Bishan': (8.7, 7.07),
    'Botanic Gardens': (3.5, 2.6),
    'Braddell': (8.47, 8.5),
    'Bright Hill': (6.49, 1.2),
    'Buona Vista': (1.40, -0.30),
    'Caldecott': (6.49, 5.39),
    'Chinatown': (8.19, 0.79),
    'Clarke Quay': (8.72, 1.22),
    'Commonwealth': (2.50, -0.30),
    'Dhoby Ghaut': (9.71, 2.21),
    'Dover': (0.00, -0.30),
    'Farrer Road': (3.03, 1.33),
    'HarbourFront': (5.96, -2.14),
    'Holland Village': (2.04, 0.94),
    'Khatib': (8.77, -1.73),
    'Little India': (7.83, 2.63),
    'Lorong Chuan': (9.67, 8.57),
    'Marymount': (7.34, 6.24),
    'Napier': (4.83, 4.33),
    'Newton': (6.23, 2.03),
    'Outram Park': (7.9, -0.30),
    'Queenstown': (3.70, -0.30),
    'Redhill': (5.10, -0.30),
    'Stevens': (4.83, 2.03),
    'Tan Kah Kee': (2.63, 2.63),
    'Tanjong Pagar': (8.80, -0.30),
    'Tiong Bahru': (6.30, -0.30),
    'Upper Thomson': (6.89, 2.89),
    'Yio Chu Kang': (8.77, 3.17),
    'Yishun': (8.47, -3.43),
}


# Exact colors to regular stations
station_to_line = {}
for line, stations in data.items():
    for station in stations:
        if station not in station_to_line:
            station_to_line[station] = line_colors[line]

# --- 4. Build Graph & Calculate Distance ---

manual_distance_overrides = {
    
    ('Dover', 'Buona Vista'): 1.4,('Buona Vista', 'Commonwealth'): 1.1,('Commonwealth', 'Queenstown'): 1.2,('Queenstown', 'Redhill'): 1.4,('Redhill', 'Tiong Bahru'): 1.2,
    ('Tiong Bahru', 'Outram Park'): 1.5,('Outram Park', 'Tanjong Pagar'): 1.0,
    ('Yishun', 'Khatib'): 1.4,('Khatib', 'Yio Chu Kang'): 4.9,('Yio Chu Kang', 'Ang Mo Kio'): 1.5,('Ang Mo Kio', 'Bishan'): 2.4,('Bishan', 'Braddell'): 1.2,
    ('HarbourFront', 'Outram Park'): 2.6,('Outram   Park', 'Chinatown'): 0.7,('Chinatown', 'Clarke Quay'): 0.6,('Clarke Quay', 'Dhoby Ghaut'): 1.4,
    ('Buona Vista', 'Holland Village'): 0.9,('Holland   Village', 'Farrer Road'): 1.4,('Farrer Road', 'Botanic Gardens'): 1.0,('Botanic Gardens','Caldecott'): 3.9 ,('Caldecott', 'Marymount'): 1.2,('Marymount', 'Bishan'): 1.6,('Bishan', 'Lorong Chuan'): 1.7,
    ('Tan Kah Kee', 'Botanic Gardens'): 1.1,('Botanic Gardens', 'Stevens'): 1.1,('Stevens', 'Newton'): 1.6,('Newton', 'Little India'): 1.4,
    ('Bright Hill', 'Upper Thomson'): 1.4,('Upper Thomson', 'Caldecott'): 2.2,('Caldecott','Stevens'): 3.0,('Stevens', 'Napier'): 1.7,
}

G = nx.Graph()
SCALE_FACTOR = 0.5

for line, stations in data.items():
    for i in range(len(stations) - 1):
        u, v = stations[i], stations[i+1]
        default_dist = np.linalg.norm(np.array(pos[u]) - np.array(pos[v])) * SCALE_FACTOR
        manual_key = (u, v) if (u, v) in manual_distance_overrides else (v, u)
        dist = manual_distance_overrides.get(manual_key, round(default_dist, 1))

        # Specify source line, color and final distance.
        G.add_edge(u, v, color=line_colors[line], line=line, distance=round(dist, 1))

# --- 5. Interactive Terminal UI ---
print("\n" + "="*45)
print(" WELCOME TO THE SINGAPORE MRT MAP ")
print("="*45)

# 5.1 Select Measurement Unit
print("\n[SELECT DISTANCE UNIT]")
print("1. Kilometers (km)")
print("2. Miles (miles)")
while True:
    unit_choice = input(" Enter number (1 or 2) [Default: 1]: ").strip()
    if unit_choice in ['1', '']:
        conversion, unit_label = 1, 'km'
        break
    elif unit_choice == '2':
        conversion, unit_label = 0.621371, 'miles'
        break
    else:
        print(" Invalid choice. Please enter 1 or 2.")

highlight = False
path_edges = []
total_distance = 0
start, end = "", ""
station_list = sorted(list(G.nodes()))

# 5.2 Select Map Mode
print("\n[DISPLAY MODE]")
route_choice = input(" Do you want to find the shortest path between 2 stations? (Y/N) [Default: N]: ").strip().upper()

if route_choice == 'Y':
    print("\n[MRT STATIONS LIST]")
    half_len = (len(station_list) + 1) // 2
    col1 = [f"{i+1:02d}. {station_list[i]}" for i in range(half_len)]
    col2 = [f"{i+1+half_len:02d}. {station_list[i+half_len]}" if i+half_len < len(station_list) else "" for i in range(half_len)]
    
    df_menu = pd.DataFrame({'Col 1': col1, 'Col 2': col2})
    print(df_menu.to_string(index=False, header=False))

    print("\n[FIND SHORTEST PATH]")
    try:
        start_idx = int(input(" Enter the NUMBER of the STARTING station (e.g. 1, 15...): ")) - 1
        end_idx = int(input(" Enter the NUMBER of the DESTINATION station: ")) - 1

        if 0 <= start_idx < len(station_list) and 0 <= end_idx < len(station_list):
            start = station_list[start_idx]
            end = station_list[end_idx]
            
            if start == end:
                print(" Start and destination are the same. Displaying the full map instead.")
            else:
                try:
                    path = nx.shortest_path(G, source=start, target=end, weight='distance')
                    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                    total_distance = sum(G[u][v]['distance'] for u, v in path_edges)
                    average_distance = total_distance / len(path_edges) if path_edges else 0

                    print("\n" + "-"*40)
                    print(f" ROUTE FOUND:")
                    print(f" Start: {start}")
                    print(f" Destination: {end}")
                    print(f" Path: {' ➔ '.join(path)}")

                    print("\n Segment distances:")
                    for u, v in path_edges:
                        d = G[u][v]['distance'] * conversion
                        print(f"  {u} -> {v}: {d:.2f} {unit_label}")

                    print(f"\n Total distance: {total_distance * conversion:.2f} {unit_label}")
                    print(f" Average segment distance: {average_distance * conversion:.2f} {unit_label}")
                    print("-"*40 + "\n")

                    highlight = True
                except nx.NetworkXNoPath:
                    print(" No connecting path between these stations.")
        else:
            print(" Invalid number. Displaying the full map instead.")
            
    except ValueError:
        print(" Error: You must enter a NUMBER. Displaying the full map instead.")
else:
    network_total = sum(d['distance'] for _, _, d in G.edges(data=True))
    network_avg = network_total / G.number_of_edges() if G.number_of_edges() > 0 else 0
    print("\n Loading full MRT network map...")
    print(f" Network total distance: {network_total * conversion:.2f} {unit_label}")
    print(f" Network average per edge: {network_avg * conversion:.2f} {unit_label}")


# --- 6. Map Rendering Configuration ---
fig, ax = plt.subplots(figsize=(17, 10.5), facecolor='#f8f9fa')

# Draw Edges (Lines)
for u, v, d in G.edges(data=True):
    is_in_path = (u, v) in path_edges or (v, u) in path_edges
    alpha = 0.9 if not highlight or is_in_path else 0.15
    width = 6 if highlight and is_in_path else 3
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v)], edge_color=d['color'], width=width, alpha=alpha, ax=ax
    )

# Draw Edge Labels (Distance + Unit)
edge_labels = {}
for u, v, d in G.edges(data=True):
    dist = d['distance'] * conversion
    edge_labels[(u, v)] = f"{dist:.1f} {unit_label}"
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, font_color='#4a4a4a', ax=ax, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# Process Nodes
interchanges = [n for n in G.nodes() if G.degree(n) > 2]
regular_stations = [n for n in G.nodes() if G.degree(n) <= 2]

# Get colors for regular stations based on their line
reg_node_colors = [station_to_line[n] for n in regular_stations]

# Draw Regular Stations 
nx.draw_networkx_nodes(
    G, pos, nodelist=regular_stations, node_size=100, 
    node_color=reg_node_colors, edgecolors='black', linewidths=1.5, ax=ax
)

# Draw Interchange Stations 
nx.draw_networkx_nodes(
    G, pos, nodelist=interchanges, node_size=100, 
    node_color='white', edgecolors='black', linewidths=2.5, ax=ax
)

# Draw Node Labels 
nx.draw_networkx_labels(
    G, label_pos, labels={n: n for n in regular_stations}, 
    font_size=7, font_color='#333333', font_family='sans-serif', 
    horizontalalignment='center', verticalalignment='center'
)
nx.draw_networkx_labels(
    G, label_pos, labels={n: n for n in interchanges}, 
    font_size=8, font_color='black', 
    horizontalalignment='center', verticalalignment='center'
)

# Legend Configuration
from matplotlib.lines import Line2D
# 1. Create lines for MRT Lines
line_elements = [Line2D([0], [0], color=c, lw=4, label=k) for k, c in line_colors.items()]

# 2. Create symbol for Interchange Station
interchange_legend = Line2D([0], [0], marker='o', color='none', label='Interchange Station',
                            markerfacecolor='white', markeredgecolor='black', 
                            markersize=10, markeredgewidth=2.5)

# 3. Create symbol for Regular Station
regular_legend = Line2D([0], [0], marker='o', color='none', label='Regular Station',
                        markerfacecolor='#888888', markeredgecolor='black', 
                        markersize=8, markeredgewidth=1.5)

# Combine all together
legend_elements = line_elements + [interchange_legend, regular_legend]

# Draw Legend
ax.legend(handles=legend_elements, loc='upper left', title="MRT Network Legend", 
          frameon=True, fontsize=10, title_fontsize=12)

# Title 
title_text = f"SINGAPORE MRT MAP (Distance Unit: {unit_label.upper()})"
plt.title(title_text, fontsize=18, fontweight='bold', pad=20, color='#1a1a1a')

# Route Info Box 
if highlight:
    plt.text(0.5, 0.96, f"Route: {start} ➔ {end} | Total Distance: {total_distance * conversion:.2f} {unit_label}", 
             transform=ax.transAxes, fontsize=13, ha='center', color='#d32f2f', fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#d32f2f', pad=6, boxstyle='round,pad=0.5'))

plt.axis('off')
plt.tight_layout()
plt.show()
